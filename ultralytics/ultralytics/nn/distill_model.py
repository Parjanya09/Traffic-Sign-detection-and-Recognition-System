# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect, RTDETRDecoder
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class DINOHead(nn.Module):
    """Small convolutional projection head used for DINO-style feature distillation."""

    def __init__(self, in_dim, hidden_dim=512, bottleneck_dim=256, out_dim=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=1),
        )

        self.last_layer = nn.Conv2d(
            bottleneck_dim,
            out_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        x = F.normalize(x, dim=1)
        return self.last_layer(x).flatten(1)


class FeatureHook:
    """Picklable forward hook that stores a module output in a shared dictionary."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """
    Teacher-student wrapper supporting:

    1. YOLO Detect-family models such as YOLO26.
       - Feature distillation from the Detect input features.
       - Optional YOLO head score-weighted feature distillation.
       - DINO-style feature distillation.

    2. RT-DETR models.
       - Feature distillation from the three multiscale feature maps
         feeding RTDETRDecoder.
       - DINO-style feature distillation.
       - No YOLO-specific one2many/one2one score distillation.

    Important:
        For RT-DETR the final RTDETRDecoder output is NOT treated as a YOLO
        Detect head. Its multiscale decoder inputs are used instead.

    The returned training loss includes both the normal detection loss and
    the enabled distillation losses.
    """

    def __init__(
        self,
        teacher_model: str | Path | nn.Module | None,
        student_model: nn.Module,
    ):
        super().__init__()

        ch = student_model.yaml.get("channels", 3)
        device = next(student_model.parameters()).device

        self.student_model = student_model
        self.student_model.train()
        self.student_model.requires_grad_(True)

        self.self_distill = bool(student_model.args.get("dino_distill", False))

        # If dino_distill=True and no external teacher is supplied, use an
        # EMA teacher initialized as an exact copy of the student.
        if self.self_distill:
            # Keep the EMA teacher outside nn.Module._modules. Ultralytics'
            # optimizer builder may otherwise include frozen teacher params.
            object.__setattr__(
                self,
                "teacher_model",
                deepcopy(student_model).to(device),
            )
        else:
            if teacher_model is None:
                raise ValueError(
                    "A teacher checkpoint/model is required when dino_distill=False."
                )

            if isinstance(teacher_model, (str, Path)):
                teacher_model = load_checkpoint(teacher_model)[0]

                # Keep the input channel count compatible with the student.
                if teacher_model.yaml.get("channels", 3) != ch:
                    weights = teacher_model
                    teacher_model = type(weights)(
                        weights.yaml.copy(),
                        ch=ch,
                        nc=weights.yaml["nc"],
                        verbose=False,
                    )
                    teacher_model.load(weights)

            object.__setattr__(self, "teacher_model", teacher_model.to(device))

        self.model_type = self._detect_model_type(student_model)

        self._freeze_teacher()

        # Feature layers differ between YOLO Detect and RT-DETR.
        self.feats_idx = self.get_distill_layers(student_model)

        self._teacher_feats: dict[int, object] = {}
        self._student_feats: dict[int, object] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []

        self._register_feature_hooks()

        # ------------------------------------------------------------
        # Infer feature dimensions with a dummy forward.
        # ------------------------------------------------------------
        imgsz = student_model.args.get("imgsz", 640)
        if isinstance(imgsz, (list, tuple)):
            imgsz_h, imgsz_w = int(imgsz[0]), int(imgsz[1])
        else:
            imgsz_h = imgsz_w = int(imgsz)

        student_model.eval()

        with torch.no_grad():
            im = torch.zeros(
                2,
                ch,
                imgsz_h,
                imgsz_w,
                device=device,
            )

            self._teacher_feats.clear()
            self._student_feats.clear()

            self.teacher_model(im)
            student_model(im)

        student_model.train()

        teacher_output = [self._as_feature(self._teacher_feats[idx]) for idx in self.feats_idx]
        student_output = [self._as_feature(self._student_feats[idx]) for idx in self.feats_idx]

        copy_attr(self, student_model)

        # Main feature-distillation weight.
        self.dis = float(self.student_model.args.get("dis", 6.0))

        # DINO loss weight. If not explicitly supplied, keep it modest so
        # detection loss remains dominant.
        self.dino_weight = float(
            self.student_model.args.get("dino_weight", 1.0)
        )

        # ------------------------------------------------------------
        # Feature projectors.
        # Every selected layer is a real [B,C,H,W] feature map.
        # ------------------------------------------------------------
        projectors = []

        # Both YOLO and RT-DETR use all selected feature maps for feature KD.
        # The detection head/decoder itself is deliberately excluded from
        # self.feats_idx, so there is no need for a special final-layer case.
        projector_pairs = zip(student_output, teacher_output)

        for student_out, teacher_out in projector_pairs:
            student_dim = student_out.shape[1]
            teacher_dim = teacher_out.shape[1]

            if student_dim == teacher_dim:
                projectors.append(nn.Identity())
            else:
                projectors.append(
                    nn.Sequential(
                        nn.Conv2d(
                            student_dim,
                            teacher_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            teacher_dim,
                            teacher_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                    )
                )

        self.projector = nn.ModuleList(projectors).to(device)

        # ------------------------------------------------------------
        # DINO feature selection.
        # YOLO26:  [16, 19, 22] -> layer 22 (deepest Detect input)
        # RT-DETR: [21, 24, 27] -> layer 24 (middle multiscale feature)
        # ------------------------------------------------------------
        if self.model_type == "yolo":
            dino_index = len(student_output) - 1
        elif self.model_type == "rtdetr":
            dino_index = len(student_output) // 2
        else:
            raise ValueError(
                f"Unsupported model type for DINO: {self.model_type}"
            )

        dino_student_feat = student_output[dino_index]
        dino_teacher_feat = teacher_output[dino_index]

        self.dino_feature_idx = self.feats_idx[dino_index]

        self.dino_student_projector = DINOHead(
            dino_student_feat.shape[1],
            hidden_dim=512,
            bottleneck_dim=256,
            out_dim=1024,
        ).to(device)

        _dino_teacher_projector = DINOHead(
            dino_teacher_feat.shape[1],
            hidden_dim=512,
            bottleneck_dim=256,
            out_dim=1024,
        ).to(device)

        _dino_teacher_projector.load_state_dict(
            self.dino_student_projector.state_dict()
        )

        object.__setattr__(self, "dino_teacher_projector", _dino_teacher_projector)

        for param in self.dino_teacher_projector.parameters():
            param.requires_grad = False

        self.register_buffer(
            "dino_center",
            torch.zeros(1, 1024, device=device),
        )

        self.dino_center_momentum = 0.9

    # ------------------------------------------------------------------
    # Model detection / feature selection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_model_type(model: nn.Module) -> str:
        """Return 'yolo' or 'rtdetr' based on the final detection module."""

        for m in model.model:
            if isinstance(m, RTDETRDecoder):
                return "rtdetr"

        for m in model.model:
            if isinstance(m, Detect):
                return "yolo"

        raise ValueError(
            "Unsupported model architecture. Expected a YOLO Detect-family "
            "head or an RTDETRDecoder."
        )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """
        Find feature layers used for distillation.

        YOLO:
            Uses the feature maps feeding Detect. The Detect head itself is
            deliberately excluded because its training output is a tuple,
            not a [B,C,H,W] feature tensor.

        RT-DETR:
            Uses the multiscale feature maps feeding RTDETRDecoder.
            The decoder itself is deliberately not hooked because its output
            is query-based rather than a [B,C,H,W] feature map.
        """

        for m in model.model:
            if isinstance(m, Detect):
                # m.f are the multiscale feature maps consumed by Detect.
                # Do NOT append m.i: the Detect head output is not a plain
                # [B,C,H,W] feature map during training.
                return list(m.f)

            if isinstance(m, RTDETRDecoder):
                # RT-DETR decoder receives multiscale feature maps through f.
                feature_indices = list(m.f)

                if len(feature_indices) < 2:
                    raise ValueError(
                        "RTDETRDecoder must receive at least two feature maps "
                        "for feature distillation."
                    )

                return feature_indices

        raise ValueError(
            "No supported Detect or RTDETRDecoder head found in model."
        )

    # ------------------------------------------------------------------
    # Teacher handling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self):
        """
        EMA-update the self-distillation teacher.

        For an externally supplied teacher checkpoint, this does nothing.
        """

        if not self.self_distill or self.teacher_model is None:
            return

        momentum = float(
            self.student_model.args.get("distill_momentum", 0.996)
        )

        for teacher_param, student_param in zip(
            self.teacher_model.parameters(),
            self.student_model.parameters(),
        ):
            teacher_param.mul_(momentum).add_(
                student_param,
                alpha=1.0 - momentum,
            )

        # Copy buffers (BN running statistics etc.).
        for teacher_buffer, student_buffer in zip(
            self.teacher_model.buffers(),
            self.student_model.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

        # EMA-update the DINO teacher projection head too.
        for teacher_param, student_param in zip(
            self.dino_teacher_projector.parameters(),
            self.dino_student_projector.parameters(),
        ):
            teacher_param.mul_(momentum).add_(
                student_param,
                alpha=1.0 - momentum,
            )

    def _freeze_teacher(self):
        """Keep the detection teacher frozen."""

        if self.teacher_model is None:
            return

        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Feature hooks
    # ------------------------------------------------------------------

    def _register_feature_hooks(self) -> None:
        """Register feature hooks on both student and teacher."""

        self._remove_feature_hooks()

        for idx in self.feats_idx:
            if idx < 0 or idx >= len(self.student_model.model):
                raise IndexError(
                    f"Student feature layer index {idx} is outside model range."
                )

            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(
                    FeatureHook(self._student_feats, idx)
                )
            )

            if self.teacher_model is not None:
                if idx >= len(self.teacher_model.model):
                    raise IndexError(
                        f"Teacher feature layer index {idx} is outside model range."
                    )

                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(
                        FeatureHook(self._teacher_feats, idx)
                    )
                )

    def _remove_feature_hooks(self) -> None:
        """Remove previously registered feature hooks."""

        for handle in self._student_hooks:
            handle.remove()

        self._student_hooks.clear()

        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()

            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove stale FeatureHook instances."""

        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    # ------------------------------------------------------------------
    # Generic feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_feature(output):
        """
        Convert a hooked output into a [B,C,H,W] feature tensor.

        RT-DETR multiscale inputs and YOLO neck features are tensors.
        A tuple/list/dict is accepted defensively and the first suitable
        tensor is selected.
        """

        if torch.is_tensor(output):
            if output.ndim != 4:
                raise ValueError(
                    f"Expected a 4D feature tensor [B,C,H,W], got shape {tuple(output.shape)}."
                )
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.ndim == 4:
                    return item

        if isinstance(output, dict):
            for value in output.values():
                if torch.is_tensor(value) and value.ndim == 4:
                    return value
                if isinstance(value, (tuple, list)):
                    for item in value:
                        if torch.is_tensor(item) and item.ndim == 4:
                            return item

        raise TypeError(
            f"Could not extract a [B,C,H,W] feature tensor from "
            f"{type(output).__name__}."
        )

    # ------------------------------------------------------------------
    # PyTorch serialization
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Remove live tensors/hooks before pickling."""

        self._teacher_feats.clear()
        self._student_feats.clear()

        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []

        return state

    def __setstate__(self, state):
        """Restore state and recreate feature hooks after unpickling."""

        self.__dict__.update(state)

        self._teacher_feats = {}
        self._student_feats = {}

        self._register_feature_hooks()

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Set train mode while forcing the detection teacher to eval mode."""

        super().train(mode)
        self._freeze_teacher()

        return self

    def _apply(self, fn):
        """Apply device/dtype transforms to the unregistered teacher modules."""
        super()._apply(fn)
        self.teacher_model._apply(fn)
        self.dino_teacher_projector._apply(fn)
        return self

    def forward(self, x, *args, **kwargs):
        """Run student inference or compute the training loss."""

        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)

        return self.student_model.predict(x, *args, **kwargs)

    def fuse(
        self,
        verbose: bool = True,
        imgsz: int | list[int, int] = 640,
    ):
        """Remove distillation hooks and return the student for inference."""

        self._remove_feature_hooks()

        return self.student_model.fuse(
            verbose=verbose,
            imgsz=imgsz,
        )

    # ------------------------------------------------------------------
    # DINO augmentation
    # ------------------------------------------------------------------

    def dino_augment(self, crop):
        """Apply lightweight DINO-style image augmentations."""

        if torch.rand(1, device=crop.device).item() < 0.8:
            brightness = torch.empty(
                1,
                device=crop.device,
            ).uniform_(0.6, 1.4).item()

            contrast = torch.empty(
                1,
                device=crop.device,
            ).uniform_(0.6, 1.4).item()

            saturation = torch.empty(
                1,
                device=crop.device,
            ).uniform_(0.6, 1.4).item()

            crop = crop * brightness

            mean = crop.mean(
                dim=(2, 3),
                keepdim=True,
            )

            crop = (crop - mean) * contrast + mean

            gray = crop.mean(
                dim=1,
                keepdim=True,
            )

            crop = gray + (crop - gray) * saturation

            crop = crop.clamp(0, 1)

        if torch.rand(1, device=crop.device).item() < 0.2:
            gray = crop.mean(
                dim=1,
                keepdim=True,
            )

            crop = gray.repeat(1, 3, 1, 1)

        if torch.rand(1, device=crop.device).item() < 0.5:
            crop = F.avg_pool2d(
                crop,
                kernel_size=5,
                stride=1,
                padding=2,
            )

        if torch.rand(1, device=crop.device).item() < 0.2:
            crop = torch.where(
                crop > 0.5,
                1.0 - crop,
                crop,
            )

        return crop.clamp(0, 1)

    def create_dino_views(self, images):
        """Create two global and eight local crops per batch."""

        batch_size, _, h, w = images.shape

        global_views = []
        local_views = []

        # ------------------------------------------------------------
        # Two global views
        # ------------------------------------------------------------
        for _ in range(2):
            crops = []

            for b in range(batch_size):
                scale = torch.empty(
                    1,
                    device=images.device,
                ).uniform_(0.32, 1.0).item()

                crop_h = max(
                    32,
                    int(h * scale),
                )

                crop_w = max(
                    32,
                    int(w * scale),
                )

                top = torch.randint(
                    0,
                    max(1, h - crop_h + 1),
                    (1,),
                    device=images.device,
                ).item()

                left = torch.randint(
                    0,
                    max(1, w - crop_w + 1),
                    (1,),
                    device=images.device,
                ).item()

                crop = images[
                    b:b + 1,
                    :,
                    top:top + crop_h,
                    left:left + crop_w,
                ]

                crop = F.interpolate(
                    crop,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

                if torch.rand(1, device=images.device).item() < 0.5:
                    crop = crop.flip(-1)

                crop = self.dino_augment(crop)

                crops.append(crop)

            global_views.append(
                torch.cat(crops, dim=0)
            )

        # ------------------------------------------------------------
        # Eight local views
        # ------------------------------------------------------------
        local_size = max(
            32,
            min(h, w) // 2,
        )

        for _ in range(8):
            crops = []

            for b in range(batch_size):
                scale = torch.empty(
                    1,
                    device=images.device,
                ).uniform_(0.05, 0.32).item()

                crop_h = max(
                    32,
                    int(h * scale),
                )

                crop_w = max(
                    32,
                    int(w * scale),
                )

                top = torch.randint(
                    0,
                    max(1, h - crop_h + 1),
                    (1,),
                    device=images.device,
                ).item()

                left = torch.randint(
                    0,
                    max(1, w - crop_w + 1),
                    (1,),
                    device=images.device,
                ).item()

                crop = images[
                    b:b + 1,
                    :,
                    top:top + crop_h,
                    left:left + crop_w,
                ]

                crop = F.interpolate(
                    crop,
                    size=(local_size, local_size),
                    mode="bilinear",
                    align_corners=False,
                )

                if torch.rand(1, device=images.device).item() < 0.5:
                    crop = crop.flip(-1)

                crop = self.dino_augment(crop)
                crops.append(crop)

            local_views.append(
                torch.cat(crops, dim=0)
            )

        return global_views, local_views

    # ------------------------------------------------------------------
    # DINO loss
    # ------------------------------------------------------------------

    def dino_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ) -> torch.Tensor:
        """DINO cross-entropy with a moving teacher center."""

        student_output = student_output / student_temp

        student_log_probs = F.log_softmax(
            student_output,
            dim=1,
        )

        with torch.no_grad():
            batch_center = teacher_output.mean(
                dim=0,
                keepdim=True,
            )

            self.dino_center.mul_(
                self.dino_center_momentum
            ).add_(
                batch_center,
                alpha=1.0 - self.dino_center_momentum,
            )

            teacher_output = teacher_output - self.dino_center
            teacher_output = teacher_output / teacher_temp

            teacher_probs = F.softmax(
                teacher_output,
                dim=1,
            )

        loss = -(
            teacher_probs * student_log_probs
        ).sum(dim=1).mean()

        return loss

    # ------------------------------------------------------------------
    # Distillation losses
    # ------------------------------------------------------------------

    def loss_sl2(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        feat_idx: int,
        teacher_scores: tuple | None = None,
    ) -> torch.Tensor:
        """
        Compute feature L2 loss.

        For YOLO, teacher_scores are used for score weighting.

        For RT-DETR, teacher_scores=None and the loss becomes a normalized
        mean squared feature loss.
        """

        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if student_feat.shape[1] != teacher_feat.shape[1]:
            raise ValueError(
                "Student/teacher channel dimensions do not match after "
                f"projection: {student_feat.shape[1]} vs {teacher_feat.shape[1]}"
            )

        n, c = student_feat.shape[:2]

        student_flat = student_feat.view(
            n,
            c,
            -1,
        )

        teacher_flat = teacher_feat.view(
            n,
            c,
            -1,
        )

        mse = F.mse_loss(
            student_flat,
            teacher_flat,
            reduction="none",
        )

        # RT-DETR: architecture-agnostic feature loss.
        if teacher_scores is None:
            return mse.mean()

        # YOLO: score-weighted feature loss.
        teacher_score = teacher_scores[feat_idx]

        if teacher_score.shape[-1] != student_flat.shape[-1]:
            teacher_score = F.interpolate(
                teacher_score,
                size=student_flat.shape[-1],
                mode="linear",
                align_corners=False,
            )

        weighted_mse = (
            mse * teacher_score
        ).sum() / (
            teacher_score.sum() * c + 1e-9
        )

        return weighted_mse

    # ------------------------------------------------------------------
    # Main loss
    # ------------------------------------------------------------------

    def loss(self, batch, preds=None):
        """
        Compute:

            L_total =
                L_detection
                + dis * L_feature
                + dino_weight * L_DINO

        YOLO:
            L_feature is multiscale feature MSE on Detect input features.

        RT-DETR:
            L_feature is architecture-agnostic multiscale feature MSE.
        """

        device = batch["img"].device

        loss_distill = torch.zeros(
            1,
            device=device,
        )

        loss_dino = torch.zeros(
            1,
            device=device,
        )

        # ------------------------------------------------------------
        # Validation / evaluation.
        #
        # Do not run the expensive teacher + DINO pipeline here.
        # ------------------------------------------------------------
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])

            regular_loss, loss_items = self.student_model.loss(
                batch,
                preds,
            )

            if isinstance(loss_items, dict):
                loss_items["dino_loss"] = loss_dino.detach()
                loss_items["dis_loss"] = loss_distill.detach()

            return self._combine_detection_and_distill_loss(
                regular_loss,
                loss_distill,
                loss_dino,
                batch["img"].shape[0],
            ), loss_items

        # ------------------------------------------------------------
        # EMA teacher update for self-distillation.
        #
        # This happens before the teacher forward, so the teacher uses
        # the previous/EMA student state.
        # ------------------------------------------------------------
        if self.self_distill:
            self.update_teacher()

        # ------------------------------------------------------------
        # Clear hook storage.
        # ------------------------------------------------------------
        self._teacher_feats.clear()
        self._student_feats.clear()

        # ------------------------------------------------------------
        # Teacher forward.
        # ------------------------------------------------------------
        with torch.no_grad():
            self.teacher_model(batch["img"])

        # ------------------------------------------------------------
        # Student forward.
        # ------------------------------------------------------------
        preds = self.student_model(batch["img"])

        regular_loss, loss_items = self.student_model.loss(
            batch,
            preds,
        )

        # Clone references to the original feature tensors.
        original_teacher_feats = {
            idx: self._as_feature(feat)
            for idx, feat in self._teacher_feats.items()
        }

        original_student_feats = {
            idx: self._as_feature(feat)
            for idx, feat in self._student_feats.items()
        }

        # ============================================================
        # DINO DISTILLATION
        # ============================================================

        global_views, local_views = self.create_dino_views(
            batch["img"]
        )

        teacher_dino_views = []

        for view in global_views:
            self._teacher_feats.clear()

            with torch.no_grad():
                self.teacher_model(view)

            teacher_feat = self._as_feature(
                self._teacher_feats[self.dino_feature_idx]
            )

            teacher_dino_views.append(
                self.dino_teacher_projector(teacher_feat)
            )

        student_dino_views = []

        for view in global_views + local_views:
            self._student_feats.clear()

            self.student_model(view)

            student_feat = self._as_feature(
                self._student_feats[self.dino_feature_idx]
            )

            student_dino_views.append(
                self.dino_student_projector(student_feat)
            )

        num_terms = 0

        for teacher_view in teacher_dino_views:
            for student_view in student_dino_views:
                loss_dino += self.dino_loss(
                    student_view,
                    teacher_view,
                )

                num_terms += 1

        loss_dino = loss_dino / max(
            num_terms,
            1,
        )

        # ============================================================
        # FEATURE DISTILLATION
        # ============================================================

        if self.model_type == "yolo":
            # --------------------------------------------------------
            # YOLO feature distillation.
            # --------------------------------------------------------
            # Only the feature maps feeding Detect are distilled. The Detect
            # head output is intentionally excluded from self.feats_idx.
            # Use the same normalized feature MSE as RT-DETR here so the
            # distillation target is a genuine [B,C,H,W] representation.
            # --------------------------------------------------------
            for i, feat_idx in enumerate(self.feats_idx):
                teacher_feat = self._as_feature(
                    original_teacher_feats[feat_idx]
                )

                student_feat = self.projector[i](
                    self._as_feature(
                        original_student_feats[feat_idx]
                    )
                )

                loss_distill += (
                    self.loss_sl2(
                        student_feat,
                        teacher_feat,
                        feat_idx=i,
                        teacher_scores=None,
                    )
                    * self.dis
                )

        elif self.model_type == "rtdetr":
            # --------------------------------------------------------
            # RT-DETR feature distillation.
            #
            # There is no YOLO one2many/one2one score tensor here.
            # Distill the multiscale feature maps directly.
            # --------------------------------------------------------
            for i, feat_idx in enumerate(
                self.feats_idx
            ):
                teacher_feat = self._as_feature(
                    original_teacher_feats[feat_idx]
                )

                student_feat = self.projector[i](
                    self._as_feature(
                        original_student_feats[feat_idx]
                    )
                )

                loss_distill += (
                    self.loss_sl2(
                        student_feat,
                        teacher_feat,
                        feat_idx=i,
                        teacher_scores=None,
                    )
                    * self.dis
                )

        # ------------------------------------------------------------
        # Logging.
        # ------------------------------------------------------------
        if isinstance(loss_items, dict):
            loss_items["dino_loss"] = loss_dino.detach()
            loss_items["dis_loss"] = loss_distill.detach()

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # The previous implementation computed loss_dino and
        # loss_distill but returned only the detection loss.
        #
        # This version explicitly adds both terms to the optimization
        # loss.
        # ------------------------------------------------------------
        total_loss = self._combine_detection_and_distill_loss(
            regular_loss,
            loss_distill,
            loss_dino,
            batch["img"].shape[0],
        )

        return total_loss, loss_items

    def _combine_detection_and_distill_loss(
        self,
        regular_loss,
        loss_distill,
        loss_dino,
        batch_size,
    ):
        """
        Combine Ultralytics detection loss with scalar distillation terms.

        Ultralytics versions can return the detection loss as either a
        tensor or a list/tuple of tensors, so handle both forms.
        """

        if isinstance(regular_loss, (list, tuple)):
            terms = [
                x if torch.is_tensor(x) else torch.as_tensor(x, device=loss_distill.device)
                for x in regular_loss
            ]
            detection_loss = torch.stack([x.reshape(()) for x in terms]).sum()
        else:
            detection_loss = regular_loss

        if not torch.is_tensor(detection_loss):
            detection_loss = torch.as_tensor(
                detection_loss,
                device=loss_distill.device,
            )

        # Detection loss is normally already scaled by batch size.
        # Match that convention for the additional losses.
        distill_term = (
            loss_distill * batch_size
        )

        dino_term = (
            loss_dino * self.dino_weight * batch_size
        )

        return (
            detection_loss
            + distill_term
            + dino_term
        )

    # ------------------------------------------------------------------
    # YOLO output compatibility helper
    # ------------------------------------------------------------------

    def decouple_outputs(
        self,
        preds,
        branch: str = "one2one",
    ):
        """
        Normalize YOLO output formats.

        This function is intentionally used only for YOLO distillation.
        RT-DETR feature distillation never calls it.
        """

        if isinstance(preds, tuple):
            # YOLO validation format.
            preds = preds[1]

        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]

        return preds

    # ------------------------------------------------------------------
    # Ultralytics model interface forwarding
    # ------------------------------------------------------------------

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(
            self.student_model,
            "end2end",
            False,
        )

    @end2end.setter
    def end2end(self, value):
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        self.student_model.set_head_attr(**kwargs)
