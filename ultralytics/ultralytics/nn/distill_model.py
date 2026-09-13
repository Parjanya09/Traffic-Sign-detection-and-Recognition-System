# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr
from copy import deepcopy

from .tasks import load_checkpoint

class DINOHead(nn.Module):
    """Prototype head for DINO-style self-distillation."""

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
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the layer index to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        fuse: Fuse and return the student model for inference and export.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """
    #modified __init__ 08-09-2026
    def __init__(self, teacher_model: str | Path | nn.Module | None, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        device = next(student_model.parameters()).device

        self.student_model = student_model
        self.student_model.train()
        self.student_model.requires_grad_(True)
        self.self_distill = bool(student_model.args.get("dino_distill", False))

        if self.self_distill:
            self.teacher_model = deepcopy(student_model).to(device)
        else:
            if isinstance(teacher_model, (str, Path)):
                teacher_model = load_checkpoint(teacher_model)[0]
                if teacher_model.yaml.get("channels", 3) != ch:
                    weights = teacher_model
                    teacher_model = type(weights)(
                    weights.yaml.copy(),
                    ch=ch,
                    nc=weights.yaml["nc"],
                    verbose=False,
                    )
                    teacher_model.load(weights)
            self.teacher_model = teacher_model.to(device)

        self._freeze_teacher() # modified this student model and teacher model section from device ..
        self.feats_idx = self.get_distill_layers(student_model)

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.get("imgsz")
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            self.teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.get("dis", 6.0)
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)
        dino_student_feat = self.decouple_outputs(student_output[-2])
        dino_teacher_feat = self.decouple_outputs(teacher_output[-2])
        self.dino_student_projector = DINOHead(
            dino_student_feat.shape[1],
            hidden_dim=512,
            bottleneck_dim=256,
            out_dim=1024,
        ).to(device)
        self.dino_teacher_projector = DINOHead(
            dino_teacher_feat.shape[1],
            hidden_dim=512,
            bottleneck_dim=256,
            out_dim=1024,
        ).to(device)
        self.dino_teacher_projector.load_state_dict(
            self.dino_student_projector.state_dict()
        )
        for param in self.dino_teacher_projector.parameters():
            param.requires_grad = False
        self.register_buffer("dino_center",torch.zeros(1, 1024, device=device),)
        self.dino_center_momentum = 0.9

    # added update_tecaher funtion 08-09-2026
    @torch.no_grad()
    def update_teacher(self):
        """Update DINO teacher using EMA of student."""
        if not self.self_distill or self.teacher_model is None:
            return

        momentum = self.student_model.args.get("distill_momentum", 0.996)
        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters(),):
            teacher_param.mul_(momentum).add_(student_param,alpha=1.0 - momentum,)

        for teacher_buffer, student_buffer in zip(self.teacher_model.buffers(),self.student_model.buffers(),):
            teacher_buffer.copy_(student_buffer)
        for teacher_param, student_param in zip(self.dino_teacher_projector.parameters(),self.dino_student_projector.parameters(),):
            teacher_param.mul_(momentum).add_(student_param,alpha=1.0 - momentum,)

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        for idx in self.feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640):
        """Fuse and return the student model, dropping the training-only distillation wrapper."""
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def dino_augment(self, crop):
        """Apply DINO-style augmentations to a crop."""
    # Color jitter
        if torch.rand(1, device=crop.device).item() < 0.8:
            brightness = torch.empty(1, device=crop.device).uniform_(0.6, 1.4).item()
            contrast = torch.empty(1, device=crop.device).uniform_(0.6, 1.4).item()
            saturation = torch.empty(1, device=crop.device).uniform_(0.6, 1.4).item()

        # Brightness
            crop = crop * brightness

        # Contrast around per-image mean
            mean = crop.mean(dim=(2, 3), keepdim=True)
            crop = (crop - mean) * contrast + mean

        # Approximate saturation adjustment
            gray = crop.mean(dim=1, keepdim=True)
            crop = gray + (crop - gray) * saturation

            crop = crop.clamp(0, 1)

    # Random grayscale
        if torch.rand(1, device=crop.device).item() < 0.2:
            gray = crop.mean(dim=1, keepdim=True)
            crop = gray.repeat(1, 3, 1, 1)

    # Gaussian blur
        if torch.rand(1, device=crop.device).item() < 0.5:
            crop = F.avg_pool2d(
            crop,
            kernel_size=5,
            stride=1,
            padding=2,
        )

    # Solarization
        if torch.rand(1, device=crop.device).item() < 0.2:
            crop = torch.where(crop > 0.5, 1.0 - crop, crop)
        return crop.clamp(0, 1)

    def create_dino_views(self, images):
        """Create DINO-style global and local crops independently per image."""
        batch_size, _, h, w = images.shape

        global_views = []
        local_views = []

        # Two global crops
        for _ in range(2):
            crops = []

            for b in range(batch_size):
                scale = torch.empty(1, device=images.device).uniform_(0.32, 1.0).item()

                crop_h = max(32, int(h * scale))
                crop_w = max(32, int(w * scale))

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

            global_views.append(torch.cat(crops, dim=0))

    # Eight local crops
        local_size = max(32, min(h, w) // 2)

        for _ in range(8):
            crops = []

            for b in range(batch_size):
                scale = torch.empty(1, device=images.device).uniform_(0.05, 0.32).item()

                crop_h = max(32, int(h * scale))
                crop_w = max(32, int(w * scale))

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

                crops.append(crop)

            local_views.append(torch.cat(crops, dim=0))

        return global_views, local_views
        
    def dino_loss(self,student_output: torch.Tensor,teacher_output: torch.Tensor,student_temp: float = 0.1,teacher_temp: float = 0.04,) -> torch.Tensor:
        """Compute DINO cross-entropy with teacher centering."""
        student_output = student_output / student_temp
        student_log_probs = F.log_softmax(student_output, dim=1)

        with torch.no_grad():
            batch_center = teacher_output.mean(dim=0, keepdim=True)
            self.dino_center.mul_(self.dino_center_momentum).add_(
                batch_center,
                alpha=1.0 - self.dino_center_momentum,
            )

            teacher_output = teacher_output - self.dino_center
            teacher_output = teacher_output / teacher_temp
            teacher_probs = F.softmax(teacher_output, dim=1)
        loss = -(teacher_probs * student_log_probs).sum(dim=1).mean()

        return loss

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        loss_dino = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features
        regular_loss, loss_items = self.student_model.loss(batch, preds)
        original_teacher_feats = {
            idx: feat
            for idx, feat in self._teacher_feats.items()}
        original_student_feats = {
            idx: feat
            for idx, feat in self._student_feats.items()
            }
        #modified
        global_views, local_views = self.create_dino_views(batch["img"])
        teacher_dino_views = []
        for view in global_views:
            self._teacher_feats.clear()
            with torch.no_grad():
                self.teacher_model(view)
            teacher_feat = self.decouple_outputs(
                self._teacher_feats[self.feats_idx[-2]]
            )
            teacher_dino_views.append(
                self.dino_teacher_projector(teacher_feat)
            )
        student_dino_views = []
        for view in global_views + local_views:
            self._student_feats.clear()
            self.student_model(view)
            student_feat = self.decouple_outputs(
                self._student_feats[self.feats_idx[-2]]
            )

            student_dino_views.append(
                self.dino_student_projector(student_feat)
            )
        loss_dino = torch.zeros(1, device=batch["img"].device)
        num_terms = 0
        for teacher_view in teacher_dino_views:
            for student_view in student_dino_views:
                loss_dino += self.dino_loss(
                    student_view,
                    teacher_view,
                )
                num_terms += 1

        loss_dino = loss_dino / max(num_terms, 1)
        ##

        teacher_head_feat = original_teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        # neck feature sizes vary per batch (e.g. multi_scale), so split scores by the live teacher feats
        neck_feats = [original_teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(original_teacher_feats[feat_idx])
            student_feat = self.projector[i](self.decouple_outputs(original_student_feats[feat_idx]))
            loss_distill += (
                self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
            )

        loss_items["dino_loss"] = loss_dino.detach()
        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        loss_dino = loss_dino * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_dino]), loss_items # 

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
