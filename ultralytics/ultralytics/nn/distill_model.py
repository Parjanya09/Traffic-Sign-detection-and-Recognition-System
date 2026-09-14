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
    def __init__(self, in_dim, hidden_dim=512, bottleneck_dim=256, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, bottleneck_dim, 1),
        )
        self.last_layer = nn.Conv2d(bottleneck_dim, out_dim, 1, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        x = F.normalize(x, dim=1)
        return self.last_layer(x).flatten(1)


class FeatureHook:
    def __init__(self, feat_dict: dict, idx: int):
        self.feat_dict, self.idx = feat_dict, idx

    def __call__(self, module, inputs, output):
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """Memory-safe YOLO/RT-DETR teacher-student KD with optional DINO.

    Recommended first run:
        dino_distill=True
        dino_global_views=1
        dino_local_views=0

    Later, after memory is verified:
        dino_global_views=2
        dino_local_views=2 or 4
    """

    def __init__(self, teacher_model: str | Path | nn.Module | None, student_model: nn.Module):
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        device = next(student_model.parameters()).device

        self.student_model = student_model
        self.student_model.train()
        self.student_model.requires_grad_(True)

        self.self_distill = bool(student_model.args.get("dino_distill", False))
        self.dino_enabled = self.self_distill
        self.dino_global_views = max(0, int(student_model.args.get("dino_global_views", 1)))
        self.dino_local_views = max(0, int(student_model.args.get("dino_local_views", 0)))
        self.dino_weight = float(student_model.args.get("dino_weight", 1.0))
        self.dis = float(student_model.args.get("dis", 6.0))

        if self.self_distill:
            self.teacher_model = deepcopy(student_model).to(device)
        else:
            if isinstance(teacher_model, (str, Path)):
                teacher_model = load_checkpoint(teacher_model)[0]
                if teacher_model.yaml.get("channels", 3) != ch:
                    weights = teacher_model
                    teacher_model = type(weights)(
                        weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False
                    )
                    teacher_model.load(weights)
            self.teacher_model = teacher_model.to(device)

        self._freeze_teacher()
        self.is_rtdetr = any(isinstance(m, RTDETRDecoder) for m in student_model.model)
        self.feats_idx = self.get_distill_layers(student_model)

        self._teacher_feats, self._student_feats = {}, {}
        self._teacher_hooks, self._student_hooks = [], []
        self._register_feature_hooks()

        imgsz = student_model.args.get("imgsz", 640)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        dummy_size = min(int(imgsz), 320)

        student_model.eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, ch, dummy_size, dummy_size, device=device)
            self.teacher_model(dummy)
            student_model(dummy)
        student_model.train()

        teacher_outs = [self._as_feature_tensor(self._teacher_feats[i]) for i in self.feats_idx]
        student_outs = [self._as_feature_tensor(self._student_feats[i]) for i in self.feats_idx]

        copy_attr(self, student_model)

        self.projector = nn.ModuleList()
        for sf, tf in zip(student_outs, teacher_outs):
            if sf.shape[1] == tf.shape[1]:
                self.projector.append(nn.Identity())
            else:
                self.projector.append(nn.Sequential(
                    nn.Conv2d(sf.shape[1], tf.shape[1], 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(tf.shape[1], tf.shape[1], 1),
                ))

        dino_sf = student_outs[-1]
        dino_tf = teacher_outs[-1]
        self.dino_student_projector = DINOHead(dino_sf.shape[1]).to(device)
        self.dino_teacher_projector = DINOHead(dino_tf.shape[1]).to(device)
        self.dino_teacher_projector.load_state_dict(self.dino_student_projector.state_dict())
        self.dino_teacher_projector.requires_grad_(False)
        self.dino_teacher_projector.eval()

        self.register_buffer("dino_center", torch.zeros(1, 1024, device=device))
        self.dino_center_momentum = 0.9

        self._teacher_feats.clear()
        self._student_feats.clear()

    @staticmethod
    def _as_feature_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            for v in reversed(x):
                if torch.is_tensor(v):
                    return v
        if isinstance(x, dict):
            for k in ("features", "feat", "output"):
                if k in x and torch.is_tensor(x[k]):
                    return x[k]
            for v in reversed(list(x.values())):
                if torch.is_tensor(v):
                    return v
        raise TypeError(f"Expected feature tensor, got {type(x)}")

    @staticmethod
    def get_distill_layers(model):
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
            if isinstance(m, RTDETRDecoder):
                return list(m.f)
        raise ValueError("No Detect or RTDETRDecoder head found in model")

    def _freeze_teacher(self):
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        if hasattr(self, "dino_teacher_projector"):
            self.dino_teacher_projector.eval()
            self.dino_teacher_projector.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_teacher()
        return self

    @torch.no_grad()
    def update_teacher(self):
        if not self.self_distill or self.teacher_model is None:
            return
        momentum = float(self.student_model.args.get("distill_momentum", 0.996))
        for tp, sp in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            tp.mul_(momentum).add_(sp, alpha=1.0 - momentum)
        for tb, sb in zip(self.teacher_model.buffers(), self.student_model.buffers()):
            tb.copy_(sb)
        for tp, sp in zip(self.dino_teacher_projector.parameters(), self.dino_student_projector.parameters()):
            tp.mul_(momentum).add_(sp, alpha=1.0 - momentum)

    def __getstate__(self):
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"], state["_student_hooks"] = [], []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._teacher_feats, self._student_feats = {}, {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self):
        for h in self._student_hooks:
            h.remove()
        self._student_hooks.clear()
        for h in self._teacher_hooks:
            h.remove()
        self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module):
        for hid, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[hid]

    def _register_feature_hooks(self):
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

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose=True, imgsz=640):
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def dino_augment(self, crop):
        if torch.rand((), device=crop.device) < 0.8:
            b = torch.empty((), device=crop.device).uniform_(0.6, 1.4)
            c = torch.empty((), device=crop.device).uniform_(0.6, 1.4)
            s = torch.empty((), device=crop.device).uniform_(0.6, 1.4)
            crop = crop * b
            mean = crop.mean(dim=(2, 3), keepdim=True)
            crop = (crop - mean) * c + mean
            gray = crop.mean(dim=1, keepdim=True)
            crop = gray + (crop - gray) * s
        if torch.rand((), device=crop.device) < 0.2:
            gray = crop.mean(dim=1, keepdim=True)
            crop = gray.repeat(1, 3, 1, 1)
        if torch.rand((), device=crop.device) < 0.5:
            crop = F.avg_pool2d(crop, 5, 1, 2)
        if torch.rand((), device=crop.device) < 0.2:
            crop = torch.where(crop > 0.5, 1.0 - crop, crop)
        return crop.clamp(0, 1)

    def _make_one_view(self, images, local=False):
        bsz, _, h, w = images.shape
        crops = []
        low, high = (0.05, 0.32) if local else (0.32, 1.0)
        out_size = max(32, min(h, w) // 2) if local else (h, w)

        for b in range(bsz):
            scale = torch.empty((), device=images.device).uniform_(low, high).item()
            ch, cw = max(32, int(h * scale)), max(32, int(w * scale))
            top = torch.randint(0, max(1, h - ch + 1), (), device=images.device).item()
            left = torch.randint(0, max(1, w - cw + 1), (), device=images.device).item()
            crop = images[b:b+1, :, top:top+ch, left:left+cw]
            crop = F.interpolate(crop, size=out_size, mode="bilinear", align_corners=False)
            if torch.rand((), device=images.device) < 0.5:
                crop = crop.flip(-1)
            crops.append(self.dino_augment(crop))
        return torch.cat(crops, dim=0)

    @torch.no_grad()
    def _teacher_dino_output(self, view):
        self._teacher_feats.clear()
        self.teacher_model(view)
        feat = self._as_feature_tensor(self._teacher_feats[self.feats_idx[-1]])
        return self.dino_teacher_projector(feat)

    def _student_dino_output(self, view):
        self._student_feats.clear()
        self.student_model(view)
        feat = self._as_feature_tensor(self._student_feats[self.feats_idx[-1]])
        return self.dino_student_projector(feat)

    def dino_loss(self, student_output, teacher_output, student_temp=0.1, teacher_temp=0.04):
        student_log_probs = F.log_softmax(student_output / student_temp, dim=1)
        with torch.no_grad():
            center = teacher_output.mean(dim=0, keepdim=True)
            self.dino_center.mul_(self.dino_center_momentum).add_(
                center, alpha=1.0 - self.dino_center_momentum
            )
            teacher_probs = F.softmax(
                (teacher_output - self.dino_center) / teacher_temp, dim=1
            )
        return -(teacher_probs * student_log_probs).sum(dim=1).mean()

    def _compute_dino_loss(self, images):
        if not self.dino_enabled or self.dino_weight == 0:
            return images.new_zeros(())

        total = images.new_zeros(())
        terms = 0

        # Each view is created, forwarded, used, and deleted before the next.
        for _ in range(self.dino_global_views):
            view = self._make_one_view(images, local=False)
            with torch.no_grad():
                teacher_out = self._teacher_dino_output(view)
            student_out = self._student_dino_output(view)
            total = total + self.dino_loss(student_out, teacher_out)
            terms += 1
            del teacher_out, student_out, view

        for _ in range(self.dino_local_views):
            view = self._make_one_view(images, local=True)
            with torch.no_grad():
                teacher_out = self._teacher_dino_output(view)
            student_out = self._student_dino_output(view)
            total = total + self.dino_loss(student_out, teacher_out)
            terms += 1
            del teacher_out, student_out, view

        return total / max(terms, 1)

    def _compute_rtdetr_feature_loss(self, teacher_feats, student_feats):
        total = next(iter(teacher_feats.values())).new_zeros(())
        for i, idx in enumerate(self.feats_idx):
            tf = self._as_feature_tensor(teacher_feats[idx]).detach()
            sf = self.projector[i](self._as_feature_tensor(student_feats[idx]))
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear", align_corners=False)
            total = total + F.mse_loss(sf, tf)
        return total / max(len(self.feats_idx), 1)

    def _compute_yolo_feature_loss(self, teacher_feats, student_feats):
        head = teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(head, "one2many")["scores"]
            + self.decouple_outputs(head, "one2one")["scores"]
        ) / 2
        neck = [teacher_feats[i] for i in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck], dim=-1)
        scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)

        total = head.new_zeros(())
        for i, idx in enumerate(self.feats_idx[:-1]):
            tf = self.decouple_outputs(teacher_feats[idx]).detach()
            sf = self.projector[i](self.decouple_outputs(student_feats[idx]))
            total = total + self.loss_sl2(sf, tf, i, scores)
        return total * self.dis

    def loss(self, batch, preds=None):
        # NOTE: teacher forward uses torch.no_grad(), NOT torch.inference_mode().
        # Its outputs are used as targets by autograd-tracked KD losses.
        images = batch["img"]
        zero = images.new_zeros(())

        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dino_loss"] = zero.detach()
            loss_items["dis_loss"] = zero.detach()
            return torch.cat([regular_loss, zero]), loss_items

        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(images)

        preds = self.student_model(images)
        regular_loss, loss_items = self.student_model.loss(batch, preds)

        teacher_feats = dict(self._teacher_feats)
        student_feats = dict(self._student_feats)

        if self.is_rtdetr:
            loss_distill = self._compute_rtdetr_feature_loss(teacher_feats, student_feats)
        else:
            loss_distill = self._compute_yolo_feature_loss(teacher_feats, student_feats)

        loss_dino = self._compute_dino_loss(images) * self.dino_weight

        batch_size = images.shape[0]
        total_loss = torch.cat([regular_loss]) + (loss_distill + loss_dino) * batch_size

        loss_items["dino_loss"] = loss_dino.detach()
        loss_items["dis_loss"] = loss_distill.detach()

        self._teacher_feats.clear()
        self._student_feats.clear()

        return total_loss, loss_items

    def loss_sl2(self, student_feat, teacher_feat, feat_idx, teacher_scores):
        score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        sf = student_feat.view(n, c, -1)
        tf = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(sf, tf, reduction="none")
        return (mse * score).sum() / (score.sum() * c + 1e-9)

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value):
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch="one2one"):
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
