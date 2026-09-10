import torch

from ultralytics import YOLO
from ultralytics.nn.distill_model import DistillationModel


y = YOLO("yolo26n.pt")

s = y.model
s.args["dino_distill"] = True
s.args["imgsz"] = 320

d = DistillationModel(None, s)

x = torch.rand(2, 3, 320, 320)

g, l = d.create_dino_views(x)

# -------------------------
# Student view 1
# -------------------------
d.student_model(g[0])

student_feat_1 = d.decouple_outputs(
    d._student_feats[d.feats_idx[-2]]
)

# -------------------------
# Student view 2
# -------------------------
d.student_model(g[1])

student_feat_2 = d.decouple_outputs(
    d._student_feats[d.feats_idx[-2]]
)

# -------------------------
# Teacher view
# -------------------------
with torch.no_grad():
    d.teacher_model(g[0])

    teacher_feat = d.decouple_outputs(
        d._teacher_feats[d.feats_idx[-2]]
    )

    teacher_output = d.dino_teacher_projector(
        teacher_feat
    )

# -------------------------
# DINO projections
# -------------------------
student_output_1 = d.dino_student_projector(
    student_feat_1
)

student_output_2 = d.dino_student_projector(
    student_feat_2
)

# -------------------------
# Cross-view DINO loss
# -------------------------
teacher_probs = torch.softmax(
    teacher_output,
    dim=1,
)

student_log_probs_1 = torch.log_softmax(
    student_output_1,
    dim=1,
)

student_log_probs_2 = torch.log_softmax(
    student_output_2,
    dim=1,
)

loss_1 = -(
    teacher_probs * student_log_probs_1
).sum(dim=1).mean()

loss_2 = -(
    teacher_probs * student_log_probs_2
).sum(dim=1).mean()

loss = loss_1 + loss_2

print("DINO loss:", loss.item())

# -------------------------
# Backward
# -------------------------
loss.backward()

student_head_grads = [
    p.grad
    for p in d.dino_student_projector.parameters()
    if p.grad is not None
]

teacher_head_grads = [
    p.grad
    for p in d.dino_teacher_projector.parameters()
    if p.grad is not None
]

student_model_grads = [
    p.grad
    for p in d.student_model.parameters()
    if p.grad is not None
]

print(
    "Student head gradients:",
    len(student_head_grads),
)

print(
    "Student gradients finite:",
    all(
        torch.isfinite(g).all().item()
        for g in student_head_grads
    ),
)

print(
    "Teacher head gradients:",
    len(teacher_head_grads),
)

print(
    "Student model gradients:",
    len(student_model_grads),
)

print(
    "Student model gradients finite:",
    all(
        torch.isfinite(g).all().item()
        for g in student_model_grads
    ),
)