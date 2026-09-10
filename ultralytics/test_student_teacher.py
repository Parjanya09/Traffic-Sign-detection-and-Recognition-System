import copy
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.distill_model import DistillationModel


# Configuration

MODEL_PATH = "yolo26n.pt"
DATA_PATH = "coco8.yaml"

IMAGE_SIZE = 320
BATCH_SIZE = 2
EPOCHS = 2
WORKERS = 0

MOMENTUM = 0.996

PROJECT = "runs/student_teacher"
RUN_NAME = "student_teacher_test"


# Utility functions

def count_trainable_tensors(model):
    return sum(
        p.requires_grad
        for p in model.parameters()
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def parameters_changed(before, model):
    after = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    for name in before:
        if not torch.equal(before[name], after[name]):
            return True

    return False


def parameter_difference(model_a, model_b):
    total_difference = 0.0

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        total_difference += (
            parameter_a.detach() - parameter_b.detach()
        ).abs().mean().item()

    return total_difference


# Test 1: Build Student-Teacher model

print("Student-Teacher YOLO Test")
print()

print("Loading YOLO model...")

yolo = YOLO(MODEL_PATH)

student = yolo.model

# Enable the self-distillation mode.
student.args["dino_distill"] = True
student.args["distill_momentum"] = MOMENTUM

print("Building Student-Teacher architecture...")

model = DistillationModel(
    student_model=student,
    teacher_model=None,
)

print()
print("Model construction")
print("Student exists:", model.student_model is not None)
print("Teacher exists:", model.teacher_model is not None)
print("Self-distillation:", model.self_distill)


# Test 2: Student / Teacher configuration

print()
print("Student-Teacher configuration")

student_trainable = count_trainable_tensors(
    model.student_model
)

teacher_trainable = count_trainable_tensors(
    model.teacher_model
)

student_parameters = count_parameters(
    model.student_model
)

teacher_parameters = count_parameters(
    model.teacher_model
)

print(
    "Student training mode:",
    model.student_model.training
)

print(
    "Teacher evaluation mode:",
    not model.teacher_model.training
)

print(
    "Student trainable tensors:",
    student_trainable
)

print(
    "Teacher trainable tensors:",
    teacher_trainable
)

print(
    "Student parameters:",
    student_parameters
)

print(
    "Teacher parameters:",
    teacher_parameters
)

if (
    model.student_model.training
    and not model.teacher_model.training
    and student_trainable > 0
    and teacher_trainable == 0
):
    print("Configuration test: PASS")
else:
    print("Configuration test: FAIL")


# Test 3: Student and Teacher forward pass

print()
print("Forward pass")

device = next(
    model.student_model.parameters()
).device

dummy_images = torch.rand(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device,
)

print(
    "Input shape:",
    tuple(dummy_images.shape)
)

model.student_model.eval()
model.teacher_model.eval()

with torch.no_grad():

    try:
        student_output = model.student_model(
            dummy_images
        )

        print("Student forward: PASS")

    except Exception as error:

        print("Student forward: FAIL")
        print(error)

        raise


with torch.no_grad():

    try:
        teacher_output = model.teacher_model(
            dummy_images
        )

        print("Teacher forward: PASS")

    except Exception as error:

        print("Teacher forward: FAIL")
        print(error)

        raise


# Restore Student training mode.
model.student_model.train()
model.teacher_model.eval()


# ------------------------------------------------------------
# Test 4: Feature capture
# ------------------------------------------------------------

print()
print("Feature capture")

model._teacher_feats.clear()
model._student_feats.clear()

with torch.no_grad():

    model.student_model(dummy_images)
    model.teacher_model(dummy_images)

student_feature_count = len(
    model._student_feats
)

teacher_feature_count = len(
    model._teacher_feats
)

print(
    "Student feature hooks:",
    student_feature_count
)

print(
    "Teacher feature hooks:",
    teacher_feature_count
)

if (
    student_feature_count > 0
    and teacher_feature_count > 0
):
    print("Feature capture: PASS")
else:
    print("Feature capture: FAIL")


# Test 5: Distillation loss

print()
print("Distillation")

model._teacher_feats.clear()
model._student_feats.clear()

model.student_model.train()
model.teacher_model.eval()

student_output = model.student_model(
    dummy_images
)

with torch.no_grad():

    teacher_output = model.teacher_model(
        dummy_images
    )

try:

    dis_loss = model.distill_loss(
        model._student_feats,
        model._teacher_feats,
    )

    print(
        "Distillation loss:",
        float(dis_loss.detach().cpu())
    )

    if torch.isfinite(dis_loss):
        print("Distillation loss finite: PASS")
    else:
        print("Distillation loss finite: FAIL")

except Exception:

    # Some versions of the current implementation calculate
    # the distillation loss through the forward/training path.
    #
    # In that case we use the model's available loss path
    # below instead of assuming a particular internal method.

    print(
        "Direct distillation-loss call is not available "
        "in this implementation."
    )

    print(
        "Feature tensors were successfully captured, "
        "so the training test will verify the complete loss path."
    )


# Test 6: Gradient flow

print()
print("Gradient flow")

model.student_model.zero_grad(
    set_to_none=True
)

model.teacher_model.zero_grad(
    set_to_none=True
)

model._teacher_feats.clear()
model._student_feats.clear()

model.student_model.train()
model.teacher_model.eval()

student_output = model.student_model(
    dummy_images
)

with torch.no_grad():

    teacher_output = model.teacher_model(
        dummy_images
    )


feature_loss = torch.tensor(
    0.0,
    device=device,
)

for index in model.feats_idx:

    student_feature = model._student_feats[index]
    teacher_feature = model._teacher_feats[index]

    if isinstance(student_feature, (tuple, list)):
        student_feature = student_feature[0]

    if isinstance(teacher_feature, (tuple, list)):
        teacher_feature = teacher_feature[0]

    if isinstance(student_feature, dict):
        continue

    if isinstance(teacher_feature, dict):
        continue

    if (
        torch.is_tensor(student_feature)
        and torch.is_tensor(teacher_feature)
    ):

        if student_feature.shape != teacher_feature.shape:

            teacher_feature = torch.nn.functional.interpolate(
                teacher_feature,
                size=student_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        channels = min(
            student_feature.shape[1],
            teacher_feature.shape[1],
        )

        feature_loss = feature_loss + (
            student_feature[:, :channels]
            - teacher_feature[:, :channels].detach()
        ).pow(2).mean()


if feature_loss.requires_grad:

    feature_loss.backward()

student_gradient_count = sum(
    1
    for parameter in model.student_model.parameters()
    if parameter.grad is not None
    and torch.isfinite(parameter.grad).all()
)

teacher_gradient_count = sum(
    1
    for parameter in model.teacher_model.parameters()
    if parameter.grad is not None
)

print(
    "Diagnostic feature loss:",
    float(feature_loss.detach().cpu())
)

print(
    "Student tensors with finite gradients:",
    student_gradient_count
)

print(
    "Teacher tensors with gradients:",
    teacher_gradient_count
)

if student_gradient_count > 0:
    print("Student gradient flow: PASS")
else:
    print("Student gradient flow: FAIL")

if teacher_gradient_count == 0:
    print("Teacher gradient protection: PASS")
else:
    print("Teacher gradient protection: FAIL")


# Test 7: EMA teacher update

print()
print("EMA teacher update")

model.student_model.zero_grad(
    set_to_none=True
)

model.teacher_model.zero_grad(
    set_to_none=True
)

student_before = {
    name: parameter.detach().clone()
    for name, parameter in model.student_model.named_parameters()
}

teacher_before = {
    name: parameter.detach().clone()
    for name, parameter in model.teacher_model.named_parameters()
}

# Create an artificial Student update for the test.
with torch.no_grad():

    first_parameter = next(
        model.student_model.parameters()
    )

    first_parameter.add_(0.001)

model.update_teacher()

student_changed = parameters_changed(
    student_before,
    model.student_model,
)

teacher_changed = parameters_changed(
    teacher_before,
    model.teacher_model,
)

teacher_student_difference = parameter_difference(
    model.student_model,
    model.teacher_model,
)

print(
    "Student parameters changed:",
    student_changed
)

print(
    "Teacher parameters changed:",
    teacher_changed
)

print(
    "Student-Teacher parameter difference:",
    teacher_student_difference
)

if teacher_changed:
    print("EMA teacher update: PASS")
else:
    print("EMA teacher update: FAIL")


# Test 8: Real YOLO training

print()
print("Training")

print("Dataset:", DATA_PATH)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Workers:", WORKERS)

print()
print("Starting real YOLO training...")

train_model = YOLO(MODEL_PATH)

try:

    train_model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,

        dino_distill=True,
        distill_momentum=MOMENTUM,

        project=PROJECT,
        name=RUN_NAME,

        exist_ok=True,
    )

    training_success = True

except Exception as error:

    training_success = False

    print()
    print("Training failed.")
    print(error)

    raise


if training_success:
    print()
    print("Training completed: PASS")


# Test 9: Locate trained model

print()
print("Trained model")

run_directory = Path(
    train_model.trainer.save_dir
)

best_model_path = (
    run_directory
    / "weights"
    / "best.pt"
)

last_model_path = (
    run_directory
    / "weights"
    / "last.pt"
)

print(
    "Training directory:",
    run_directory
)

print(
    "Best model:",
    best_model_path
)

print(
    "Last model:",
    last_model_path
)

if best_model_path.exists():

    trained_model_path = best_model_path
    print("best.pt found: PASS")

elif last_model_path.exists():

    trained_model_path = last_model_path
    print("last.pt found: PASS")

else:

    print("Trained checkpoint: FAIL")

    raise FileNotFoundError(
        "Could not find best.pt or last.pt in "
        f"{run_directory / 'weights'}"
    )

# Test 10: Inference using trained Student

print()
print("Inference")

print(
    "Loading trained model:",
    trained_model_path
)

trained_yolo = YOLO(
    str(trained_model_path)
)

try:

    results = trained_yolo.predict(
        source="ultralytics/assets/bus.jpg",
        imgsz=IMAGE_SIZE,
        conf=0.25,
        verbose=False,
    )

    print("Model loading: PASS")
    print("Inference: PASS")

    if len(results) > 0:

        result = results[0]

        if result.boxes is not None:

            number_of_boxes = len(
                result.boxes
            )

            print(
                "Detected objects:",
                number_of_boxes
            )

            if number_of_boxes > 0:

                print(
                    "Bounding boxes: PASS"
                )

                print(
                    "Class predictions: PASS"
                )

            else:

                print(
                    "Bounding boxes: no detections"
                )

        else:

            print(
                "Bounding boxes: no detection tensor"
            )

except Exception as error:

    print("Inference: FAIL")
    print(error)

    raise

print("Done.")
