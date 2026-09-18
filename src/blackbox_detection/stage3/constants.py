from __future__ import annotations

ACCEL_CLASSES = ("ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED")
STEER_CLASSES = ("LEFT", "STRAIGHT", "RIGHT")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CAN_TARGETS = (
    "speed_mps",
    "accel_from_speed_mps2",
    "steering_deg",
    "yaw_rate_rps",
)

TARGET_TO_VALID = {
    "speed_mps": "valid_speed",
    "accel_from_speed_mps2": "valid_accel_from_speed",
    "steering_deg": "valid_steer",
    "yaw_rate_rps": "valid_yaw",
}
