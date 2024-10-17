from pathlib import Path
import importlib.resources as resources

### Simulation envs fixed constants
DT = 0.02  # 0.02 ms -> 1/0.2 = 50 hz
FPS = 50


JOINTS = [
    "joint_left_wheel",
    "joint_right_wheel",
    "lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
    "head_pan",
    "head_tilt",
]

ACTIONS = [
    "left_wheel_vel",
    "right_wheel_vel",
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
    "head_pan",
    "head_tilt",
]

# 0 0 0.44 0.05 0 0 0 0 0 0
START_POSE = [
    0,
    0,
    0.44,
    0.05,
    0,
    0,
    0,
    0,
    0,
    0,
]


ASSETS_DIR = Path(__file__).parent.resolve() / "models"  # note: absolute path
# STRETCH_MODEL_DIR = resources.files('stretch_mujoco') / 'stretch_mujoco/models'

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2

############################ Helper functions ############################


def normalize_master_gripper_position(x):
    return (x - MASTER_GRIPPER_POSITION_CLOSE) / (
        MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
    )


def normalize_puppet_gripper_position(x):
    return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
        PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
    )


def unnormalize_master_gripper_position(x):
    return x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE


def unnormalize_puppet_gripper_position(x):
    return x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE


def convert_position_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_position(normalize_master_gripper_position(x))


def normalizer_master_gripper_joint(x):
    return (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)


def normalize_puppet_gripper_joint(x):
    return (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)


def unnormalize_master_gripper_joint(x):
    return x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE


def unnormalize_puppet_gripper_joint(x):
    return x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE


def convert_join_from_master_to_puppet(x):
    return unnormalize_puppet_gripper_joint(normalizer_master_gripper_joint(x))


def normalize_master_gripper_velocity(x):
    return x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)


def normalize_puppet_gripper_velocity(x):
    return x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def convert_master_from_position_to_joint(x):
    return (
        normalize_master_gripper_position(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
        + MASTER_GRIPPER_JOINT_CLOSE
    )


def convert_master_from_joint_to_position(x):
    return unnormalize_master_gripper_position(
        (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    )


def convert_puppet_from_position_to_join(x):
    return (
        normalize_puppet_gripper_position(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
        + PUPPET_GRIPPER_JOINT_CLOSE
    )


def convert_puppet_from_joint_to_position(x):
    return unnormalize_puppet_gripper_position(
        (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    )
