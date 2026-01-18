# H36M Skeleton Format

This document describes the Human3.6M (H36M) 17-joint skeleton format used throughout this pipeline.

## Joint Definitions

| Index | Name | Description | Parent |
|-------|------|-------------|--------|
| 0 | Hip | Pelvis/Root joint | - |
| 1 | RHip | Right Hip | Hip (0) |
| 2 | RKnee | Right Knee | RHip (1) |
| 3 | RFoot | Right Ankle/Foot | RKnee (2) |
| 4 | LHip | Left Hip | Hip (0) |
| 5 | LKnee | Left Knee | LHip (4) |
| 6 | LFoot | Left Ankle/Foot | LKnee (5) |
| 7 | Spine | Lower Spine | Hip (0) |
| 8 | Thorax | Upper Spine/Chest | Spine (7) |
| 9 | Neck | Neck base | Thorax (8) |
| 10 | Head | Head top | Neck (9) |
| 11 | LShoulder | Left Shoulder | Thorax (8) |
| 12 | LElbow | Left Elbow | LShoulder (11) |
| 13 | LWrist | Left Wrist | LElbow (12) |
| 14 | RShoulder | Right Shoulder | Thorax (8) |
| 15 | RElbow | Right Elbow | RShoulder (14) |
| 16 | RWrist | Right Wrist | RElbow (15) |

## Bone Connections

```python
H36M_BONES = [
    # Spine chain
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head

    # Left leg
    (0, 4),   # Hip -> LHip
    (4, 5),   # LHip -> LKnee
    (5, 6),   # LKnee -> LFoot

    # Right leg
    (0, 1),   # Hip -> RHip
    (1, 2),   # RHip -> RKnee
    (2, 3),   # RKnee -> RFoot

    # Left arm
    (8, 11),  # Thorax -> LShoulder
    (11, 12), # LShoulder -> LElbow
    (12, 13), # LElbow -> LWrist

    # Right arm
    (8, 14),  # Thorax -> RShoulder
    (14, 15), # RShoulder -> RElbow
    (15, 16), # RElbow -> RWrist
]
```

## Skeleton Hierarchy

```
Hip (0)
├── RHip (1)
│   └── RKnee (2)
│       └── RFoot (3)
├── LHip (4)
│   └── LKnee (5)
│       └── LFoot (6)
└── Spine (7)
    └── Thorax (8)
        ├── Neck (9)
        │   └── Head (10)
        ├── LShoulder (11)
        │   └── LElbow (12)
        │       └── LWrist (13)
        └── RShoulder (14)
            └── RElbow (15)
                └── RWrist (16)
```

## Expected Bone Lengths

For a 1.75m tall person, approximate bone lengths:

| Bone | Length (m) | Range (m) |
|------|------------|-----------|
| Hip → Spine | 0.13 | 0.10 - 0.25 |
| Spine → Thorax | 0.15 | 0.15 - 0.35 |
| Thorax → Neck | 0.15 | 0.10 - 0.25 |
| Neck → Head | 0.20 | 0.15 - 0.30 |
| Hip → RHip/LHip | 0.10 | 0.05 - 0.20 |
| RHip → RKnee (Thigh) | 0.45 | 0.35 - 0.55 |
| RKnee → RFoot (Shin) | 0.43 | 0.35 - 0.55 |
| Thorax → Shoulder | 0.20 | 0.10 - 0.30 |
| Shoulder → Elbow (Upper arm) | 0.30 | 0.25 - 0.40 |
| Elbow → Wrist (Forearm) | 0.26 | 0.20 - 0.35 |

## COCO to H36M Mapping

The 2D detector (YOLOv8-Pose) outputs COCO 17-joint format. The conversion to H36M is:

| H36M Index | H36M Name | COCO Index | COCO Name |
|------------|-----------|------------|-----------|
| 0 | Hip | (11+12)/2 | Mid-hip (computed) |
| 1 | RHip | 12 | Right Hip |
| 2 | RKnee | 14 | Right Knee |
| 3 | RFoot | 16 | Right Ankle |
| 4 | LHip | 11 | Left Hip |
| 5 | LKnee | 13 | Left Knee |
| 6 | LFoot | 15 | Left Ankle |
| 7 | Spine | (0+hip)/2 | Computed |
| 8 | Thorax | (5+6)/2 | Mid-shoulder |
| 9 | Neck | 0 | Nose (approximate) |
| 10 | Head | 0 | Nose (approximate) |
| 11 | LShoulder | 5 | Left Shoulder |
| 12 | LElbow | 7 | Left Elbow |
| 13 | LWrist | 9 | Left Wrist |
| 14 | RShoulder | 6 | Right Shoulder |
| 15 | RElbow | 8 | Right Elbow |
| 16 | RWrist | 10 | Right Wrist |

## Data Format

### Pickle File Structure

```python
{
    'video_name': str,
    'video_path': str,
    'resolution': (width, height),
    'fps': float,
    'total_frames': int,
    'keypoints_2d_coco': np.ndarray,    # Shape: (T, 17, 2) - Original COCO
    'keypoints_2d_h36m': np.ndarray,    # Shape: (T, 17, 2) - Converted H36M
    'keypoints_3d': np.ndarray,          # Shape: (T, 17, 3) - 3D positions
    'scores': np.ndarray,                # Shape: (T, 17) - Confidence scores
    'joint_names': list,                 # H36M joint names
    'skeleton': list,                    # Bone connections
}
```

### Accessing Data

```python
import pickle
import numpy as np

with open('output/video_poses.pkl', 'rb') as f:
    data = pickle.load(f)

# Get 3D keypoints
kp3d = data['keypoints_3d']  # Shape: (num_frames, 17, 3)

# Get specific joint across all frames
hip = kp3d[:, 0, :]       # Hip positions, shape (T, 3)
head = kp3d[:, 10, :]     # Head positions

# Get single frame
frame_100 = kp3d[100]     # Shape: (17, 3)

# Compute bone length
import numpy as np
hip_to_spine = np.linalg.norm(kp3d[:, 7] - kp3d[:, 0], axis=1)  # Per-frame length
```

## Symmetry

The skeleton has bilateral symmetry:

| Left Side | Right Side |
|-----------|------------|
| LHip (4) | RHip (1) |
| LKnee (5) | RKnee (2) |
| LFoot (6) | RFoot (3) |
| LShoulder (11) | RShoulder (14) |
| LElbow (12) | RElbow (15) |
| LWrist (13) | RWrist (16) |

For flip augmentation (test-time augmentation in MotionBERT):

```python
def flip_skeleton(kp3d):
    """Flip skeleton horizontally (mirror X axis)."""
    flipped = kp3d.copy()
    flipped[..., 0] = -flipped[..., 0]  # Negate X

    # Swap left-right joints
    swap_pairs = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]
    for left, right in swap_pairs:
        flipped[..., left, :], flipped[..., right, :] = \
            flipped[..., right, :].copy(), flipped[..., left, :].copy()

    return flipped
```
