# Coordinate Systems

This document describes the different coordinate systems used in the pipeline and how to convert between them.

## Overview

The pipeline uses three main coordinate systems:

1. **Image/Pixel Coordinates** - 2D detection output
2. **MotionBERT Camera Space** - 3D lifting output
3. **Blender World Space** - FBX export

## 1. Image/Pixel Coordinates

Used by: YOLOv8-Pose, 2D keypoints

```
Origin: Top-left corner
X-axis: → Right (0 to width)
Y-axis: ↓ Down (0 to height)

     (0,0) ─────────────────→ X (width)
       │
       │      Person
       │       /|\
       │      / | \
       │     /  |  \
       │        |
       ↓       / \
       Y
    (height)
```

### Normalization for MotionBERT

Before feeding to MotionBERT, 2D keypoints are normalized:

```python
# Normalize to [-1, 1] range centered at image center
kpts_norm = np.zeros_like(kpts)
kpts_norm[..., 0] = (kpts[..., 0] - width/2) / (min(width, height)/2)
kpts_norm[..., 1] = (kpts[..., 1] - height/2) / (min(width, height)/2)
```

## 2. MotionBERT Camera Space

Used by: MotionBERT 3D output

```
Origin: Camera center (approximately at pelvis/hip)
X-axis: → Right (positive = subject's left)
Y-axis: ↓ Down (positive = towards feet)
Z-axis: → Away from camera (depth, positive = further)

       Camera
         │
         │
         ▼
    ─────┼───── X
         │
         │ ↓ Y (down)
         │
         └───→ Z (depth)
```

**Key characteristics:**
- Y increases downward (towards feet)
- Z increases away from camera
- Normalized scale (not in real-world units)
- First frame hip Z is set to 0

### Sample output values:
```
Hip:    (0.50, 0.04, 0.00)   # Near center, slightly down from origin
Head:   (0.52, -0.39, -0.04)  # Above hip (negative Y)
RFoot:  (0.45, 0.46, 0.19)    # Below hip (positive Y)
```

## 3. Blender World Space

Used by: FBX export, Blender animation

```
Origin: Ground level, centered
X-axis: → Right
Y-axis: → Forward (depth)
Z-axis: ↑ Up (height)

           Z (up)
           ↑
           │   Person
           │    /|\
           │   / | \
           │  /  |  \
           │     |
           │    / \
    ───────┼─────────→ Y (forward/depth)
          /│
         / │
        ↙  │
       X (right)
```

**Key characteristics:**
- Z-up convention (standard for 3D software)
- Ground at Z=0
- Scaled to real-world units (meters)

## Transformation: MotionBERT → Blender

```python
def transform_to_blender(kp3d_motionbert, target_height=1.75):
    """
    Convert MotionBERT camera space to Blender world space.

    MotionBERT: X=right, Y=down, Z=depth
    Blender:    X=right, Y=depth, Z=up
    """
    result = np.zeros_like(kp3d_motionbert)

    # Coordinate swap
    result[..., 0] = kp3d_motionbert[..., 0]   # X stays X
    result[..., 1] = kp3d_motionbert[..., 2]   # Z becomes Y (depth)
    result[..., 2] = -kp3d_motionbert[..., 1]  # -Y becomes Z (up)

    # Center horizontally
    hip_x_mean = result[:, 0, 0].mean()
    result[..., 0] -= hip_x_mean

    # Ground feet (Z=0)
    foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
    result[..., 2] -= foot_z_min

    # Scale to target height
    head_z = result[:, 10, 2].mean()
    foot_z = min(result[:, 3, 2].mean(), result[:, 6, 2].mean())
    current_height = head_z - foot_z

    if current_height > 0.1:
        scale = target_height / current_height
        result *= scale

        # Re-ground after scaling
        foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
        result[..., 2] -= foot_z_min

    return result
```

### Transformation Matrix

The coordinate transformation can also be expressed as a matrix:

```python
# MotionBERT to Blender transformation matrix
# [X_blender]   [1  0  0] [X_mb]
# [Y_blender] = [0  0  1] [Y_mb]
# [Z_blender]   [0 -1  0] [Z_mb]

transform_matrix = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])

kp3d_blender = kp3d_motionbert @ transform_matrix.T
```

## Other Coordinate Systems

### Unity (Left-handed, Y-up)

```python
# MotionBERT to Unity
result[..., 0] = kp3d[..., 0]    # X stays X
result[..., 1] = -kp3d[..., 1]   # -Y becomes Y (up)
result[..., 2] = kp3d[..., 2]    # Z stays Z (depth)
```

### Unreal Engine (Z-up, centimeters)

```python
# MotionBERT to Unreal (meters to centimeters)
result[..., 0] = kp3d[..., 0] * 100    # X in cm
result[..., 1] = kp3d[..., 2] * 100    # Z becomes Y
result[..., 2] = -kp3d[..., 1] * 100   # -Y becomes Z
```

## Handedness

| System | Handedness | Up Axis |
|--------|------------|---------|
| MotionBERT | Right-handed | -Y (down is positive) |
| Blender | Right-handed | +Z |
| Unity | Left-handed | +Y |
| Unreal | Left-handed | +Z |

## Common Issues

### Skeleton appears upside down
- Check Y-axis flip in transformation
- Ensure `result[..., 2] = -kp3d[..., 1]` (negate Y)

### Skeleton faces wrong direction
- Swap or negate the depth axis
- Check if camera was behind the subject

### Scale is wrong
- MotionBERT outputs in normalized coordinates
- Must scale to real-world units (default: 1.75m height)

### Skeleton floating or underground
- Ground the feet after transformation
- `result[..., 2] -= result[:, [3,6], 2].min()`
