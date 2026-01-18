"""
Export 3D skeleton to OBJ files.

Creates simple 3D geometry (spheres for joints, cylinders for bones)
that can be opened in Blender, Maya, or any 3D software.

Coordinate System Notes:
- MotionBERT outputs normalized 3D coordinates in a camera-relative frame.
  The skeleton orientation varies per frame based on camera position.
  The anatomical "up" direction (foot→head) can be along any axis.

- Blender uses Z-up right-handed coordinate system:
  - X: left-right (positive = right)
  - Y: front-back (positive = forward)
  - Z: up-down (positive = up)

- Conversion (auto mode): Computes the anatomical up direction from the
  foot-to-head vector and rotates the skeleton so this aligns with Z+.
  This ensures the skeleton stands upright in Blender regardless of the
  original orientation in the MotionBERT output.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
import pickle


# H36M skeleton connections (parent -> child)
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # Right leg: Hip -> RHip -> RKnee -> RFoot
    (0, 4), (4, 5), (5, 6),      # Left leg: Hip -> LHip -> LKnee -> LFoot
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine: Hip -> Spine -> Thorax -> Neck -> Head
    (8, 11), (11, 12), (12, 13), # Left arm: Thorax -> LShoulder -> LElbow -> LWrist
    (8, 14), (14, 15), (15, 16)  # Right arm: Thorax -> RShoulder -> RElbow -> RWrist
]

H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Joint indices for computing skeleton height
HEAD_IDX = 10
LFOOT_IDX = 6
RFOOT_IDX = 3


def compute_up_direction(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute the natural 'up' direction of the skeleton based on anatomy.

    Uses the vector from feet to head as the primary up direction,
    with hip->thorax/shoulders as secondary signals.

    Args:
        keypoints: Array of shape (17, 3)

    Returns:
        Normalized up direction vector
    """
    # Get key anatomical points
    hip = keypoints[0]  # Hip/pelvis
    thorax = keypoints[8]  # Thorax
    lshoulder = keypoints[11]
    rshoulder = keypoints[14]
    neck = keypoints[9]
    head = keypoints[10]
    lfoot = keypoints[6]
    rfoot = keypoints[3]

    # Compute foot center
    foot_center = (lfoot + rfoot) / 2.0

    # Primary up direction: from feet to head
    up_primary = head - foot_center

    # Secondary: from hip to shoulder center
    shoulder_center = (lshoulder + rshoulder) / 2.0
    up_secondary = shoulder_center - hip

    # Tertiary: from hip to neck
    up_tertiary = neck - hip

    # Weighted average (feet-to-head is most reliable)
    up_dir = 0.5 * up_primary + 0.3 * up_secondary + 0.2 * up_tertiary

    norm = np.linalg.norm(up_dir)
    if norm > 1e-6:
        up_dir = up_dir / norm
    else:
        up_dir = np.array([0, 0, 1])  # Default up if computation fails

    return up_dir


def compute_forward_direction(keypoints: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
    """
    Compute the forward direction of the skeleton (facing direction).

    Uses the cross product of shoulder vector and up direction.

    Args:
        keypoints: Array of shape (17, 3)
        up_dir: The skeleton's up direction

    Returns:
        Normalized forward direction vector
    """
    lshoulder = keypoints[11]
    rshoulder = keypoints[14]

    # Shoulder vector (left to right)
    shoulder_vec = rshoulder - lshoulder
    norm = np.linalg.norm(shoulder_vec)
    if norm > 1e-6:
        shoulder_vec = shoulder_vec / norm
    else:
        shoulder_vec = np.array([1, 0, 0])

    # Forward is cross product of up and shoulder (right-hand rule)
    forward = np.cross(up_dir, shoulder_vec)
    norm = np.linalg.norm(forward)
    if norm > 1e-6:
        forward = forward / norm
    else:
        forward = np.array([0, 1, 0])

    return forward


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to rotate vec_from to vec_to.

    Uses Rodrigues' rotation formula.
    """
    a = vec_from / np.linalg.norm(vec_from)
    b = vec_to / np.linalg.norm(vec_to)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if c < -0.9999:
        # Vectors are nearly opposite, use a different axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        return -np.eye(3) + 2 * np.outer(perp, perp)

    if np.linalg.norm(v) < 1e-6:
        return np.eye(3)  # Vectors are already aligned

    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))


def detect_vertical_axis(keypoints: np.ndarray) -> Tuple[int, int]:
    """
    Detect which axis is vertical and its direction based on anatomical constraints.

    For a human skeleton, the vertical axis should have:
    - Head above hip
    - Hip above feet

    Args:
        keypoints: Array of shape (17, 3)

    Returns:
        Tuple of (axis_index, direction) where:
        - axis_index: 0, 1, or 2 for X, Y, Z
        - direction: 1 (positive is up) or -1 (negative is up)
    """
    hip = keypoints[0]
    head = keypoints[10]
    lfoot = keypoints[6]
    rfoot = keypoints[3]
    foot_center = (lfoot + rfoot) / 2.0

    # Check each axis
    best_axis = 2  # Default to Z
    best_direction = 1
    best_score = -1

    for axis in range(3):
        # Check if head > hip > feet (positive up)
        head_hip = head[axis] - hip[axis]
        hip_foot = hip[axis] - foot_center[axis]

        # Both should be positive for "up" direction
        score_pos = (1 if head_hip > 0 else 0) + (1 if hip_foot > 0 else 0)
        score_pos += abs(head_hip) + abs(hip_foot)  # Magnitude bonus

        # Check if head < hip < feet (negative up, i.e., inverted)
        score_neg = (1 if head_hip < 0 else 0) + (1 if hip_foot < 0 else 0)
        score_neg += abs(head_hip) + abs(hip_foot)

        if score_pos > best_score:
            best_score = score_pos
            best_axis = axis
            best_direction = 1

        if score_neg > best_score:
            best_score = score_neg
            best_axis = axis
            best_direction = -1

    return best_axis, best_direction


def convert_to_blender_coords(keypoints: np.ndarray, coord_mode: str = 'auto') -> np.ndarray:
    """
    Convert skeleton to Blender coordinate system (Z-up).

    Supports multiple coordinate system modes:
    - 'auto': Automatically rotate skeleton so foot-to-head points up (Z+)
    - 'motionbert': Same as auto - rotate based on anatomical up direction
    - 'y_up': Assume input is Y-up (common in 3D software)
    - 'z_up': Assume input is already Z-up (no change)
    - 'camera': Assume camera space (X-right, Y-down, Z-forward)

    Note: MotionBERT outputs coordinates in a camera-relative frame where
    the skeleton can be oriented in any direction. The 'auto' mode detects
    the anatomical "up" direction (foot→head) and rotates the skeleton
    to align with Blender's Z+ axis.

    Args:
        keypoints: Array of shape (17, 3) or (T, 17, 3)
        coord_mode: Coordinate system mode

    Returns:
        Transformed keypoints in Blender Z-up coordinate system
    """
    if keypoints.ndim == 3:  # Sequence (T, 17, 3)
        converted = np.zeros_like(keypoints)
        for t in range(keypoints.shape[0]):
            converted[t] = convert_to_blender_coords(keypoints[t], coord_mode)
        return converted

    kpts = keypoints.copy()

    if coord_mode == 'z_up':
        # Already Z-up, no conversion needed
        return kpts

    elif coord_mode == 'y_up':
        # Standard Y-up to Z-up: swap Y and Z
        result = np.zeros_like(kpts)
        result[:, 0] = kpts[:, 0]   # X stays X
        result[:, 1] = -kpts[:, 2]  # Y = -Z (flip depth)
        result[:, 2] = kpts[:, 1]   # Z = Y (height)
        return result

    elif coord_mode == 'camera':
        # Camera space: X-right, Y-down, Z-forward
        # To Blender: X-right, Y-forward, Z-up
        result = np.zeros_like(kpts)
        result[:, 0] = kpts[:, 0]   # X stays X
        result[:, 1] = kpts[:, 2]   # Y = Z (depth to forward)
        result[:, 2] = -kpts[:, 1]  # Z = -Y (flip down to up)
        return result

    else:  # 'auto' or 'motionbert' mode
        # MotionBERT outputs in camera-relative coordinates where the
        # skeleton can be oriented in any direction. We rotate it so
        # the foot-to-head direction aligns with Blender's Z+ axis.

        # Compute anatomical up direction (foot center to head)
        head = kpts[10]  # Head joint
        lfoot = kpts[6]  # Left foot
        rfoot = kpts[3]  # Right foot
        foot_center = (lfoot + rfoot) / 2.0

        up_vec = head - foot_center
        up_norm = np.linalg.norm(up_vec)

        if up_norm < 1e-6:
            # Degenerate case - return as-is
            return kpts

        up_dir = up_vec / up_norm

        # Target: Blender Z+ axis
        target_up = np.array([0.0, 0.0, 1.0])

        # Compute rotation matrix using Rodrigues' formula
        R = rotation_matrix_from_vectors(up_dir, target_up)

        # Apply rotation to all joints
        result = (R @ kpts.T).T

        return result


def compute_skeleton_height(keypoints: np.ndarray) -> float:
    """
    Compute the height of the skeleton from feet to head.

    Args:
        keypoints: Array of shape (17, 3) with joint positions

    Returns:
        Height of skeleton (distance from lowest foot to head)
    """
    head = keypoints[HEAD_IDX]
    lfoot = keypoints[LFOOT_IDX]
    rfoot = keypoints[RFOOT_IDX]

    # Use the lower foot
    foot_z = min(lfoot[2], rfoot[2])
    foot_pos = lfoot if lfoot[2] <= rfoot[2] else rfoot

    # Compute height as vertical distance (Z in Blender coords)
    height = abs(head[2] - foot_z)

    # If height seems wrong (e.g., person lying down), use max extent
    if height < 0.1:  # Less than 10% of normalized range
        all_z = keypoints[:, 2]
        height = all_z.max() - all_z.min()

    return max(height, 0.01)  # Ensure non-zero


def normalize_and_scale_skeleton(keypoints: np.ndarray,
                                  target_height: float = 1.7) -> Tuple[np.ndarray, float]:
    """
    Normalize skeleton to have feet at ground level and scale to target height.

    Args:
        keypoints: Array of shape (17, 3) with joint positions (in Blender coords)
        target_height: Target height in meters (default 1.7m for average human)

    Returns:
        Tuple of (scaled_keypoints, scale_factor)
    """
    # Find current height
    current_height = compute_skeleton_height(keypoints)

    # Compute scale factor
    scale_factor = target_height / current_height if current_height > 0 else 1.0

    # Scale keypoints
    scaled = keypoints * scale_factor

    # Move feet to ground level (Z=0)
    min_z = min(scaled[LFOOT_IDX, 2], scaled[RFOOT_IDX, 2])
    scaled[:, 2] -= min_z

    # Center X and Y at origin (hip position)
    hip = scaled[0]  # Hip is joint 0
    scaled[:, 0] -= hip[0]
    scaled[:, 1] -= hip[1]

    return scaled, scale_factor


def create_sphere_vertices(center: np.ndarray, radius: float = 0.02,
                          segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple UV sphere mesh."""
    vertices = []
    faces = []

    # Generate vertices
    for i in range(segments + 1):
        lat = np.pi * i / segments
        for j in range(segments):
            lon = 2 * np.pi * j / segments
            x = center[0] + radius * np.sin(lat) * np.cos(lon)
            y = center[1] + radius * np.sin(lat) * np.sin(lon)
            z = center[2] + radius * np.cos(lat)
            vertices.append([x, y, z])

    # Generate faces
    for i in range(segments):
        for j in range(segments):
            p1 = i * segments + j
            p2 = i * segments + (j + 1) % segments
            p3 = (i + 1) * segments + (j + 1) % segments
            p4 = (i + 1) * segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)


def create_cylinder_vertices(start: np.ndarray, end: np.ndarray,
                            radius: float = 0.01, segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create a cylinder mesh between two points."""
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []
    faces = []

    # Generate circle vertices at start and end
    for t, center in enumerate([start, end]):
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    # Generate faces (side faces)
    for i in range(segments):
        p1 = i
        p2 = (i + 1) % segments
        p3 = segments + (i + 1) % segments
        p4 = segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    # Cap faces
    # Start cap
    center_start_idx = len(vertices)
    vertices.append(start)
    for i in range(segments):
        faces.append([center_start_idx, (i + 1) % segments, i])

    # End cap
    center_end_idx = len(vertices)
    vertices.append(end)
    for i in range(segments):
        faces.append([center_end_idx, segments + i, segments + (i + 1) % segments])

    return np.array(vertices), np.array(faces)


def skeleton_to_obj(keypoints_3d: np.ndarray,
                   output_path: str,
                   joint_radius: float = None,
                   bone_radius: float = None,
                   scale: float = 1.0,
                   target_height: float = 1.7,
                   convert_coords: bool = True,
                   coord_mode: str = 'auto',
                   auto_scale: bool = True) -> str:
    """
    Convert a single frame 3D skeleton to OBJ format.

    Args:
        keypoints_3d: Array of shape (17, 3) with 3D joint positions
        output_path: Path to save OBJ file
        joint_radius: Radius of joint spheres (auto-computed if None)
        bone_radius: Radius of bone cylinders (auto-computed if None)
        scale: Additional scale factor for the skeleton
        target_height: Target skeleton height in meters (default 1.7m)
        convert_coords: Convert from MotionBERT to Blender coordinates
        coord_mode: Coordinate mode ('auto', 'y_up', 'z_up', 'camera')
        auto_scale: Automatically scale skeleton to target height

    Returns:
        Path to saved OBJ file
    """
    keypoints = np.array(keypoints_3d).copy()

    # Step 1: Convert coordinate system (MotionBERT -> Blender)
    if convert_coords:
        keypoints = convert_to_blender_coords(keypoints, coord_mode=coord_mode)

    # Step 2: Normalize and scale to target height
    if auto_scale:
        keypoints, computed_scale = normalize_and_scale_skeleton(keypoints, target_height)
    else:
        keypoints = keypoints * scale

    # Step 3: Compute proportional radii if not specified
    skeleton_height = compute_skeleton_height(keypoints)
    if joint_radius is None:
        joint_radius = skeleton_height * 0.025  # 2.5% of height
    if bone_radius is None:
        bone_radius = skeleton_height * 0.015   # 1.5% of height

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # Create spheres for each joint
    for i, joint in enumerate(keypoints):
        if np.any(np.isnan(joint)) or np.allclose(joint, 0):
            continue

        verts, faces = create_sphere_vertices(joint, joint_radius)
        all_vertices.extend(verts)
        all_faces.extend(faces + vertex_offset)
        vertex_offset += len(verts)

    # Create cylinders for each bone
    for i, j in H36M_SKELETON:
        start, end = keypoints[i], keypoints[j]

        if np.any(np.isnan(start)) or np.any(np.isnan(end)):
            continue
        if np.allclose(start, 0) or np.allclose(end, 0):
            continue

        verts, faces = create_cylinder_vertices(start, end, bone_radius)
        if len(verts) > 0:
            all_vertices.extend(verts)
            all_faces.extend(faces + vertex_offset)
            vertex_offset += len(verts)

    # Write OBJ file
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(f"# 3D Skeleton OBJ\n")
        f.write(f"# Joints: 17 (H36M format)\n")
        f.write(f"# Skeleton height: {skeleton_height:.3f}m\n")
        f.write(f"# Joint radius: {joint_radius:.4f}m\n")
        f.write(f"# Bone radius: {bone_radius:.4f}m\n")
        f.write(f"# Coordinate system: Blender (Z-up)\n")
        f.write(f"# Generated by VideoPoseEstimation\n\n")

        # Joint reference (for debugging)
        f.write("# Joint positions:\n")
        for idx, name in enumerate(H36M_JOINT_NAMES):
            pos = keypoints[idx]
            f.write(f"#   {idx:2d}. {name:12s}: ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})\n")
        f.write("\n")

        # Write vertices
        for v in all_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces (OBJ uses 1-indexed)
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return str(output_path)


def export_skeleton_sequence(keypoints_3d: np.ndarray,
                            output_dir: str,
                            prefix: str = "skeleton",
                            joint_radius: float = None,
                            bone_radius: float = None,
                            scale: float = 1.0,
                            target_height: float = 1.7,
                            convert_coords: bool = True,
                            coord_mode: str = 'auto',
                            auto_scale: bool = True,
                            every_n_frames: int = 1) -> List[str]:
    """
    Export a sequence of 3D skeletons to OBJ files.

    Args:
        keypoints_3d: Array of shape (T, 17, 3) with 3D joint positions
        output_dir: Directory to save OBJ files
        prefix: Filename prefix
        joint_radius: Radius of joint spheres (auto-computed if None)
        bone_radius: Radius of bone cylinders (auto-computed if None)
        scale: Additional scale factor
        target_height: Target skeleton height in meters (default 1.7m)
        convert_coords: Convert from MotionBERT to Blender coordinates
        coord_mode: Coordinate mode ('auto', 'y_up', 'z_up', 'camera')
        auto_scale: Automatically scale skeleton to target height
        every_n_frames: Export every N frames (1 = all frames)

    Returns:
        List of paths to saved OBJ files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    total_frames = len(keypoints_3d)
    export_count = (total_frames + every_n_frames - 1) // every_n_frames

    print(f"Exporting {export_count} OBJ files...")
    print(f"  Target height: {target_height}m")
    print(f"  Coordinate mode: {coord_mode}")

    for frame_idx in range(0, total_frames, every_n_frames):
        output_path = output_dir / f"{prefix}_{frame_idx:06d}.obj"
        skeleton_to_obj(
            keypoints_3d[frame_idx],
            output_path,
            joint_radius=joint_radius,
            bone_radius=bone_radius,
            scale=scale,
            target_height=target_height,
            convert_coords=convert_coords,
            coord_mode=coord_mode,
            auto_scale=auto_scale
        )
        saved_files.append(str(output_path))

        if (frame_idx + 1) % 100 == 0:
            print(f"  Exported {frame_idx + 1}/{total_frames} frames")

    print(f"Saved {len(saved_files)} OBJ files to {output_dir}")
    return saved_files


def export_skeleton_animated_obj(keypoints_3d: np.ndarray,
                                 output_path: str,
                                 fps: float = 30.0,
                                 scale: float = 1.0,
                                 target_height: float = 1.7,
                                 convert_coords: bool = True,
                                 auto_scale: bool = True) -> str:
    """
    Export skeleton sequence as a single OBJ with vertex animation data.
    Also creates a text file with per-frame joint positions.

    Args:
        keypoints_3d: Array of shape (T, 17, 3)
        output_path: Path for output files
        fps: Frames per second
        scale: Additional scale factor
        target_height: Target skeleton height in meters
        convert_coords: Convert from MotionBERT to Blender coordinates
        auto_scale: Automatically scale skeleton to target height

    Returns:
        Path to saved OBJ file
    """
    output_path = Path(output_path)

    # Export first frame as base OBJ (with all transformations)
    skeleton_to_obj(
        keypoints_3d[0],
        output_path,
        scale=scale,
        target_height=target_height,
        convert_coords=convert_coords,
        auto_scale=auto_scale
    )

    # Process all frames for animation data
    all_frames = []
    for frame in keypoints_3d:
        kpts = frame.copy()
        if convert_coords:
            kpts = convert_to_blender_coords(kpts)
        if auto_scale:
            kpts, _ = normalize_and_scale_skeleton(kpts, target_height)
        else:
            kpts = kpts * scale
        all_frames.append(kpts)

    # Create simple vertex position file (custom format)
    anim_path = output_path.with_suffix('.txt')
    with open(anim_path, 'w') as f:
        f.write(f"# Skeleton Animation Data\n")
        f.write(f"# Frames: {len(keypoints_3d)}\n")
        f.write(f"# FPS: {fps}\n")
        f.write(f"# Joints: 17 (H36M format)\n")
        f.write(f"# Coordinate system: Blender (Z-up)\n\n")

        for frame_idx, frame in enumerate(all_frames):
            f.write(f"frame {frame_idx}\n")
            for joint_idx, joint in enumerate(frame):
                f.write(f"  {H36M_JOINT_NAMES[joint_idx]}: {joint[0]:.6f} {joint[1]:.6f} {joint[2]:.6f}\n")
            f.write("\n")

    print(f"Saved base OBJ: {output_path}")
    print(f"Saved animation data: {anim_path}")

    return str(output_path)


def load_and_export(results_path: str,
                   output_dir: str,
                   every_n_frames: int = 1,
                   scale: float = 1.0,
                   target_height: float = 1.7,
                   convert_coords: bool = True,
                   auto_scale: bool = True) -> List[str]:
    """
    Load results from pickle/json and export to OBJ.

    Args:
        results_path: Path to poses.pkl or poses.json
        output_dir: Directory to save OBJ files
        every_n_frames: Export every N frames
        scale: Additional scale factor
        target_height: Target skeleton height in meters
        convert_coords: Convert from MotionBERT to Blender coordinates
        auto_scale: Automatically scale skeleton to target height

    Returns:
        List of saved file paths
    """
    results_path = Path(results_path)

    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(results_path, 'r') as f:
            data = json.load(f)

    keypoints_3d = np.array(data['keypoints_3d'])

    if keypoints_3d is None or len(keypoints_3d) == 0:
        print("ERROR: No 3D keypoints found in results file")
        return []

    video_name = data.get('video_name', 'skeleton')

    return export_skeleton_sequence(
        keypoints_3d,
        output_dir,
        prefix=video_name,
        every_n_frames=every_n_frames,
        scale=scale,
        target_height=target_height,
        convert_coords=convert_coords,
        auto_scale=auto_scale
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Export 3D skeleton to OBJ files for Blender/Maya',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default settings (1.7m height, Blender coords)
  python -m utils.export_obj -i output/video_poses.pkl -o output/obj

  # Export every 5th frame with 1.8m height
  python -m utils.export_obj -i output/video_poses.pkl -o output/obj -e 5 --height 1.8

  # Export without coordinate conversion (raw MotionBERT output)
  python -m utils.export_obj -i output/video_poses.pkl -o output/obj --no-convert
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to poses.pkl or poses.json')
    parser.add_argument('--output', '-o', default='output/obj',
                        help='Output directory')
    parser.add_argument('--every', '-e', type=int, default=1,
                        help='Export every N frames (default: 1 = all frames)')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Additional scale factor (applied after auto-scaling)')
    parser.add_argument('--height', type=float, default=1.7,
                        help='Target skeleton height in meters (default: 1.7m)')
    parser.add_argument('--no-convert', action='store_true',
                        help='Disable coordinate conversion (keep raw MotionBERT coords)')
    parser.add_argument('--no-autoscale', action='store_true',
                        help='Disable automatic scaling to target height')

    args = parser.parse_args()

    load_and_export(
        args.input,
        args.output,
        every_n_frames=args.every,
        scale=args.scale,
        target_height=args.height,
        convert_coords=not args.no_convert,
        auto_scale=not args.no_autoscale
    )
