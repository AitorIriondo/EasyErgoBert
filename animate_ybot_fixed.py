"""
Animate Y-Bot with motion capture data from MotionBERT.

This script:
1. Loads the Y-Bot FBX with its Mixamo armature
2. Loads 3D pose data from the pipeline
3. Retargets the motion to the Mixamo skeleton
4. Exports animated FBX

Usage:
    blender --background --python animate_ybot_fixed.py -- \
        --input output_fixed/aitor_garden_walk_poses.pkl \
        --output output_fixed/ybot_animated.fbx
"""
import bpy
import pickle
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler
import math
import sys
import os
import argparse

# H36M joint names
H36M_JOINTS = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Mapping from H36M joints to Mixamo bones
# H36M index -> Mixamo bone name
H36M_TO_MIXAMO = {
    0: 'mixamorig:Hips',
    1: 'mixamorig:RightUpLeg',
    2: 'mixamorig:RightLeg',
    3: 'mixamorig:RightFoot',
    4: 'mixamorig:LeftUpLeg',
    5: 'mixamorig:LeftLeg',
    6: 'mixamorig:LeftFoot',
    7: 'mixamorig:Spine',
    8: 'mixamorig:Spine2',
    9: 'mixamorig:Neck',
    10: 'mixamorig:Head',
    11: 'mixamorig:LeftArm',
    12: 'mixamorig:LeftForeArm',
    13: 'mixamorig:LeftHand',
    14: 'mixamorig:RightArm',
    15: 'mixamorig:RightForeArm',
    16: 'mixamorig:RightHand',
}

# Bone chains for rotation calculation
# (parent_h36m_idx, child_h36m_idx, mixamo_bone)
BONE_CHAINS = [
    # Spine
    (0, 7, 'mixamorig:Spine'),
    (7, 8, 'mixamorig:Spine1'),
    (7, 8, 'mixamorig:Spine2'),
    (8, 9, 'mixamorig:Neck'),
    (9, 10, 'mixamorig:Head'),
    # Left leg
    (0, 4, 'mixamorig:LeftUpLeg'),
    (4, 5, 'mixamorig:LeftLeg'),
    (5, 6, 'mixamorig:LeftFoot'),
    # Right leg
    (0, 1, 'mixamorig:RightUpLeg'),
    (1, 2, 'mixamorig:RightLeg'),
    (2, 3, 'mixamorig:RightFoot'),
    # Left arm
    (8, 11, 'mixamorig:LeftShoulder'),
    (11, 12, 'mixamorig:LeftArm'),
    (12, 13, 'mixamorig:LeftForeArm'),
    # Right arm
    (8, 14, 'mixamorig:RightShoulder'),
    (14, 15, 'mixamorig:RightArm'),
    (15, 16, 'mixamorig:RightForeArm'),
]


def transform_to_blender(kp3d, target_height=1.75):
    """Transform MotionBERT coordinates to Blender Z-up."""
    n_frames = kp3d.shape[0]
    result = np.zeros_like(kp3d)

    for i in range(n_frames):
        frame = kp3d[i]
        result[i, :, 0] = frame[:, 0]
        result[i, :, 1] = frame[:, 2]
        result[i, :, 2] = -frame[:, 1]

    # Center at hip
    hip_x = result[:, 0, 0].mean()
    result[:, :, 0] -= hip_x

    # Ground feet
    foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
    result[:, :, 2] -= foot_z_min

    # Scale to target height
    head_z = result[:, 10, 2].mean()
    foot_z = min(result[:, 3, 2].mean(), result[:, 6, 2].mean())
    current_height = head_z - foot_z

    if current_height > 0.1:
        scale = target_height / current_height
        result *= scale
        foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
        result[:, :, 2] -= foot_z_min

    return result


def get_bone_rotation(parent_pos, child_pos, bone, rest_direction=None):
    """
    Calculate rotation to orient bone from parent to child position.
    """
    direction = Vector(child_pos) - Vector(parent_pos)
    length = direction.length

    if length < 0.001:
        return Quaternion()

    direction.normalize()

    # Get the bone's rest direction (Y-axis in bone local space for Mixamo)
    if rest_direction is None:
        rest_direction = Vector((0, 1, 0))

    # Calculate rotation from rest to target direction
    rotation = rest_direction.rotation_difference(direction)

    return rotation


def compute_bone_rotations(frame_data, armature_obj):
    """
    Compute bone rotations for a single frame.
    Returns dict of bone_name -> Quaternion
    """
    rotations = {}

    # Get pose bones
    pose_bones = armature_obj.pose.bones

    for parent_idx, child_idx, bone_name in BONE_CHAINS:
        if bone_name not in pose_bones:
            continue

        parent_pos = frame_data[parent_idx]
        child_pos = frame_data[child_idx]

        # Get bone's rest pose direction
        bone = pose_bones[bone_name]

        # Calculate direction in world space
        direction = Vector(child_pos) - Vector(parent_pos)
        if direction.length < 0.001:
            continue
        direction.normalize()

        # For Mixamo, bones generally point along Y axis in rest pose
        # We need to find the rotation that aligns Y with our direction
        rest_dir = Vector((0, 1, 0))

        # Account for bone's rest rotation
        if bone.bone.matrix_local:
            rest_dir = bone.bone.matrix_local.to_3x3() @ rest_dir
            rest_dir.normalize()

        # Calculate rotation
        rot = rest_dir.rotation_difference(direction)
        rotations[bone_name] = rot

    return rotations


def animate_armature(armature_obj, kp3d, fps=30, frame_step=1):
    """Apply animation to armature using position-based approach."""
    n_frames = kp3d.shape[0]
    total_anim_frames = (n_frames + frame_step - 1) // frame_step

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_anim_frames
    bpy.context.scene.render.fps = int(fps)

    # Set armature as active
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    print(f"Animating {total_anim_frames} frames...")

    for frame_idx in range(0, n_frames, frame_step):
        anim_frame = frame_idx // frame_step + 1
        bpy.context.scene.frame_set(anim_frame)

        frame_data = kp3d[frame_idx]

        # Move hips (root motion)
        hip_bone = armature_obj.pose.bones.get('mixamorig:Hips')
        if hip_bone:
            hip_pos = Vector(frame_data[0])
            hip_bone.location = hip_pos
            hip_bone.keyframe_insert(data_path="location", frame=anim_frame)

        # Compute and apply rotations for each bone chain
        for parent_idx, child_idx, bone_name in BONE_CHAINS:
            bone = armature_obj.pose.bones.get(bone_name)
            if not bone:
                continue

            parent_pos = Vector(frame_data[parent_idx])
            child_pos = Vector(frame_data[child_idx])

            direction = child_pos - parent_pos
            if direction.length < 0.001:
                continue
            direction.normalize()

            # Get bone's rest direction in armature space
            bone_matrix = bone.bone.matrix_local
            rest_dir = bone_matrix.to_3x3() @ Vector((0, 1, 0))
            rest_dir.normalize()

            # Calculate rotation needed
            rot_diff = rest_dir.rotation_difference(direction)

            # Apply as delta rotation
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rot_diff
            bone.keyframe_insert(data_path="rotation_quaternion", frame=anim_frame)

        if (frame_idx + 1) % 200 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Animation complete!")


def animate_armature_ik(armature_obj, kp3d, fps=30, frame_step=1):
    """
    Alternative: Animate using bone positions directly.
    This moves bones to match joint positions.
    """
    n_frames = kp3d.shape[0]
    total_anim_frames = (n_frames + frame_step - 1) // frame_step

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_anim_frames
    bpy.context.scene.render.fps = int(fps)

    bpy.context.view_layer.objects.active = armature_obj

    print(f"Animating {total_anim_frames} frames with position tracking...")

    # Create empties for each joint to use as targets
    empties = {}
    for idx, joint_name in enumerate(H36M_JOINTS):
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.02)
        empty = bpy.context.active_object
        empty.name = f"Target_{joint_name}"
        empties[idx] = empty

    # Animate empties
    for frame_idx in range(0, n_frames, frame_step):
        anim_frame = frame_idx // frame_step + 1
        frame_data = kp3d[frame_idx]

        for idx, empty in empties.items():
            empty.location = Vector(frame_data[idx])
            empty.keyframe_insert(data_path="location", frame=anim_frame)

    # Now we need to make bones follow the empties
    # The simplest approach is to use the hip position for root motion
    # and calculate limb rotations based on IK-like solving

    bpy.ops.object.mode_set(mode='POSE')

    for frame_idx in range(0, n_frames, frame_step):
        anim_frame = frame_idx // frame_step + 1
        bpy.context.scene.frame_set(anim_frame)

        frame_data = kp3d[frame_idx]

        # Root motion (hips)
        hip_bone = armature_obj.pose.bones.get('mixamorig:Hips')
        if hip_bone:
            hip_bone.location = Vector(frame_data[0])
            hip_bone.keyframe_insert(data_path="location", frame=anim_frame)

            # Hip rotation based on leg positions
            left_hip = Vector(frame_data[4])
            right_hip = Vector(frame_data[1])
            hip_dir = (left_hip - right_hip).normalized()

            # Calculate hip rotation
            forward = Vector((0, 1, 0))
            up = Vector((0, 0, 1))
            right = hip_dir
            forward = up.cross(right).normalized()

            rot_matrix = Matrix((right, forward, up)).transposed()
            hip_bone.rotation_mode = 'QUATERNION'
            hip_bone.rotation_quaternion = rot_matrix.to_quaternion()
            hip_bone.keyframe_insert(data_path="rotation_quaternion", frame=anim_frame)

        if (frame_idx + 1) % 200 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Clean up empties
    for empty in empties.values():
        bpy.data.objects.remove(empty)

    print("Animation complete!")


def simple_bone_animation(armature_obj, kp3d, fps=30, frame_step=1):
    """
    Simplified animation: Only animate root position and major bone rotations.
    """
    n_frames = kp3d.shape[0]
    total_anim_frames = (n_frames + frame_step - 1) // frame_step

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_anim_frames
    bpy.context.scene.render.fps = int(fps)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    print(f"Animating {total_anim_frames} frames...")

    # Bone mapping with parent-child relationships
    bone_targets = [
        # (bone_name, parent_joint_idx, child_joint_idx)
        ('mixamorig:Hips', 0, 7),  # Hip to Spine direction
        ('mixamorig:Spine', 0, 7),
        ('mixamorig:Spine1', 7, 8),
        ('mixamorig:Spine2', 7, 8),
        ('mixamorig:Neck', 8, 9),
        ('mixamorig:Head', 9, 10),
        ('mixamorig:LeftUpLeg', 4, 5),
        ('mixamorig:LeftLeg', 5, 6),
        ('mixamorig:RightUpLeg', 1, 2),
        ('mixamorig:RightLeg', 2, 3),
        ('mixamorig:LeftArm', 11, 12),
        ('mixamorig:LeftForeArm', 12, 13),
        ('mixamorig:RightArm', 14, 15),
        ('mixamorig:RightForeArm', 15, 16),
    ]

    for frame_idx in range(0, n_frames, frame_step):
        anim_frame = frame_idx // frame_step + 1
        frame_data = kp3d[frame_idx]

        # Animate hip position (root motion)
        hip_bone = armature_obj.pose.bones.get('mixamorig:Hips')
        if hip_bone:
            hip_bone.location = Vector(frame_data[0])
            hip_bone.keyframe_insert(data_path="location", frame=anim_frame)

        # Animate bone rotations
        for bone_name, parent_idx, child_idx in bone_targets:
            bone = armature_obj.pose.bones.get(bone_name)
            if not bone:
                continue

            # Direction from parent joint to child joint
            parent_pos = Vector(frame_data[parent_idx])
            child_pos = Vector(frame_data[child_idx])
            target_dir = (child_pos - parent_pos).normalized()

            if target_dir.length < 0.001:
                continue

            # Bone rest direction (typically Y-axis for Mixamo)
            rest_dir = Vector((0, 1, 0))

            # Get bone's rest pose matrix
            rest_matrix = bone.bone.matrix_local.to_3x3()
            bone_rest_dir = rest_matrix @ rest_dir
            bone_rest_dir.normalize()

            # Compute rotation from rest to target
            rotation = bone_rest_dir.rotation_difference(target_dir)

            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rotation
            bone.keyframe_insert(data_path="rotation_quaternion", frame=anim_frame)

        if (frame_idx + 1) % 200 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")

    bpy.ops.object.mode_set(mode='OBJECT')
    print("Animation complete!")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Animate Y-Bot with motion data")
    parser.add_argument('--input', '-i', required=True, help='Input pickle file')
    parser.add_argument('--output', '-o', required=True, help='Output FBX file')
    parser.add_argument('--ybot', default=r'C:\VideoPoseEstimation\y_bot.fbx', help='Y-Bot FBX path')
    parser.add_argument('--height', type=float, default=1.75, help='Target height')
    parser.add_argument('--frame-step', type=int, default=1, help='Frame step')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames (0=all)')

    args = parser.parse_args(argv)

    print("=" * 60)
    print("Y-Bot Animation from MotionBERT Data")
    print("=" * 60)

    # Load pose data
    print(f"\nLoading: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    kp3d = np.array(data['keypoints_3d'])
    fps = data.get('fps', 30.0)

    if args.max_frames > 0:
        kp3d = kp3d[:args.max_frames]

    print(f"Frames: {kp3d.shape[0]}, FPS: {fps}")

    # Transform coordinates
    print(f"\nTransforming coordinates (height: {args.height}m)...")
    kp3d_blender = transform_to_blender(kp3d, args.height)

    # Clear scene and load Y-Bot
    print(f"\nLoading Y-Bot: {args.ybot}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=args.ybot)

    # Find armature
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break

    if not armature:
        print("ERROR: No armature found in Y-Bot!")
        return

    print(f"Found armature: {armature.name}")

    # Apply animation
    print("\nApplying animation...")
    simple_bone_animation(armature, kp3d_blender, fps=fps, frame_step=args.frame_step)

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # Save blend file
    blend_path = args.output.replace('.fbx', '_ybot.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"\nSaved: {blend_path}")

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
    )
    print(f"Exported: {args.output}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
