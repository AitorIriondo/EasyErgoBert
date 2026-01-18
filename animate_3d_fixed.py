"""
Animate 3D skeleton from fixed MotionBERT output and export to FBX.

MotionBERT outputs in camera-centered normalized coordinates:
- X: horizontal (positive = right)
- Y: vertical (positive = down, i.e., towards feet)
- Z: depth (positive = away from camera)

This script transforms to Blender's Z-up coordinate system and scales
to realistic human proportions.

Usage:
    blender --background --python animate_3d_fixed.py -- --input output_fixed/aitor_garden_walk_poses.pkl --output output_fixed/animation.fbx
"""
import bpy
import pickle
import numpy as np
from mathutils import Vector, Matrix
import sys
import os
import argparse

H36M_JOINTS = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

H36M_BONES = [
    (0, 7),   # Hip -> Spine
    (7, 8),   # Spine -> Thorax
    (8, 9),   # Thorax -> Neck
    (9, 10),  # Neck -> Head
    (0, 4),   # Hip -> LHip
    (4, 5),   # LHip -> LKnee
    (5, 6),   # LKnee -> LFoot
    (0, 1),   # Hip -> RHip
    (1, 2),   # RHip -> RKnee
    (2, 3),   # RKnee -> RFoot
    (8, 11),  # Thorax -> LShoulder
    (11, 12), # LShoulder -> LElbow
    (12, 13), # LElbow -> LWrist
    (8, 14),  # Thorax -> RShoulder
    (14, 15), # RShoulder -> RElbow
    (15, 16), # RElbow -> RWrist
]


def transform_to_blender(kp3d, target_height=1.75):
    """
    Transform MotionBERT camera-space coordinates to Blender Z-up coordinates.

    MotionBERT: X=right, Y=down, Z=depth
    Blender:    X=right, Y=depth, Z=up

    Transform: X_blender = X_mb
               Y_blender = Z_mb
               Z_blender = -Y_mb (flip Y to get Z-up)
    """
    n_frames = kp3d.shape[0]
    result = np.zeros_like(kp3d)

    for i in range(n_frames):
        frame = kp3d[i]

        # Transform coordinates
        result[i, :, 0] = frame[:, 0]   # X stays X
        result[i, :, 1] = frame[:, 2]   # Z becomes Y (depth)
        result[i, :, 2] = -frame[:, 1]  # -Y becomes Z (up)

    # Center horizontally (X) at hip
    hip_x = result[:, 0, 0].mean()
    result[:, :, 0] -= hip_x

    # Put feet on ground (Z=0)
    foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
    result[:, :, 2] -= foot_z_min

    # Scale to target height
    # Measure current height as head Z - foot Z (after grounding)
    head_z = result[:, 10, 2].mean()
    foot_z = min(result[:, 3, 2].mean(), result[:, 6, 2].mean())
    current_height = head_z - foot_z

    if current_height > 0.1:  # Sanity check
        scale = target_height / current_height
        result *= scale

        # Re-ground feet after scaling
        foot_z_min = min(result[:, 3, 2].min(), result[:, 6, 2].min())
        result[:, :, 2] -= foot_z_min

    return result


def create_armature(name="PoseSkeleton"):
    """Create armature with H36M bone structure."""
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature_obj = bpy.context.active_object
    armature_obj.name = name
    armature = armature_obj.data
    armature.name = name + "_Data"

    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Create bones
    bones_created = {}

    for parent_idx, child_idx in H36M_BONES:
        bone_name = f"{H36M_JOINTS[parent_idx]}_{H36M_JOINTS[child_idx]}"

        bone = armature.edit_bones.new(bone_name)
        bone.head = Vector((0, 0, 0))
        bone.tail = Vector((0, 0.1, 0))  # Temporary, will be set in animation

        bones_created[bone_name] = (parent_idx, child_idx)

    bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj, bones_created


def create_mesh_skeleton(frame0):
    """Create visual mesh skeleton (spheres for joints, cylinders for bones)."""
    # Materials
    joint_mat = bpy.data.materials.new("JointMaterial")
    joint_mat.diffuse_color = (0.9, 0.2, 0.2, 1.0)

    bone_mat = bpy.data.materials.new("BoneMaterial")
    bone_mat.diffuse_color = (0.2, 0.5, 0.9, 1.0)

    joints = []
    bones = []

    # Create joint spheres
    for i, name in enumerate(H36M_JOINTS):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.025 * 1.75,  # Scale with target height
            location=Vector(frame0[i]),
            segments=12,
            ring_count=8
        )
        sphere = bpy.context.active_object
        sphere.name = f"J_{name}"
        sphere.data.materials.append(joint_mat)
        joints.append(sphere)

    # Create bone cylinders
    for parent_idx, child_idx in H36M_BONES:
        parent_pos = Vector(frame0[parent_idx])
        child_pos = Vector(frame0[child_idx])

        direction = child_pos - parent_pos
        length = direction.length

        if length < 0.001:
            bones.append((None, parent_idx, child_idx, 0))
            continue

        midpoint = (parent_pos + child_pos) / 2

        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.012 * 1.75,  # Scale with target height
            depth=length,
            location=midpoint,
            vertices=8
        )

        cylinder = bpy.context.active_object
        cylinder.name = f"B_{H36M_JOINTS[parent_idx]}_{H36M_JOINTS[child_idx]}"

        # Orient cylinder along bone direction
        up = Vector((0, 0, 1))
        rotation = up.rotation_difference(direction.normalized())
        cylinder.rotation_mode = 'QUATERNION'
        cylinder.rotation_quaternion = rotation

        cylinder.data.materials.append(bone_mat)
        bones.append((cylinder, parent_idx, child_idx, length))

    return joints, bones


def animate_skeleton(joints, bones, kp3d, fps=30, frame_step=1):
    """Animate the mesh skeleton."""
    n_frames = kp3d.shape[0]
    total_anim_frames = (n_frames + frame_step - 1) // frame_step

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_anim_frames
    bpy.context.scene.render.fps = int(fps)

    print(f"Animating {total_anim_frames} frames...")

    for frame_idx in range(0, n_frames, frame_step):
        anim_frame = frame_idx // frame_step + 1
        frame_data = kp3d[frame_idx]

        # Animate joints
        for i, joint_obj in enumerate(joints):
            joint_obj.location = Vector(frame_data[i])
            joint_obj.keyframe_insert(data_path="location", frame=anim_frame)

        # Animate bones
        for bone_obj, parent_idx, child_idx, orig_length in bones:
            if bone_obj is None:
                continue

            parent_pos = Vector(frame_data[parent_idx])
            child_pos = Vector(frame_data[child_idx])

            direction = child_pos - parent_pos
            length = direction.length

            if length < 0.001:
                continue

            # Update position
            midpoint = (parent_pos + child_pos) / 2
            bone_obj.location = midpoint
            bone_obj.keyframe_insert(data_path="location", frame=anim_frame)

            # Update scale (length)
            scale_factor = length / orig_length if orig_length > 0.001 else 1
            bone_obj.scale = (1, 1, scale_factor)
            bone_obj.keyframe_insert(data_path="scale", frame=anim_frame)

            # Update rotation
            up = Vector((0, 0, 1))
            rotation = up.rotation_difference(direction.normalized())
            bone_obj.rotation_quaternion = rotation
            bone_obj.keyframe_insert(data_path="rotation_quaternion", frame=anim_frame)

        if (frame_idx + 1) % 200 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")

    print("Animation complete!")


def setup_scene():
    """Set up camera, lighting, and ground plane."""
    # Camera facing the skeleton from the front
    bpy.ops.object.camera_add(location=(0, -5, 1.2))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.4, 0, 0)  # Point at skeleton
    bpy.context.scene.camera = camera

    # Sun light
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = bpy.data.materials.new("GroundMaterial")
    ground_mat.diffuse_color = (0.3, 0.3, 0.35, 1.0)
    ground.data.materials.append(ground_mat)

    # Render settings
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, 'BLENDER_EEVEE_NEXT') else 'BLENDER_EEVEE'


def main():
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Animate 3D skeleton and export FBX")
    parser.add_argument('--input', '-i', required=True, help='Input pickle file')
    parser.add_argument('--output', '-o', required=True, help='Output FBX file')
    parser.add_argument('--height', type=float, default=1.75, help='Target skeleton height in meters')
    parser.add_argument('--frame-step', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames to process (0=all)')
    parser.add_argument('--render-preview', action='store_true', help='Render preview images')

    args = parser.parse_args(argv)

    print("=" * 60)
    print("3D Skeleton Animation (Fixed MotionBERT Output)")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    kp3d = np.array(data['keypoints_3d'])
    fps = data.get('fps', 30.0)

    if args.max_frames > 0:
        kp3d = kp3d[:args.max_frames]

    print(f"Frames: {kp3d.shape[0]}, FPS: {fps}")

    # Transform coordinates
    print(f"\nTransforming to Blender coordinates (target height: {args.height}m)...")
    kp3d_blender = transform_to_blender(kp3d, target_height=args.height)

    # Debug: print first frame info
    frame0 = kp3d_blender[0]
    print(f"\nFrame 0 after transform:")
    print(f"  Hip: ({frame0[0, 0]:.3f}, {frame0[0, 1]:.3f}, {frame0[0, 2]:.3f})")
    print(f"  Head: ({frame0[10, 0]:.3f}, {frame0[10, 1]:.3f}, {frame0[10, 2]:.3f})")
    print(f"  RFoot: ({frame0[3, 0]:.3f}, {frame0[3, 1]:.3f}, {frame0[3, 2]:.3f})")
    print(f"  LFoot: ({frame0[6, 0]:.3f}, {frame0[6, 1]:.3f}, {frame0[6, 2]:.3f})")

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create skeleton
    print("\nCreating skeleton...")
    joints, bones = create_mesh_skeleton(kp3d_blender[0])

    # Animate
    print("\nAnimating...")
    animate_skeleton(joints, bones, kp3d_blender, fps=fps, frame_step=args.frame_step)

    # Setup scene
    setup_scene()

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # Save .blend file
    blend_path = args.output.replace('.fbx', '.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"\nSaved: {blend_path}")

    # Render preview if requested
    if args.render_preview:
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        for frame_num in [1, 50, 100]:
            if frame_num <= bpy.context.scene.frame_end:
                bpy.context.scene.frame_set(frame_num)
                preview_path = os.path.join(output_dir, f"{base_name}_frame{frame_num:04d}.png")
                bpy.context.scene.render.filepath = preview_path
                bpy.ops.render.render(write_still=True)
                print(f"Rendered: {preview_path}")

    # Export FBX with full animation
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,  # No simplification - keep all keyframes
        add_leaf_bones=False,
        object_types={'MESH', 'EMPTY', 'OTHER'},
    )
    print(f"Exported: {args.output}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
