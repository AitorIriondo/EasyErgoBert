# 3D Pose Estimation Project - Session Notes
**Date:** January 16, 2025
**Status:** In Progress

## Project Overview
Building a 3D pose estimation pipeline using:
- **RTMPose/YOLOv8-Pose** for 2D keypoint detection
- **MotionBERT** for 2D→3D lifting
- Target: Process videos and export 3D skeleton data

## Hardware
- **GPU:** NVIDIA RTX 5090 Laptop (Blackwell architecture, sm_120)
- **Issue:** PyTorch doesn't fully support RTX 5090 yet (needs CUDA 12.8+ and PyTorch nightly with Python 3.11+)
- **Workaround:** Using CPU for now, or upgrade to Python 3.11 for GPU support

## Environment
- **Conda environment:** `pose_estimation`
- **Python:** 3.8 (too old for RTX 5090 GPU support)
- **Location:** `C:\VideoPoseEstimation`

## Files Created

### Setup Scripts
- `env_setup.bat` / `env_setup.ps1` / `env_setup.sh` - Environment setup
- `download_models.bat` / `download_models.ps1` / `download_models.sh` - Model downloads
- `test_installation.py` - Verify installation

### Main Scripts
- `process_video.py` - Original pipeline (has MMPose/MMCV version conflicts)
- `process_video_simple.py` - **USE THIS** - Simplified pipeline using YOLOv8-Pose

### Utilities
- `utils/keypoint_converter.py` - COCO (17 joints) → H36M (17 joints) conversion
- `utils/visualizer.py` - 3D skeleton visualization with matplotlib
- `utils/export_obj.py` - Export 3D skeleton as OBJ files

### Models Downloaded
Located in `C:\VideoPoseEstimation\models/`:
- `rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth` - Person detector
- `rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth` - 2D pose
- `best_epoch.bin` - MotionBERT checkpoint (user downloaded)

### Test Video
- `input/aitor_garden_walk.mp4` (~78MB, 1136 frames, 1920x1080, 30fps)

## Issues Encountered & Solutions

### 1. MMPose/MMCV Version Conflicts
**Problem:** MMCV 2.2.0 incompatible with MMDet, complex dependency chain
**Solution:** Created `process_video_simple.py` using YOLOv8-Pose instead

### 2. RTX 5090 Not Supported
**Problem:** PyTorch 2.0.1/2.4.1 don't support sm_120 (Blackwell)
**Solution:** Use `--device cpu` or upgrade to Python 3.11 + PyTorch nightly

### 3. No Skeleton Rendering (All Zeros)
**Problem:** rtmlib failed silently, YOLO fallback not triggered
**Solution:** Rewrote detector to use YOLOv8-Pose directly

### 4. OBJ Export Was Broken (Fixed Jan 17)
**Problem:** OBJ files were a mess of lines with skeleton wrongly rigged
**Root Cause:**
- MotionBERT outputs in a normalized camera-space coordinate system
- The vertical axis was not aligned with Blender's Z-up convention
- Joint/bone radii were fixed values that didn't scale with the skeleton
**Solution:**
- Added automatic detection of the skeleton's up direction (feet→head vector)
- Rotate skeleton to align with Blender's Z-up coordinate system
- Scale skeleton to realistic height (default 1.7m)
- Compute joint/bone radii proportionally (2.5% and 1.5% of height)
- Added `coord_mode` parameter for different coordinate systems

## Current Pipeline Status

### Working:
- [x] Environment setup
- [x] Model downloads
- [x] YOLOv8-Pose detection
- [x] COCO→H36M conversion
- [x] MotionBERT 3D lifting (on CPU)
- [x] JSON/Pickle output
- [x] Video rendering with skeleton overlay
- [x] OBJ export for 3D skeleton (FIXED - Jan 17)

### Not Yet Implemented:
- [ ] BVH export
- [ ] Blender rig animation

## Commands to Run

### Activate Environment
```batch
conda activate pose_estimation
cd C:\VideoPoseEstimation
```

### Install/Update Dependencies
```batch
pip install ultralytics  # For YOLOv8-Pose
```

### Process Video (CPU - works with RTX 5090)
```batch
python process_video_simple.py --input input/aitor_garden_walk.mp4 --output output/ --render --device cpu
```

### Process Video with OBJ Export
```batch
python process_video_simple.py --input input/aitor_garden_walk.mp4 --output output/ --render --export-obj --device cpu
```

### Export OBJ from Existing Results
```batch
# Export all frames at 1.7m height (default)
python -m utils.export_obj -i output/aitor_garden_walk_poses.pkl -o output/obj_files

# Export every 5th frame at 1.8m height
python -m utils.export_obj -i output/aitor_garden_walk_poses.pkl -o output/obj_files -e 5 --height 1.8

# Export without coordinate conversion (raw MotionBERT output)
python -m utils.export_obj -i output/aitor_garden_walk_poses.pkl -o output/obj_files --no-convert
```

## For GPU Support (Future)
Upgrade to Python 3.11 for RTX 5090:
```batch
conda create -n pose_py311 python=3.11 -y
conda activate pose_py311
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics opencv-python numpy scipy matplotlib tqdm einops timm pyyaml
```

## Next Steps

1. ~~**Test the fixed pipeline** - Run with YOLOv8-Pose on CPU~~ DONE
2. ~~**Verify skeleton rendering** - Check that keypoints are detected~~ DONE
3. ~~**Test OBJ export** - Verify OBJ files open in Blender~~ FIXED

4. **Add BVH export** - Standard motion capture format
5. **Add Blender script** - Animate Rigify rig with motion data
6. **Add FBX export** - For Unity/Unreal import

## H36M Joint Mapping (17 joints)
```
Index  Joint Name    Description
─────────────────────────────────
0      Hip          Pelvis/Root
1      RHip         Right Hip
2      RKnee        Right Knee
3      RFoot        Right Ankle
4      LHip         Left Hip
5      LKnee        Left Knee
6      LFoot        Left Ankle
7      Spine        Lower Spine
8      Thorax       Upper Spine/Chest
9      Neck/Nose    Neck
10     Head         Head Top
11     LShoulder    Left Shoulder
12     LElbow       Left Elbow
13     LWrist       Left Wrist
14     RShoulder    Right Shoulder
15     RElbow       Right Elbow
16     RWrist       Right Wrist
```

## Skeleton Connections
```
Right Leg:  Hip→RHip→RKnee→RFoot
Left Leg:   Hip→LHip→LKnee→LFoot
Spine:      Hip→Spine→Thorax→Neck→Head
Left Arm:   Thorax→LShoulder→LElbow→LWrist
Right Arm:  Thorax→RShoulder→RElbow→RWrist
```

## FBX Animation Export (Added Jan 18)

### Issue Discovered: MotionBERT 3D Output Corrupted
During FBX export development, we discovered the MotionBERT 3D lifting output has issues:
- **2D keypoints are correct** - The YOLOv8-Pose detection works perfectly
- **3D keypoints are corrupted** - Bone lengths are wrong (e.g., Hip→Spine = 3.27m instead of 0.15m)
- The joints don't form a coherent skeleton in 3D space

This suggests an issue with the MotionBERT model/checkpoint or preprocessing. **TODO: Investigate MotionBERT pipeline.**

### Working Solution: 2D-to-3D Animation
Created `animate_2d_to_3d.py` which:
- Uses the correct 2D keypoints directly
- Converts to 3D using simple depth estimation
- Creates proper human skeleton with correct proportions
- Exports animated FBX

### Export Commands
```batch
# Export animated skeleton from pose data
blender --background --python animate_2d_to_3d.py -- --input output/aitor_garden_walk_poses.pkl --output output/animation.fbx

# With preview renders
blender --background --python animate_2d_to_3d.py -- --input output/aitor_garden_walk_poses.pkl --output output/animation.fbx --render
```

### Output Files Created
- `output/aitor_walk_animated.fbx` (35 MB) - Full 1136-frame animation
- `output/aitor_walk_animated.blend` (2.5 MB) - Blender project file

### Blender Scripts Created
- `animate_2d_to_3d.py` - Main working script (converts 2D to 3D, animates, exports FBX)
- `animate_final.py` - Uses MotionBERT 3D data (produces corrupted output due to bad data)
- `debug_bones.py` - Debug script to check bone lengths
- `check_2d_keypoints.py` - Verify 2D keypoint quality

### Y-Bot FBX Files
- `y_bot.fbx` - Has 65-bone Mixamo armature (good for future retargeting)
- `Y_Bot_Base.fbx` - No armature, just mesh segments

## MotionBERT Fix (Jan 18) - ROOT CAUSE FOUND

### Issue Discovered
The MotionBERT 3D output was corrupted because **wrong checkpoint was being used**:
- **Installed checkpoint**: `mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin` (MESH model for body reconstruction)
- **Required checkpoint**: `pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` (POSE3D model for 3D lifting)

The mesh model outputs SMPL body mesh parameters, not 3D joint positions!

### Solution Created
1. **`process_video_fixed.py`** - Fixed pipeline with correct MotionBERT integration:
   - Uses correct pose3d config (`MB_ft_h36m_global_lite.yaml`)
   - Loads correct checkpoint (`pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`)
   - Uses `dim_in=3` (x, y, confidence) - not dim_in=2
   - Uses `checkpoint['model_pos']` key - not `checkpoint['model']`
   - Adds flip test-time augmentation for accuracy
   - Validates bone lengths after lifting

2. **`download_pose3d_checkpoint.py`** - Helper to check/download checkpoints

### Download Required
Download pose3d checkpoint from OneDrive:
```
https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73
```

Look for: `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`
Place in: `MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/`

### GPU Environment
- Use `body4d` conda environment for GPU acceleration (RTX 5090)
- Has PyTorch 2.9.1+cu128 with CUDA 12.8 support

### Commands to Run (After Downloading Checkpoint)
```batch
conda activate body4d
python download_pose3d_checkpoint.py  # Verify checkpoint
python process_video_fixed.py --input input/aitor_garden_walk.mp4 --output output_fixed/ --render
```

### FBX Export (Working!)
```batch
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python animate_3d_fixed.py -- --input output_fixed/aitor_garden_walk_poses.pkl --output output_fixed/aitor_walk_3d.fbx --render-preview
```

### Verified Results (Jan 18)
- **Pipeline processing**: 56.57 FPS (1136 frames in 20 seconds)
- **3D Lifting**: 0.34 seconds (3341 FPS!)
- **Bone lengths now correct**:
  - Hip-Spine: 0.129m (was 3.27m with wrong checkpoint)
  - Thigh: 0.235m, Lower leg: 0.221m
  - Upper arm: 0.156m, Forearm: 0.131m
- **Output files**:
  - `output_fixed/aitor_walk_3d.fbx` (167 KB)
  - `output_fixed/aitor_walk_3d.blend` (3.4 MB)
  - Preview renders showing correct humanoid skeleton

## User Request for Future
- Animate Blender rig (Rigify or similar) with motion capture data
- Export as FBX for use in game engines/animation software
- Options: BVH export, Blender Python script, or direct FBX
