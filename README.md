# EasyErgoBert - 3D Pose Estimation Pipeline

A complete pipeline for extracting 3D human pose from video using **YOLOv8-Pose** for 2D detection and **MotionBERT** for 2D-to-3D lifting. Exports to FBX for use in Blender, Unity, Unreal, and other 3D applications.

## Features

- **Fast 2D Detection**: YOLOv8-Pose running at ~60 FPS on GPU
- **Accurate 3D Lifting**: MotionBERT with proper pose3d checkpoint
- **GPU Accelerated**: Full RTX 5090/4090/3090 support via PyTorch 2.9+ and CUDA 12.8
- **FBX Export**: Animated skeleton export via Blender
- **Multiple Outputs**: JSON, Pickle, rendered video with skeleton overlay

## Quick Start

```bash
# 1. Activate environment
conda activate body4d

# 2. Process video
python process_video_fixed.py --input input/video.mp4 --output output/ --render

# 3. Export to FBX (requires Blender)
blender --background --python animate_3d_fixed.py -- \
    --input output/video_poses.pkl \
    --output output/animation.fbx
```

## Installation

### 1. Create Conda Environment

```bash
# Create environment with Python 3.11+ for modern GPU support
conda create -n body4d python=3.11 -y
conda activate body4d

# Install PyTorch with CUDA 12.8 (for RTX 40/50 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MotionBERT Checkpoint

**CRITICAL**: You must download the correct **pose3d** checkpoint, NOT the mesh checkpoint!

1. Go to: https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73
2. Navigate to the `pose3d` folder
3. Download `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` (~60 MB)
4. Place it in: `MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/`

```bash
# Verify checkpoint
python download_pose3d_checkpoint.py
```

### 3. Install Blender (for FBX export)

Download Blender 4.0+ from https://www.blender.org/download/

## Usage

### Process Video

```bash
# Basic usage (GPU)
python process_video_fixed.py --input video.mp4 --output output/

# With skeleton overlay video
python process_video_fixed.py --input video.mp4 --output output/ --render

# CPU mode (slower but works without CUDA)
python process_video_fixed.py --input video.mp4 --output output/ --device cpu

# Track specific person (if multiple people in video)
python process_video_fixed.py --input video.mp4 --output output/ --person 1
```

### Export to FBX

```bash
# Windows
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background \
    --python animate_3d_fixed.py -- \
    --input output/video_poses.pkl \
    --output output/animation.fbx

# Linux/Mac
blender --background --python animate_3d_fixed.py -- \
    --input output/video_poses.pkl \
    --output output/animation.fbx

# With preview renders
blender --background --python animate_3d_fixed.py -- \
    --input output/video_poses.pkl \
    --output output/animation.fbx \
    --render-preview

# Custom skeleton height (default 1.75m)
blender --background --python animate_3d_fixed.py -- \
    --input output/video_poses.pkl \
    --output output/animation.fbx \
    --height 1.80
```

### Export to Y-Bot (Mixamo Character)

For a more realistic animation with a humanoid character, use the Y-Bot export:

```bash
# Windows - Export with Y-Bot
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background \
    --python animate_ybot_fixed.py -- \
    --input output/video_poses.pkl \
    --ybot models/Y_Bot.fbx \
    --output output/ybot_animated.fbx

# Linux/Mac
blender --background --python animate_ybot_fixed.py -- \
    --input output/video_poses.pkl \
    --ybot models/Y_Bot.fbx \
    --output output/ybot_animated.fbx
```

**Note**: You need to download the Y-Bot FBX from [Mixamo](https://www.mixamo.com/) and place it in `models/Y_Bot.fbx`. The script maps H36M joints to Mixamo armature bones.

## Output Files

| File | Description |
|------|-------------|
| `*_poses.json` | 3D pose data in JSON format |
| `*_poses.pkl` | 3D pose data in Python pickle format |
| `*_skeleton.mp4` | Video with 2D skeleton overlay |
| `*.fbx` | Animated 3D skeleton for Blender/Unity/Unreal |
| `*.blend` | Blender project file |

## Project Structure

```
EasyErgoBert/
├── process_video_fixed.py    # Main pipeline script (USE THIS)
├── animate_3d_fixed.py       # Blender FBX export (skeleton)
├── animate_ybot_fixed.py     # Blender FBX export (Y-Bot character)
├── download_pose3d_checkpoint.py  # Checkpoint verification
├── requirements.txt          # Python dependencies
├── MotionBERT/              # MotionBERT (clone separately)
│   ├── checkpoint/
│   │   └── pose3d/          # PUT CHECKPOINT HERE
│   ├── configs/
│   └── lib/
├── utils/
│   ├── keypoint_converter.py # COCO to H36M conversion
│   ├── visualizer.py        # 3D visualization
│   └── export_obj.py        # OBJ export
├── docs/                    # Documentation
│   ├── SKELETON_FORMAT.md   # Skeleton joint definitions
│   ├── COORDINATE_SYSTEMS.md # Coordinate system details
│   └── SETUP_GUIDE.md       # Detailed setup instructions
├── input/                   # Input videos
└── output/                  # Output files
```

## Skeleton Format

This pipeline uses the **Human3.6M (H36M) 17-joint skeleton**. See `docs/SKELETON_FORMAT.md` for full details.

```
Joint Index  Name         Description
──────────────────────────────────────
0            Hip          Pelvis/Root
1            RHip         Right Hip
2            RKnee        Right Knee
3            RFoot        Right Ankle
4            LHip         Left Hip
5            LKnee        Left Knee
6            LFoot        Left Ankle
7            Spine        Lower Spine
8            Thorax       Upper Spine/Chest
9            Neck         Neck
10           Head         Head Top
11           LShoulder    Left Shoulder
12           LElbow       Left Elbow
13           LWrist       Left Wrist
14           RShoulder    Right Shoulder
15           RElbow       Right Elbow
16           RWrist       Right Wrist
```

## Performance

Tested on NVIDIA RTX 5090 Laptop GPU:

| Stage | Speed |
|-------|-------|
| 2D Detection (YOLOv8) | ~60 FPS |
| 3D Lifting (MotionBERT) | ~3300 FPS |
| Total Pipeline | ~56 FPS |

## Troubleshooting

### "CUDA not available"
- Ensure you have NVIDIA drivers installed
- Use PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- For older GPUs, use CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "Checkpoint not found"
- Download from OneDrive link above
- Make sure it's the **pose3d** checkpoint, not mesh
- Verify path: `MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`

### "Bone lengths look wrong"
- You're using the wrong checkpoint (mesh instead of pose3d)
- The mesh checkpoint outputs SMPL parameters, not joint positions

### "FBX has no animation"
- Use `animate_3d_fixed.py` (not older versions)
- Check that the pickle file has `keypoints_3d` data

## API Usage

```python
from process_video_fixed import PoseEstimationPipeline

# Initialize pipeline
pipeline = PoseEstimationPipeline(device='cuda:0')

# Process video
results = pipeline.process_video(
    video_path='input/video.mp4',
    output_dir='output/',
    person_idx=0,
    render_video=True
)

# Access 3D keypoints
import numpy as np
keypoints_3d = np.array(results['keypoints_3d'])  # Shape: (T, 17, 3)
```

## License

This project uses:
- [MotionBERT](https://github.com/Walter0807/MotionBERT) - MIT License
- [YOLOv8](https://github.com/ultralytics/ultralytics) - AGPL-3.0 License
- [Blender](https://www.blender.org/) - GPL License

## Acknowledgments

- MotionBERT: Zhu et al., "Learning Human Motion Representations: A Unified Perspective"
- YOLOv8: Ultralytics
