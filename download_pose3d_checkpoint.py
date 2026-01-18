"""
Download the correct MotionBERT pose3d checkpoint.

The pose3d checkpoint (FT_MB_lite_MB_ft_h36m_global_lite) is required for
proper 3D pose estimation. The mesh checkpoint that's currently installed
is NOT suitable for pose estimation.

OneDrive link: https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73

=== WHAT TO DOWNLOAD ===
You need: FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin

=== ALTERNATIVE CHECKPOINTS (if available) ===
Full model (larger, potentially more accurate):
- FT_MB_release_MB_ft_h36m_global/best_epoch.bin
  Config: configs/pose3d/MB_ft_h36m_global.yaml
  Model params: dim_feat=512, mlp_ratio=2

Lite model (current, smaller, faster):
- FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
  Config: configs/pose3d/MB_ft_h36m_global_lite.yaml
  Model params: dim_feat=256, mlp_ratio=4

Both produce 3D poses in GLOBAL coordinates (not root-relative).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Checkpoint paths
CHECKPOINT_DIR_LITE = PROJECT_ROOT / "MotionBERT" / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite"
CHECKPOINT_DIR_FULL = PROJECT_ROOT / "MotionBERT" / "checkpoint" / "pose3d" / "FT_MB_release_MB_ft_h36m_global"

CHECKPOINT_FILE_LITE = CHECKPOINT_DIR_LITE / "best_epoch.bin"
CHECKPOINT_FILE_FULL = CHECKPOINT_DIR_FULL / "best_epoch.bin"

# Wrong checkpoint (mesh model)
MESH_CHECKPOINT = PROJECT_ROOT / "MotionBERT" / "checkpoint" / "mesh" / "FT_MB_release_MB_ft_pw3d" / "best_epoch.bin"

def check_checkpoints():
    """Check what checkpoints exist."""
    print("=" * 60)
    print("MotionBERT Checkpoint Status")
    print("=" * 60)

    # Check lite model
    if CHECKPOINT_FILE_LITE.exists():
        size_mb = CHECKPOINT_FILE_LITE.stat().st_size / (1024 * 1024)
        print(f"[OK] Pose3D Lite: {CHECKPOINT_FILE_LITE.name} ({size_mb:.1f} MB)")
        lite_ok = True
    else:
        print(f"[MISSING] Pose3D Lite: {CHECKPOINT_FILE_LITE}")
        lite_ok = False

    # Check full model
    if CHECKPOINT_FILE_FULL.exists():
        size_mb = CHECKPOINT_FILE_FULL.stat().st_size / (1024 * 1024)
        print(f"[OK] Pose3D Full: {CHECKPOINT_FILE_FULL.name} ({size_mb:.1f} MB)")
        full_ok = True
    else:
        print(f"[MISSING] Pose3D Full: {CHECKPOINT_FILE_FULL}")
        full_ok = False

    # Check mesh model (should NOT be used for pose)
    if MESH_CHECKPOINT.exists():
        size_mb = MESH_CHECKPOINT.stat().st_size / (1024 * 1024)
        print(f"[WARNING] Mesh model: {MESH_CHECKPOINT.name} ({size_mb:.1f} MB)")
        print("          ^ This is for body mesh reconstruction, NOT 3D pose!")

    print("=" * 60)
    return lite_ok or full_ok

def print_download_instructions():
    """Print download instructions."""
    print("""
DOWNLOAD INSTRUCTIONS
=====================

1. Open this OneDrive link in your browser:
   https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73

2. Navigate to the pose3d folder and find:
   - FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin (recommended, ~50MB)
   OR
   - FT_MB_release_MB_ft_h36m_global/best_epoch.bin (full model, ~100MB)

3. Download best_epoch.bin and place it in:
""")
    print(f"   {CHECKPOINT_DIR_LITE}")
    print("""
After downloading, run this script again to verify the checkpoint.

TESTING
=======
Once the checkpoint is downloaded, test with:

  conda activate body4d
  python process_video_fixed.py --input input/aitor_garden_walk.mp4 --output output/ --render

This will use GPU acceleration on your RTX 5090!
""")

def main():
    print("MotionBERT Pose3D Checkpoint Downloader")
    print()

    # Create directories
    CHECKPOINT_DIR_LITE.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR_FULL.mkdir(parents=True, exist_ok=True)

    if check_checkpoints():
        print("\nAt least one pose3d checkpoint is available!")
        print("You can now run: python process_video_fixed.py --input <video> --output <dir>")
        return True

    print_download_instructions()
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
