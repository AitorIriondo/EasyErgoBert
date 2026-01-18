"""
3D Skeleton Visualization Utilities

Provides tools for visualizing 3D pose estimation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
import pickle
import json
from pathlib import Path


# H36M skeleton connections
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # Right leg
    (0, 4), (4, 5), (5, 6),      # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13), # Left arm
    (8, 14), (14, 15), (15, 16)  # Right arm
]

H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Colors for different body parts
LIMB_COLORS = {
    'right_leg': 'blue',
    'left_leg': 'red',
    'spine': 'green',
    'left_arm': 'orange',
    'right_arm': 'purple'
}


def load_results(filepath: str) -> dict:
    """Load pose estimation results from JSON or pickle file."""
    filepath = Path(filepath)

    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Convert lists to numpy arrays
            for key in ['keypoints_2d_coco', 'keypoints_2d_h36m', 'keypoints_3d', 'scores']:
                if key in data and data[key] is not None:
                    data[key] = np.array(data[key])
            return data
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def plot_skeleton_3d(keypoints: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     title: str = "3D Skeleton",
                     elev: float = 15,
                     azim: float = 70) -> plt.Axes:
    """
    Plot a single 3D skeleton.

    Args:
        keypoints: Array of shape (17, 3)
        ax: Optional matplotlib 3D axis
        title: Plot title
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Define limb groups for coloring
    limb_groups = {
        'right_leg': [(0, 1), (1, 2), (2, 3)],
        'left_leg': [(0, 4), (4, 5), (5, 6)],
        'spine': [(0, 7), (7, 8), (8, 9), (9, 10)],
        'left_arm': [(8, 11), (11, 12), (12, 13)],
        'right_arm': [(8, 14), (14, 15), (15, 16)]
    }

    # Plot limbs
    for group_name, connections in limb_groups.items():
        color = LIMB_COLORS[group_name]
        for i, j in connections:
            ax.plot3D([keypoints[i, 0], keypoints[j, 0]],
                     [keypoints[i, 1], keypoints[j, 1]],
                     [keypoints[i, 2], keypoints[j, 2]],
                     color=color, linewidth=2)

    # Plot joints
    ax.scatter3D(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                 c='black', s=30, depthshade=True)

    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = np.array([
        keypoints[:, 0].max() - keypoints[:, 0].min(),
        keypoints[:, 1].max() - keypoints[:, 1].min(),
        keypoints[:, 2].max() - keypoints[:, 2].min()
    ]).max() / 2.0

    mid_x = (keypoints[:, 0].max() + keypoints[:, 0].min()) * 0.5
    mid_y = (keypoints[:, 1].max() + keypoints[:, 1].min()) * 0.5
    mid_z = (keypoints[:, 2].max() + keypoints[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def plot_skeleton_2d(keypoints: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     image: Optional[np.ndarray] = None,
                     title: str = "2D Skeleton") -> plt.Axes:
    """
    Plot a single 2D skeleton.

    Args:
        keypoints: Array of shape (17, 2)
        ax: Optional matplotlib axis
        image: Optional background image
        title: Plot title

    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if image is not None:
        ax.imshow(image)

    # Define limb groups for coloring
    limb_groups = {
        'right_leg': [(0, 1), (1, 2), (2, 3)],
        'left_leg': [(0, 4), (4, 5), (5, 6)],
        'spine': [(0, 7), (7, 8), (8, 9), (9, 10)],
        'left_arm': [(8, 11), (11, 12), (12, 13)],
        'right_arm': [(8, 14), (14, 15), (15, 16)]
    }

    # Plot limbs
    for group_name, connections in limb_groups.items():
        color = LIMB_COLORS[group_name]
        for i, j in connections:
            ax.plot([keypoints[i, 0], keypoints[j, 0]],
                   [keypoints[i, 1], keypoints[j, 1]],
                   color=color, linewidth=2)

    # Plot joints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c='black', s=30)

    ax.set_title(title)
    ax.set_aspect('equal')

    if image is None:
        ax.invert_yaxis()  # Flip y-axis for image coordinates

    return ax


def animate_skeleton_3d(keypoints_sequence: np.ndarray,
                        fps: float = 30,
                        save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create an animated 3D skeleton visualization.

    Args:
        keypoints_sequence: Array of shape (T, 17, 3)
        fps: Frames per second
        save_path: Optional path to save animation

    Returns:
        Matplotlib FuncAnimation object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate global bounds
    all_kpts = keypoints_sequence.reshape(-1, 3)
    max_range = np.array([
        all_kpts[:, 0].max() - all_kpts[:, 0].min(),
        all_kpts[:, 1].max() - all_kpts[:, 1].min(),
        all_kpts[:, 2].max() - all_kpts[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_kpts[:, 0].max() + all_kpts[:, 0].min()) * 0.5
    mid_y = (all_kpts[:, 1].max() + all_kpts[:, 1].min()) * 0.5
    mid_z = (all_kpts[:, 2].max() + all_kpts[:, 2].min()) * 0.5

    # Store line objects
    lines = []
    scatter = None

    def init():
        nonlocal lines, scatter
        ax.clear()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        lines = []
        for _ in H36M_SKELETON:
            line, = ax.plot3D([], [], [], linewidth=2)
            lines.append(line)

        scatter = ax.scatter3D([], [], [], c='black', s=30)
        return lines + [scatter]

    def update(frame):
        nonlocal scatter
        keypoints = keypoints_sequence[frame]

        # Update limbs
        limb_idx = 0
        for group_name, connections in [
            ('right_leg', [(0, 1), (1, 2), (2, 3)]),
            ('left_leg', [(0, 4), (4, 5), (5, 6)]),
            ('spine', [(0, 7), (7, 8), (8, 9), (9, 10)]),
            ('left_arm', [(8, 11), (11, 12), (12, 13)]),
            ('right_arm', [(8, 14), (14, 15), (15, 16)])
        ]:
            color = LIMB_COLORS[group_name]
            for i, j in connections:
                lines[limb_idx].set_data_3d(
                    [keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    [keypoints[i, 2], keypoints[j, 2]]
                )
                lines[limb_idx].set_color(color)
                limb_idx += 1

        # Update joints
        scatter._offsets3d = (keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])

        ax.set_title(f'Frame {frame}/{len(keypoints_sequence)}')
        return lines + [scatter]

    anim = FuncAnimation(fig, update, frames=len(keypoints_sequence),
                        init_func=init, blit=False, interval=1000/fps)

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to: {save_path}")

    return anim


def plot_multi_view(keypoints: np.ndarray, title: str = "Multi-View 3D Skeleton"):
    """
    Plot 3D skeleton from multiple viewpoints.

    Args:
        keypoints: Array of shape (17, 3)
        title: Overall title
    """
    fig = plt.figure(figsize=(15, 5))

    views = [
        (15, 70, "Front-Right"),
        (15, 160, "Front-Left"),
        (90, 90, "Top-Down"),
        (0, 0, "Side")
    ]

    for i, (elev, azim, view_title) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        plot_skeleton_3d(keypoints, ax=ax, title=view_title, elev=elev, azim=azim)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def compare_2d_3d(keypoints_2d: np.ndarray,
                  keypoints_3d: np.ndarray,
                  image: Optional[np.ndarray] = None):
    """
    Side-by-side comparison of 2D and 3D poses.

    Args:
        keypoints_2d: Array of shape (17, 2)
        keypoints_3d: Array of shape (17, 3)
        image: Optional background image for 2D plot
    """
    fig = plt.figure(figsize=(15, 6))

    # 2D plot
    ax1 = fig.add_subplot(1, 2, 1)
    plot_skeleton_2d(keypoints_2d, ax=ax1, image=image, title="2D Skeleton (H36M)")

    # 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_skeleton_3d(keypoints_3d, ax=ax2, title="3D Skeleton (MotionBERT)")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Load and visualize results file
        filepath = sys.argv[1]
        data = load_results(filepath)

        if data['keypoints_3d'] is not None:
            keypoints_3d = np.array(data['keypoints_3d'])

            # Show first frame
            print(f"Loaded {len(keypoints_3d)} frames")
            print("Showing first frame (close window to continue)...")

            fig = plot_multi_view(keypoints_3d[0], title="Frame 0")
            plt.show()

            # Ask about animation
            response = input("Create animation? (y/n): ")
            if response.lower() == 'y':
                print("Creating animation (this may take a while)...")
                anim = animate_skeleton_3d(keypoints_3d, fps=data.get('fps', 30))
                plt.show()
        else:
            print("No 3D keypoints found in file")
    else:
        # Demo with random data
        print("Usage: python visualizer.py <results_file.pkl>")
        print("\nShowing demo with random data...")

        demo_kpts = np.random.randn(17, 3) * 0.5
        demo_kpts[:, 2] += 2  # Move away from origin

        fig = plot_multi_view(demo_kpts, title="Demo 3D Skeleton")
        plt.show()
