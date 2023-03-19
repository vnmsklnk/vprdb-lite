import mrob
import numpy as np

from nptyping import Float, NDArray, Shape
from pathlib import Path

from vprdb.core import Database


def __load_poses(poses_path: Path):
    poses_quat = []
    with open(poses_path, "r") as file:
        for line in file:
            poses_quat.append([float(i) for i in line.strip().split(" ")])

    poses = []
    for pose in poses_quat:
        t = pose[1:4]
        R = mrob.geometry.quat_to_so3(pose[4:8])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses.append(T)

    return poses


def read_dataset(
    path_to_dataset: Path,
    color_dir: str,
    depth_dir: str,
    trajectory_file_name: str,
    intrinsics: NDArray[Shape["3, 3"], Float],
    depth_scale: int,
) -> Database:
    """
    Reads dataset from given directory
    :param path_to_dataset: The name of the directory to read
    :param color_dir: The name of the directory with color images
    :param depth_dir: The name of the directory with depth images
    :param trajectory_file_name: The name of file with trajectory
    :param intrinsics: NumPy array with camera intrinsics
    :param depth_scale: Depth scale
    """
    path_to_rgb = path_to_dataset / color_dir
    path_to_depth = path_to_dataset / depth_dir

    rgb_images = sorted(list(path_to_rgb.iterdir()))
    depth_images = sorted(list(path_to_depth.iterdir()))

    traj = __load_poses(path_to_dataset / trajectory_file_name)

    return Database(rgb_images, depth_images, depth_scale, intrinsics, traj)
