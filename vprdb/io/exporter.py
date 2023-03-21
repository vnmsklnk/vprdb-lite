#  Copyright (c) 2023, Skoltech
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import mrob
import numpy as np
import shutil

from pathlib import Path

from vprdb.core import Database


def __poses_to_txt(poses, filename_to_write):
    with open(filename_to_write, "w") as trajectory_file:
        for pose in poses:
            R = pose[:3, :3]
            t = pose[:3, 3]
            quat = mrob.geometry.so3_to_quat(R)
            pose_string = " ".join(np.concatenate((t, quat)).astype(str))
            trajectory_file.write(f"{pose_string}\n")


def export(
    database: Database,
    path_to_export: Path,
    color_dir: str,
    depth_dir: str,
    trajectory_file_name: str,
    intrinsics_file_name: str,
):
    """
    Exports Database to hard drive
    :param database: Database for exporting
    :param path_to_export: Directory for exporting. Will be created if it does not exist
    :param color_dir: Directory name for saving color images
    :param depth_dir: Directory name for saving depth images
    :param trajectory_file_name: File name for saving the trajectory
    :param intrinsics_file_name: File name for saving camera intrinsics
    """
    path_to_color = path_to_export / color_dir
    path_to_depth = path_to_export / depth_dir
    path_to_color.mkdir(parents=True, exist_ok=False)
    path_to_depth.mkdir(exist_ok=False)

    for rgb_image in database.rgb_images:
        shutil.copyfile(rgb_image, path_to_color / rgb_image.name)

    for depth_image in database.depth_images:
        shutil.copyfile(depth_image, path_to_depth / depth_image.name)

    __poses_to_txt(database.trajectory, path_to_export / trajectory_file_name)

    np.savetxt(path_to_export / intrinsics_file_name, database.intrinsics)
