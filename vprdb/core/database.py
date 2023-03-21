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
import cv2
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from functools import cached_property
from nptyping import Float, NDArray, Shape
from pathlib import Path


@dataclass(frozen=True)
class Database:
    rgb_images: list[Path]
    depth_images: list[Path]
    depth_scale: int
    intrinsics: NDArray[Shape["3, 3"], Float]
    trajectory: list[NDArray[Shape["4, 4"], Float]]

    def __post_init__(self):
        if not (len(self.trajectory) == len(self.rgb_images) == len(self.depth_images)):
            raise ValueError(
                "Trajectory, RGB images and depth images should have equal length"
            )

    def __len__(self):
        return len(self.trajectory)

    def get_pcd_by_index(self, n: int):
        """
        Converts a depth image into a point cloud
        :param n: Index of depth image to convert
        :return: Open3D point cloud
        """
        depth_image = cv2.imread(str(self.depth_images[n]), cv2.IMREAD_ANYDEPTH)
        height, width = depth_image.shape
        depth_image = o3d.geometry.Image(depth_image)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, self.intrinsics)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsics,
            depth_scale=self.depth_scale,
            depth_trunc=float("inf"),
        )
        return pcd

    @cached_property
    def bounds(
        self,
    ) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
        """
        Gets bounds of the DB scene
        :return: Min and max bounds of the scene
        """
        min_bounds = []
        max_bounds = []

        for i in range(len(self)):
            pcd = self.get_pcd_by_index(i).transform(self.trajectory[i])
            min_bounds.append(pcd.get_min_bound())
            max_bounds.append(pcd.get_max_bound())

        min_bound = np.amin(np.asarray(min_bounds), axis=0)
        max_bound = np.amax(np.asarray(max_bounds), axis=0)
        return min_bound, max_bound
