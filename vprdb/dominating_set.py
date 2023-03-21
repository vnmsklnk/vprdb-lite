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
import networkx as nx

from vprdb.core import Database, VoxelGrid


def dominating_set(db: Database, voxel_size: float, threshold: float) -> Database:
    """
    Creates reduced DB using dominating set algo
    :param db: Database to reduce
    :param voxel_size: Voxel size used to down sample point clouds
    :param threshold: The value indicating which IoU value will be enough
    to consider the point clouds as overlapping
    :return: New reduced database
    """
    min_bounds, max_bounds = db.bounds
    voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)
    # Determine which frames cover a particular voxel
    voxel_to_frames_dict = dict()
    # Frame size is the number of voxels it covers
    frames_sizes = []
    for i, pose in enumerate(db.trajectory):
        pcd = db.get_pcd_by_index(i).transform(pose)
        down_sampled_pcd = voxel_grid.voxel_down_sample(pcd)
        points = down_sampled_pcd.points
        frames_sizes.append(len(points))
        for point in points:
            voxel_index = voxel_grid.get_voxel_index(point)
            if voxel_index in voxel_to_frames_dict:
                voxel_to_frames_dict[voxel_index].append(i)
            else:
                voxel_to_frames_dict[voxel_index] = [i]

    intersections = dict()
    for covering_frames in voxel_to_frames_dict.values():
        for i, frame_1 in enumerate(covering_frames):
            for frame_2 in covering_frames[i + 1:]:
                intersection = tuple(sorted((frame_1, frame_2)))
                if intersection in intersections:
                    intersections[intersection] += 1
                else:
                    intersections[intersection] = 1

    IoUs = dict.fromkeys(intersections.keys())
    for (frame_1, frame_2), intersection in intersections.items():
        IoUs[(frame_1, frame_2)] = intersection / (
            frames_sizes[frame_1] + frames_sizes[frame_2] - intersection
        )

    G = nx.Graph()
    G.add_nodes_from(range(len(db)))
    for (fr1, fr2), IoU in IoUs.items():
        if IoU > threshold:
            G.add_edge(fr1, fr2)
    result_indices = list(nx.dominating_set(G))
    result_indices.sort()
    new_rgb = [db.rgb_images[i] for i in result_indices]
    new_depth = [db.depth_images[i] for i in result_indices]
    new_traj = [db.trajectory[i] for i in result_indices]
    return Database(new_rgb, new_depth, db.depth_scale, db.intrinsics, new_traj)
