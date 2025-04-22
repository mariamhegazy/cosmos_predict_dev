# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/autoregressive/datasets/video_dataset.py
"""

import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_predict1.autoregressive.configs.base.dataset import VideoDatasetConfig, H5DatasetConfig
from cosmos_predict1.autoregressive.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
import h5py


class H5Dataset(Dataset):
    def __init__(self, config: H5DatasetConfig):
        """Video Dataset class for loading video-to-video generation data.
        """

        super().__init__()
        self.dataset_dir = config.dataset_dir
        self.sequence_interval = config.sequence_interval
        self.sequence_length = config.num_frames
        self.video_size = config.video_size
        self.start_frame_interval = config.start_frame_interval

        

        self.video_dir = self.dataset_dir
        # self.video_paths = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if f.endswith(".mp4")]
        self.video_paths = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if f.endswith(".h5")]
        
        print(f"{len(self.video_paths)} videos in total")


        # self.mapping = self._build_index_mapping()
        # self.total_samples = len(self.mapping)

        self.samples_per_clip = (
            1 + self.sequence_length // self.start_frame_interval
            if self.sequence_length != self.start_frame_interval
            else 1
        )
        self.total_samples = (
            (len(self.video_paths) * self.samples_per_clip) - (self.samples_per_clip - 1)
            if self.sequence_length != self.start_frame_interval
            else len(self.video_paths)
        )
        

        # self.samples = self._init_samples(self.video_paths)
        # self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        # print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            input_keys=["video"],
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            input_keys=["video"],
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            input_keys=["video"],
            args={"mean": 0.5, "std": 0.5},
        )
    

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"
    
    def _build_mapping_for_video(self, video_path, sequence_length, sequence_interval, start_frame_interval):
        mapping_video = []
        with h5py.File(video_path, "r") as f:
            total_frames = f["video"].shape[0]
        max_start = total_frames - sequence_length * sequence_interval
        for start_frame in range(0, max_start + 1, start_frame_interval):
            frame_ids = [start_frame + i * sequence_interval for i in range(sequence_length)]
            if frame_ids[-1] < total_frames:
                mapping_video.append({
                    "video_path": video_path,
                    "start_frame": start_frame,
                    "frame_ids": frame_ids,
                })
        return mapping_video

    
    def _build_index_mapping(self):
        mapping = []
        # Use a thread pool to process videos concurrently
        with ThreadPoolExecutor(max_workers=200) as executor:
            futures = {
                executor.submit(
                    self._build_mapping_for_video,
                    video_path,
                    self.sequence_length,
                    self.sequence_interval,
                    self.start_frame_interval,
                ): video_path
                for video_path in self.video_paths
            }
            for future in as_completed(futures):
                mapping.extend(future.result())
        print(f"Built mapping with {len(mapping)} total samples.")
        return mapping

    

    def __len__(self):
        return self.total_samples

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all(), "Some frame_ids are out of range."
        assert (np.array(frame_ids) >= 0).all(), "Some frame_ids are negative."
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        fps = vr.get_avg_fps()
        return frame_data, fps

    def _get_frames(self, video_path, frame_ids):
        frames, fps = self._load_video(video_path, frame_ids)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)  # Rearrange from [T, H, W, C] to [T, C, H, W]
        return frames, fps

    def __getitem__(self, index):
        try:
            # sample_info = self.mapping[index]
            # video_path = sample_info["video_path"]
            # frame_ids = sample_info["frame_ids"]
            video_index = index // self.samples_per_clip
            offset_idx = index % self.samples_per_clip
            video_path = self.video_paths[video_index]
            with h5py.File(video_path, "r") as f:
                total_frames = f["video"].shape[0]
                start_frame = offset_idx
                frame_ids = [start_frame + i * self.sequence_interval for i in range(self.sequence_length)]
                selected_frames = f["video"][frame_ids]
           
            assert total_frames >= self.sequence_length * self.sequence_interval, "Not enough frames"
            # selected_frames = frames[frame_ids]
            # Choose a valid start index randomly
            # max_start = total_frames - self.sequence_length * self.sequence_interval
            # start_frame = np.random.randint(0, max_start + 1)
            # start_frame = offset_idx
            # frame_ids = [start_frame + i * self.sequence_interval for i in range(self.sequence_length)]
            # selected_frames = frames[frame_ids]

            frames = torch.from_numpy(selected_frames)
            if frames.ndim == 4 and frames.shape[-1] == 3:  # [T, H, W, C] → [T, C, H, W]
                frames = frames.permute(0, 3, 1, 2)

            data = dict()

            # video, fps = self._get_frames(video_path, frame_ids)
            data["video"] = frames
            data["fps"] = 10 #TO-DO: change this to dynamically get the correct fps for different datasets.
            data["num_frames"] = self.sequence_length
            data["orig_num_frames"] = total_frames
            data["chunk_index"] = -1
            data["frame_start"] = frame_ids[0]
            data["frame_end"] = frame_ids[-1]

            data["video_name"] = {
                "video_path": video_path,
                "start_frame_id": str(frame_ids[0]),
            }

            # resize video to smallest side aspect preserving
            data = self.resize_transform(data)
            # center crop video
            data = self.crop_transform(data)
            # normalize video
            data = self.normalize_transform(data)

            data["video"] = data["video"].permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # print(f"Loaded {video_path} with {len(frame_ids)} frames, and shape {data['video'].shape}")

            return data
        except Exception:
            
            warnings.warn(
                f"Invalid data encountered: {video_path}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(self.total_samples)]


if __name__ == "__main__":
    config = H5DatasetConfig(dataset_dir="datasets/cosmos_nemo_assets/videos/")
    dataset = H5Dataset(config)

    indices = [0, 1, 2, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data['video_name']=}\n"
                f"{data.keys()=}\n"
                "---"
            )
        )