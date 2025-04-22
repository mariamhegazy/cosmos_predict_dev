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
PYTHONPATH=. python cosmos_predict1/diffusion/training/datasets/dataset_gear.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import os
import pickle
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict1.diffusion.training.datasets.dataset_utils import Resize_Preprocess, ToTensorVideo
import h5py

class h5Dataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        sequence_interval,
        num_frames,
        video_size,
        start_frame_interval=1,
    ):
        
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_frames

        # video_dir = os.path.join(self.dataset_dir, "videos")

        # self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_paths = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith(".h5")]
        # print(f"{len(self.video_paths)} trajectories in total")
        print(f"{len(self.video_paths)} videos in total")

        # self.t5_dir = os.path.join(self.dataset_dir, "labels")
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
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")
        # self.samples = self._init_samples(self.video_paths)
        # self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        # print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"





    def __len__(self):
        return self.total_samples

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path, frame_ids):
        frames, fps = self._load_video(video_path, frame_ids)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index):
        try:
            # sample = self.samples[index]
            # video_path = sample["video_path"]
            # frame_ids = sample["frame_ids"]
            video_index = index // self.samples_per_clip
            offset_idx = index % self.samples_per_clip
            video_path = self.video_paths[video_index]
            with h5py.File(video_path, "r") as f:
                total_frames = f["video"].shape[0]
                start_frame = offset_idx
                frame_ids = [start_frame + i * self.sequence_interval for i in range(self.sequence_length)]
                selected_frames = f["video"][frame_ids]
           
            assert total_frames >= self.sequence_length * self.sequence_interval, "Not enough frames"
            frames = torch.from_numpy(selected_frames)
            if frames.ndim == 4 and frames.shape[-1] == 3:  # [T, H, W, C] → [T, C, H, W]
                frames = frames.permute(3, 0, 1, 2)
            data = dict()

            # video, fps = self._get_frames(video_path, frame_ids)
            # video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            data["video"] = frames
            print(f"video shape: {data['video'].shape}")
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": "nothing",
                "start_frame_id": str(frame_ids[0]),
            }

            # Just add these to fit the interface
            # t5_embedding = np.load(sample["t5_embedding_path"])[0]
            # with open(sample["t5_embedding_path"], "rb") as f:
            #     t5_embedding = pickle.load(f)[0]
            dummy_t5_embedding = np.zeros((15, 1024), dtype=np.float32)
            data["t5_text_embeddings"] = torch.from_numpy(dummy_t5_embedding).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 10  # TO-DO: change this to dynamically get the correct fps for different datasets.
            data["image_size"] = torch.tensor([704, 1280, 704, 1280]).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 704, 1280).cuda()

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
    dataset = Dataset(
        dataset_dir="datasets/hdvila",
        sequence_interval=1,
        num_frames=57,
        video_size=[240, 360],
    )

    indices = [0, 13, 200, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            (
                f"{idx=} "
                f"{data['video'].sum()=}\n"
                f"{data['video'].shape=}\n"
                f"{data['video_name']=}\n"
                f"{data['t5_text_embeddings'].shape=}\n"
                "---"
            )
        )
