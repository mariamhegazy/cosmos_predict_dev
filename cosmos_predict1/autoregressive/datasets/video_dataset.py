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

from cosmos_predict1.autoregressive.configs.base.dataset import VideoDatasetConfig
from cosmos_predict1.autoregressive.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
import pickle

class VideoDataset(Dataset):
    def __init__(self, config: VideoDatasetConfig):
        """Video Dataset class for loading video-to-video generation data."""

        super().__init__()
        self.dataset_dir = config.dataset_dir
        self.sequence_interval = config.sequence_interval
        self.sequence_length = config.num_frames
        self.video_size = config.video_size
        self.start_frame_interval = config.start_frame_interval
        self.cache_file = os.path.join(self.dataset_dir, "video_samples.pkl")

        self.video_dir = self.dataset_dir
        # self.video_paths = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if f.endswith(".mp4")]
        # self.video_paths = [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".webm"))]
        self.video_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.video_dir)
            for f in files
            if f.endswith((".mp4", ".webm"))
        ]
        self.video_paths = self.video_paths[:10]
        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))

        # # Limit to 1000 videos
        # self.video_paths = self.video_paths[:1000]
        # print(f"Using first {len(self.video_paths)} videos out of total available")

        # # Load cache if available
        # if os.path.exists(self.cache_file):
        #     print(f"Loading cached samples from {self.cache_file}")
        #     with open(self.cache_file, "rb") as f:
        #         self.samples = pickle.load(f)

        # else:
        #     print("Generating samples from video files...")
        #     self.samples = self._init_samples(self.video_paths)  # Already capped at 1000
        #     self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))

        #     with open(self.cache_file, "wb") as f:
        #         pickle.dump(self.samples, f)
        #         print(f"Final cache saved to {self.cache_file}")

        print(f"{len(self.samples)} samples in total")
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

    def _init_samples(self, video_paths):
        samples = []
        with ThreadPoolExecutor(200) as executor:
            future_to_video_path = {
                executor.submit(self._load_and_process_video_path, video_path): video_path for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    # def _init_samples(self, video_paths, tmp_cache_file="video_samples_tmp.pkl", checkpoint_every=50):
    #         samples = []
    #         processed_paths = set()

    #         # Load existing partial cache if it exists
    #         if os.path.exists(tmp_cache_file):
    #             with open(tmp_cache_file, "rb") as f:
    #                 samples = pickle.load(f)
    #             processed_paths = {s["video_path"] for s in samples}
    #             print(f"🔁 Loaded {len(samples)} samples from checkpoint. Resuming...")

    #         remaining_paths = [vp for vp in video_paths if vp not in processed_paths]
    #         print(f"Processing {len(remaining_paths)} remaining videos")

    #         completed = 0
    #         lock = torch.multiprocessing.Lock()  # to make checkpointing thread-safe

    #         with ThreadPoolExecutor(max_workers=32) as executor:  # use fewer workers than 200
    #             future_to_path = {
    #                 executor.submit(self._load_and_process_video_path, vp): vp
    #                 for vp in remaining_paths
    #             }

    #             for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
    #                 video_path = future_to_path[future]
    #                 try:
    #                     result = future.result()
    #                     samples.extend(result)
    #                     processed_paths.add(video_path)
    #                     completed += 1

    #                     # Save checkpoint every N videos
    #                     if completed % checkpoint_every == 0:
    #                         with lock:
    #                             with open(tmp_cache_file, "wb") as f:
    #                                 pickle.dump(samples, f)
    #                         print(f"Saved checkpoint with {len(samples)} samples after {completed} videos")

    #                 except Exception as e:
    #                     print(f"Error processing {video_path}: {e}")

    #         return samples


    def _load_and_process_video_path(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        samples = []
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["video_path"] = video_path
            sample["orig_num_frames"] = n_frames
            sample["chunk_index"] = -1
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                sample["chunk_index"] += 1
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
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
            sample = self.samples[index]
            video_path = sample["video_path"]
            frame_ids = sample["frame_ids"]

            data = dict()

            video, fps = self._get_frames(video_path, frame_ids)
            data["video"] = video
            data["fps"] = fps
            data["num_frames"] = self.sequence_length
            data["orig_num_frames"] = sample["orig_num_frames"]
            data["chunk_index"] = sample["chunk_index"]
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

            print(f"Video: {video_path}, video shape: {data['video'].shape}, ")

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['video_path']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    config = VideoDatasetConfig(dataset_dir="/capstor/store/cscs/swissai/a03/datasets/OpenDV-YouTube/videos/")
    dataset = VideoDataset(config)

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
