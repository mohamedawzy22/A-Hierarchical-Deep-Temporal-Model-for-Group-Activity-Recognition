"""
========================================================
DATASET LOADERS SUMMARY
========================================================

1) FrameLevelDataset
--------------------
Used in: b1,b4
Purpose:Load full RGB frames for frame-based or temporal sequence classification.


2) PersonLevelDataset
---------------------
Used in: b3
Purpose:Crop players using bounding boxes and train a player-level ResNet classifier.

The trained model is later used to extract deep player features.


3) FeatureDataset
-----------------
Used in: b5,b6,b7,b8
Purpose:
Load extracted player features from HDF5 files.

Features are generated using the trained player-level model from Baseline 3.

Typical tensor shape: (12, T, 2048)

========================================================
"""


import torch
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
import h5py


class FrameLevelDataset(Dataset):

    def __init__(
        self,
        pkl_path,
        labels,
        split_name,
        dataset_root,
        cfg,
        transform=None,
        mode="frame",
        logger=None
    ):

        assert mode in ["frame", "sequence"]

        self.dataset_root = dataset_root
        self.transform = transform
        self.mode = mode
        self.logger = logger

        self.video_ids = set(
            cfg["data"]["split"][split_name]
        )

        self.class_to_idx = {
            c: i for i, c in enumerate(labels)
        }

        self.samples = []

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for video_id, video in data.items():

            if video_id not in self.video_ids:
                continue

            for clip_id, clip in video.items():

                label = clip["category"]

                if label not in self.class_to_idx:
                    continue

                frame_ids = sorted(
                    clip["frame_boxes_dct"].keys()
                )

                # ======================
                # FRAME MODE
                # ======================
                if mode == "frame":

                    for fid in frame_ids:

                        self.samples.append((
                            video_id,
                            clip_id,
                            fid,
                            self.class_to_idx[label]
                        ))

                # ======================
                # SEQUENCE MODE
                # ======================
                else:

                    if len(frame_ids) >= 9:

                        self.samples.append((
                            video_id,
                            clip_id,
                            frame_ids[:9],
                            self.class_to_idx[label]
                        ))

        if logger:
            logger.info(
                f"Loading dataset | "
                f"split={split_name} | "
                f"mode={mode} | "
                f"Total samples: {len(self.samples)}"
            )

    def load_image(self, video_id, clip_id, frame_id):

        path = os.path.join(
            self.dataset_root,
            "videos",
            str(video_id),
            str(clip_id),
            f"{frame_id}.jpg"
        )

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_id, clip_id, item, label = self.samples[idx]

        # ======================
        # FRAME
        # ======================
        if self.mode == "frame":

            img = self.load_image(
                video_id,
                clip_id,
                item
            )

            return img, torch.tensor(label)

        # ======================
        # SEQUENCE
        # ======================
        frames = torch.stack([
            self.load_image(video_id, clip_id, fid)
            for fid in item
        ])

        return frames, torch.tensor(label)
        



###############################


class PersonLevelDataset(Dataset):

    def __init__(
        self,
        pkl_path,
        cfg,
        split_name,
        categories,
        dataset_root,
        transform=None,
        logger=None
    ):

        self.pkl_path = pkl_path
        self.dataset_root = dataset_root
        self.transform = transform
        self.logger = logger
        self.categories = categories

        # =========================
        # VIDEO IDS FROM CONFIG
        # =========================
        self.video_ids = set(cfg["data"]["split"][split_name])

        # =========================
        # LABEL MAPPING
        # =========================
        self.class_to_idx = {
            c: i for i, c in enumerate(categories)
        }

        self.samples = []

        # =========================
        # LOAD PKL
        # =========================
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)

        # =========================
        # BUILD SAMPLES
        # =========================
        for video_id, video in data.items():

            if str(video_id) not in self.video_ids:
                continue

            for clip_id, clip in video.items():

                frame_boxes_dct = clip["frame_boxes_dct"]

                frame_ids = sorted(frame_boxes_dct.keys())

                for frame_id in frame_ids:

                    boxes = frame_boxes_dct[frame_id]

                    for box_info in boxes:

                        label = box_info.category

                        if label not in self.class_to_idx:
                            continue

                        self.samples.append((
                            video_id,
                            clip_id,
                            frame_id,
                            box_info.box,
                            self.class_to_idx[label]
                        ))

        # =========================
        # LOG
        # =========================
        if self.logger:
            self.logger.info(
                f" PERSON dataset ready | "
                f"split={split_name} | "
                f"samples={len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_id, clip_id, frame_id, box, label = self.samples[idx]

        # =========================
        # IMAGE PATH
        # =========================
        path = os.path.join(
            self.dataset_root,
            "videos",
            str(video_id),
            str(clip_id),
            f"{frame_id}.jpg"
        )

        # =========================
        # LOAD IMAGE
        # =========================
        img = Image.open(path).convert("RGB")

        # =========================
        # CROP PLAYER
        # =========================
        x1, y1, x2, y2 = box

        img = img.crop((x1, y1, x2, y2))

        # =========================
        # TRANSFORM
        # =========================
        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(label, dtype=torch.long)
        )
    

##################################

import h5py
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):

    def __init__(
        self,
        h5_path,
        cfg,
        split_name,
        categories,
        mode="sequence",
        logger=None
    ):

        assert mode in ["sequence", "frame"]

        self.h5_path = h5_path
        self.categories = categories
        self.mode = mode
        self.logger = logger

        # Video IDs from config
        self.video_ids = set(cfg['data']['split'][split_name])

        self.samples = []

        with h5py.File(self.h5_path, "r") as h5:

            for video_id in h5.keys():

                if str(video_id) not in self.video_ids:
                    continue

                video = h5[video_id]

                for clip_id in video.keys():

                    clip = video[clip_id]

                    # (12, T, 2048)
                    tensor = clip["tensor"][:]

                    label = clip.attrs["label"]

                    # ==================================
                    # Sequence Mode
                    # ==================================
                    if self.mode == "sequence":

                        self.samples.append((tensor, label))

                    # ==================================
                    # Frame Mode
                    # ==================================
                    else:

                        # tensor shape:
                        # (12, T, 2048)

                        T = tensor.shape[1]

                        for t in range(T):

                            # (12, 2048)
                            frame_tensor = tensor[:, t, :]

                            self.samples.append(
                                (frame_tensor, label)
                            )

        # Label mapping
        self.class_to_idx = {
            c: i for i, c in enumerate(categories)
        }

        self.samples = [
            (x, self.class_to_idx[y])
            for x, y in self.samples
        ]

        if self.logger:
            self.logger.info(
                f"Loading dataset | split={split_name} | mode={mode} | Total samples: {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x, y = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )