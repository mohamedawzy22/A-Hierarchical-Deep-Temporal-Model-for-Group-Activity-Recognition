import os
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from data.annotation import load_tracking_annot
from models.b3a import Baseline3A

# =========================
# CONFIG
# =========================
class Config:
    dataset_root = r"E:\DL_Task_Mostafa_saad\VollyBall_Data"

    videos_root = os.path.join(dataset_root, "videos")
    tracking_root = os.path.join(dataset_root, "volleyball_tracking_annotation")

    save_path = "volleyball2_dataset.h5"

    num_players = 12
    feature_dim = 2048


# =========================
# DEVICE
# =========================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


# =========================
# BACKBONE
# =========================
def build_backbone():

    model = Baseline3A(num_classes=9)

    checkpoint = torch.load(
        r"C:\Users\ELBOSTAN\Downloads\B3A.pth",
        map_location="cpu"
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    backbone = model.backbone
    backbone.eval()

    return backbone


# =========================
# PREPROCESS
# =========================
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =========================
# CLIP LABELS
# =========================
def load_clip_labels(path):
    labels = {}

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            labels[parts[0].replace(".jpg", "")] = parts[1]

    return labels


# =========================
# STABLE TRACKING
# =========================
def build_stable_clip(frames, num_players, feature_dim):

    all_players = set()

    for f in frames:
        all_players.update(f["players"].keys())

    player_ids = sorted(list(all_players))[:num_players]
    pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}

    T = len(frames)

    clip_tensor = np.zeros(
        (num_players, T, feature_dim),
        dtype=np.float32
    )

    for t, frame in enumerate(frames):
        for pid, feat in frame["players"].items():
            if pid in pid_to_idx:
                clip_tensor[pid_to_idx[pid], t] = feat

    return clip_tensor


# =========================
# EXTRACT CLIP
# =========================
def extract_clip(
    clip_path,
    annot_file,
    backbone,
    preprocess,
    device,
    clip_labels,
    clip_name
):

    frame_boxes = load_tracking_annot(annot_file)
    label = clip_labels.get(clip_name, "unknown")

    frames = []

    with torch.no_grad():

        for frame_id, boxes in frame_boxes.items():

            img_path = os.path.join(clip_path, f"{frame_id}.jpg")
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")

            crops = []
            player_ids = []

            for b in boxes:
                x1, y1, x2, y2 = b.box
                crop = image.crop((x1, y1, x2, y2))

                crops.append(preprocess(crop))
                player_ids.append(b.player_ID)

            if len(crops) == 0:
                continue

            batch = torch.stack(crops).to(device)

            feats = backbone(batch)
            feats = feats.view(feats.size(0), -1).cpu().numpy()

            frames.append({
                "frame_id": frame_id,
                "players": {
                    pid: feats[i]
                    for i, pid in enumerate(player_ids)
                }
            })

    return label, frames


# =========================
# SAVE TO HDF5
# =========================
def save_hdf5(data, path):

    print("🔥 Saving to HDF5...")

    with h5py.File(path, "w") as h5:

        for video_name, clips in data.items():

            video_group = h5.create_group(video_name)

            for clip_name, clip in clips.items():

                clip_group = video_group.create_group(clip_name)

                clip_group.create_dataset(
                    "tensor",
                    data=clip["tensor"],
                    compression="gzip",
                    compression_opts=4
                )

                clip_group.attrs["label"] = clip["label"]

    print(f"✅ Saved HDF5 dataset → {path}")


# =========================
# MAIN PIPELINE
# =========================
def main():

    cfg = Config()
    device = get_device()

    backbone = build_backbone().to(device)
    preprocess = get_preprocess()

    all_data = {}

    videos = sorted(os.listdir(cfg.videos_root))

    for video in tqdm(videos, desc="Videos"):

        video_path = os.path.join(cfg.videos_root, video)
        if not os.path.isdir(video_path):
            continue

        clip_label_file = os.path.join(video_path, "annotations.txt")
        clip_labels = load_clip_labels(clip_label_file)

        clips = sorted(os.listdir(video_path))

        video_dict = {}

        for clip in clips:

            clip_path = os.path.join(video_path, clip)

            if not os.path.isdir(clip_path):
                continue

            annot_file = os.path.join(
                cfg.tracking_root,
                video,
                clip,
                f"{clip}.txt"
            )

            label, frames = extract_clip(
                clip_path,
                annot_file,
                backbone,
                preprocess,
                device,
                clip_labels,
                clip
            )

            tensor = build_stable_clip(
                frames,
                cfg.num_players,
                cfg.feature_dim
            )

            video_dict[clip] = {
                "label": label,
                "tensor": tensor
            }

        all_data[video] = video_dict

    save_hdf5(all_data, cfg.save_path)


if __name__ == "__main__":
    main()