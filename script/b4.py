import torch
from utils.transform import get_train_transform, get_val_transform
from utils.logger import get_logger
from utils.seed import set_seed
from torch.utils.data import DataLoader
from models.b4 import Baseline4
from engine.train import train_model
from engine.eval import evaluate
from Data.loader import FrameLevelDataset
import yaml



if __name__ == "__main__":

    
    
    # Load config
    with open("config/b4.yml", "r") as f:
        cfg = yaml.safe_load(f)


    set_seed(cfg['hyperparameters']['seed'])

    logger = get_logger(exp_name="Baseline4", log_name="B4", logs_dir="logs")

    train_transform = get_train_transform()
    val_transform = get_val_transform()
    # Dataset
    train_dataset = FrameLevelDataset(
        pkl_path=cfg['data']['pkl'],
        labels=cfg['classes']['labels'],
        cfg=cfg,
        split_name='train',
        dataset_root=cfg['data']['data_dir'],
        transform=train_transform,
        logger=logger,
        mode=cfg['mode']['is_sequence']
        
    )
    val_dataset = FrameLevelDataset(
        pkl_path=cfg['data']['pkl'],
        labels=cfg['classes']['labels'],
        cfg=cfg,
        split_name='val',
        dataset_root=cfg['data']['data_dir'],
        transform=val_transform,
        logger=logger,
        mode = cfg['mode']['is_sequence']
        
    )
    test_dataset = FrameLevelDataset(
        pkl_path=cfg['data']['pkl'],
        labels=cfg['classes']['labels'],
        cfg=cfg,
        split_name='test',
        dataset_root=cfg['data']['data_dir'],
        transform=val_transform,
        logger=logger,
        mode = cfg['mode']['is_sequence']
    )

    # DataLoaders بدون multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=False, num_workers=0)


model = Baseline4(num_classes=cfg['classes']['num_classes'])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    logger=logger,
    cfg=cfg,
    epochs=cfg['hyperparameters']['num_epochs'],
)


checkpoint = torch.load(r"checkpoints\baseline_4.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
evaluate(model, test_loader,device=device, class_names=cfg["classes"]["labels"], logger=logger)
