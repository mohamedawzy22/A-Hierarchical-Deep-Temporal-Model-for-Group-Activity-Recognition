import torch
from utils.logger import get_logger
from utils.seed import set_seed
from torch.utils.data import DataLoader
from models.b6 import Baseline6
from engine.train import train_model
from engine.eval import evaluate
from Data.loader import FeatureDataset
import yaml

if __name__ == "__main__":
    # Load config
    with open("config/b6.yml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['hyperparameters']['seed'])

    logger = get_logger(exp_name="Baseline6", log_name="B6", logs_dir="logs")

    # Dataset
    train_dataset = FeatureDataset(
        h5_path=cfg['data']['feat_dir'],
        categories=cfg['classes']['labels_Group'],
        cfg=cfg,
        split_name='train',
        logger= logger,
        mode=cfg["data"]["mode"]
        
    )
    val_dataset = FeatureDataset(
        h5_path=cfg['data']['feat_dir'],
        categories=cfg['classes']['labels_Group'],
        cfg=cfg,
        split_name='val',
        logger= logger,
        mode = cfg['data']['mode']
        
    )
    test_dataset = FeatureDataset(
        h5_path=cfg['data']['feat_dir'],
        categories=cfg['classes']['labels_Group'],
        cfg=cfg,
        split_name='test',
        logger= logger,
        mode = cfg['data']['mode']
    )

    # DataLoaders بدون multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg['hyperparameters']['batch_size'], shuffle=False, num_workers=0)



model = Baseline6(input_size=cfg['hyperparameters']['input_size'], 
                  hidden_size=cfg['hyperparameters']['hidden_size'],
                  num_classes=cfg['classes']['num_classes'])

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


checkpoint = torch.load(r"checkpoints\baseline_6.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluate
evaluate(model, test_loader,device=device, class_names=cfg["classes"]["labels_Group"], logger=logger)

