from engine.train import train_model
from engine.eval import evaluate
from Data.loader import PersonLevelDataset
import yaml
from torch.utils.data import DataLoader
from models.b3a import Baseline3A
from utils.logger import get_logger
from utils.seed import set_seed
from utils.transform import get_train_transform, get_val_transform
import torch


if __name__ == "__main__":
    # Load config
    with open(r"config\b3_a.yml", "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["hyperparameters"]["seed"])
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = get_logger(exp_name="Baseline3", log_name="B3_a", logs_dir="logs")
    # Dataset & DataLoader
    train_dataset = PersonLevelDataset(
        pkl_path=cfg["data"]["pkl"],
        dataset_root=cfg["data"]["data_dir"],
        split_name="train",
        cfg=cfg,
        categories=cfg["classes"]["Labels_Person"],
        transform=train_transform,
        logger=logger
        
    )
    val_dataset = PersonLevelDataset(
            pkl_path=cfg["data"]["pkl"],
            dataset_root=cfg["data"]["data_dir"],
            split_name="val",
            cfg=cfg,
            categories=cfg["classes"]["Labels_Person"],
            transform=val_transform,
            logger=logger
            
        )
    test_dataset = PersonLevelDataset(
            pkl_path=cfg["data"]["pkl"],
            dataset_root=cfg["data"]["data_dir"],
            split_name="test",
            cfg=cfg,
            categories=cfg["classes"]["Labels_Person"],
            transform=val_transform,
            logger=logger
        )
    

    train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["hyperparameters"]["batch_size"],
            shuffle=True
        )

    val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["hyperparameters"]["batch_size"],
            shuffle=False
        )


    test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["hyperparameters"]["batch_size"],
            shuffle=False)

    # Model
    model = Baseline3A(num_classes=cfg["classes"]["num_classes_person"])
    train_model(model,
                train_loader,
                val_loader, 
                epochs=cfg["hyperparameters"]["num_epochs"], 
                device=device, 
                logger=logger,
                cfg=cfg)
    
    

    # Load checkpoint
    checkpoint = torch.load(r"checkpoints\baseline_3a.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    evaluate(model, test_loader,device=device, class_names=cfg["classes"]["Labels_Person"], logger=logger)