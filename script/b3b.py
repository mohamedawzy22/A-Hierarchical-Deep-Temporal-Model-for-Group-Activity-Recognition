from engine.train import train_model
from engine.eval import evaluate
from Data.loader import FeatureDataset
import yaml
from torch.utils.data import DataLoader
from models.b3b import Baseline3B
from utils.logger import get_logger
from utils.seed import set_seed
import torch



if __name__ == "__main__":
    # Load config
    with open(r"config\b3_b.yml", "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["hyperparameters"]["seed"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = get_logger(exp_name="Baseline3", log_name="B3_b", logs_dir="logs")
    # Dataset & DataLoader
    train_dataset = FeatureDataset(
        h5_path=cfg["data"]["feat_dir"],
        cfg=cfg,
        split_name='train',
        categories=cfg["classes"]["labels_Group"],
        logger=logger,
        mode=cfg["data"]["mode"]
    )
    val_dataset = FeatureDataset(
            h5_path=cfg["data"]["feat_dir"],
            cfg=cfg,
            split_name='val',
            categories=cfg["classes"]["labels_Group"],
            logger=logger,
            mode=cfg["data"]["mode"]
        )
    test_dataset = FeatureDataset(
            h5_path=cfg["data"]["feat_dir"],
            cfg=cfg,
            split_name='test',
            categories=cfg["classes"]["labels_Group"],
            logger=logger,
            mode=cfg["data"]["mode"]
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
    model = Baseline3B(num_classes=8)
    train_model(model,
                train_loader,
                val_loader, 
                epochs=cfg["hyperparameters"]["num_epochs"], 
                device=device, 
                logger=logger,
                 cfg=cfg)
    
    

    # Load checkpoint
    checkpoint = torch.load(r"checkpoints\baseline_3b.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    evaluate(model, test_loader,device=device, class_names=cfg["classes"]["labels_Group"], logger=logger)