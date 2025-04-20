import argparse
import yaml
import os
import re
import pytorch_lightning as pl
import torch
import neptune as neptune
import numpy as np
import random
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dynaprot.model.architecture import DynaProt
from dynaprot.data.datasets import DynaProtDataset , OpenFoldBatchCollator
from dynaprot.model.callbacks import EigenvalueLoggingCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Training dynaprot")
    parser.add_argument("--data_config",default="configs/data/atlas_config.yaml", type=str, required=False, help="Path to the data config YAML file")
    parser.add_argument("--model_config",default="configs/model/dynaprot_simple.yaml", type=str, required=False, help="Path to the model config YAML file")
    parser.add_argument("--name",type=str, required=True, help="Run name identifier")
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # Replace the token placeholder with the actual environment variable
    return config


def main():
    args = parse_args()
    set_seed(42)
    
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    model_config["data_config"] = data_config
    
    model_config["train_params"]["neptune_api_key"] = os.getenv("NEPTUNE_API_TOKEN")

    
    train_dataset = DynaProtDataset(data_config, split="train")
    val_dataset = DynaProtDataset(data_config, split="val")
    test_dataset = DynaProtDataset(data_config, split="test")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config["train_params"]["batch_size"],
        collate_fn=OpenFoldBatchCollator(),
        num_workers=12,
        shuffle=True,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=model_config["train_params"]["batch_size"],
        collate_fn=OpenFoldBatchCollator(),
        num_workers=12,
        shuffle=False,
    )
    
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=model_config["train_params"]["batch_size"],
    #     collate_fn=OpenFoldBatchCollator(),
    #     num_workers=12,
    #     shuffle=False,
    # )

    
    ckpt_path = model_config["checkpoint_path"]
    
    if ckpt_path != "":
        model = DynaProt.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=model_config)
        run_id = re.search(r'\.neptune/([^/]+)', ckpt_path).group(1)
        run = neptune.init_run(
            with_id=run_id,
            project=model_config["train_params"]["project"],
            api_token=model_config["train_params"]["neptune_api_key"],
            tags=model_config["train_params"].get("tags", []),
            ) 
        neptune_logger = NeptuneLogger(
            run=run,
            log_model_checkpoints=model_config["train_params"].get("log_model_checkpoints", True)
        )
    
        print(f"Loaded from checkpoint path: {ckpt_path} and run id: {run_id}")
    else:
        model = DynaProt(model_config)

        neptune_logger = NeptuneLogger(
            project=model_config["train_params"]["project"],
            name=args.name,
            api_key=model_config["train_params"]["neptune_api_key"],
            tags=model_config["train_params"].get("tags", []),
            log_model_checkpoints=model_config["train_params"].get("log_model_checkpoints", True)
        )
    
    print(model)
    
    trainer = pl.Trainer(
        max_epochs=model_config["train_params"]["epochs"],
        logger=neptune_logger,
        accelerator=model_config["train_params"]["accelerator"],
        strategy=model_config["train_params"]["strategy"],
        devices=model_config["train_params"]["num_devices"],
        num_nodes=model_config["train_params"].get("num_nodes",1),
        precision=model_config["train_params"].get("precision", 32),
        # log_every_n_steps=10,
        callbacks= [
            # EigenvalueLoggingCallback(log_on_step=True, log_on_epoch=False)
            ModelCheckpoint(
                dirpath="ckpt/", 
                filename=args.name+"-{step:06d}", 
                save_top_k=-1,  
                every_n_train_steps=500  
            )
        ],
    )

    torch.autograd.set_detect_anomaly(True)
    trainer.fit(model,train_dataloader, val_dataloaders=val_dataloader)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    

if __name__ == "__main__":
    main()
