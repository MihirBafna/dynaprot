import argparse
import yaml
import os
import pytorch_lightning as pl
import torch
import neptune as neptune
from pytorch_lightning.loggers import NeptuneLogger
from dynaprot.model.architecture import DynaProt
from dynaprot.data.datasets import DynaProtDataset , OpenFoldBatchCollator
from dynaprot.model.callbacks import EigenvalueLoggingCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Training dynaprot")
    parser.add_argument("--data_config",default="configs/data/atlas_config.yaml", type=str, required=False, help="Path to the data config YAML file")
    parser.add_argument("--model_config",default="configs/model/dynaprot_simple.yaml", type=str, required=False, help="Path to the model config YAML file")
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # Replace the token placeholder with the actual environment variable
    return config


def main():
    args = parse_args()
    
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    model_config["data_config"] = data_config
    
    model_config["train_params"]["neptune_api_key"] = os.getenv("NEPTUNE_API_TOKEN")

    neptune_logger = NeptuneLogger(
        project=model_config["train_params"]["project"],
        api_key=model_config["train_params"]["neptune_api_key"],
        tags=model_config["train_params"].get("tags", []),
        log_model_checkpoints=model_config["train_params"].get("log_model_checkpoints", True)
    )
    
    train_dataset = DynaProtDataset(data_config)
    print(len(train_dataset), model_config["train_params"]["batch_size"])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config["train_params"]["batch_size"],
        collate_fn=OpenFoldBatchCollator(),
        num_workers=12,
        shuffle=True,
    )

    # model = DynaProt(model_config)
    
    ckpt_path = model_config["checkpoint_path"]
    
    if ckpt_path != "":
        model = DynaProt.load_from_checkpoint(checkpoint_path=ckpt_path, config=model_config)
        print(f"Loaded from checkpoint path: {ckpt_path}")
    else:
        model = DynaProt(model_config)

    trainer = pl.Trainer(
        max_epochs=model_config["train_params"]["epochs"],
        logger=neptune_logger,
        accelerator=model_config["train_params"]["accelerator"],
        strategy=model_config["train_params"]["strategy"],
        devices=model_config["train_params"]["num_devices"],
        num_nodes=model_config["train_params"].get("num_nodes",1),
        precision=model_config["train_params"].get("precision", 32),
        log_every_n_steps=1,
        callbacks= [
            EigenvalueLoggingCallback(log_on_step=True, log_on_epoch=False)
        ],
    )

    trainer.fit(model,train_dataloader, val_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()
