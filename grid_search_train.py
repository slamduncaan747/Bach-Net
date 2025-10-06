import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import BachNet
from ChoraleDataset import ChoraleDataset
import os
import wandb
from sklearn.model_selection import ParameterGrid

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(8)

grid = {
    "learning_rate": [.001],
    "dropout_rate": [0.25, .3, .4],
    "weight_decay": [1e-4, 2e-5, 3e-3],
}

config_list = list(ParameterGrid(grid))

def config_train(config, epochs = 20):
    # Hyperparameters
    val_iters = 5

    batch_size = 1024

    learning_rate = config['learning_rate']
    dropout_rate = config['dropout_rate']
    weight_decay = config['weight_decay']
    # Initialize Logging
    wandb.init(
        project="Bach-Net Grid Search",
        config={
            "epochs": epochs,
            "val_iters": val_iters,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay":  weight_decay,
            "dropout_rate": dropout_rate,
        }
    )
    wandb_config = wandb.config

    # Create data loaders
    train_set = ChoraleDataset('Bach-Net/train_dataset.pkl')
    val_set = ChoraleDataset('Bach-Net/val_dataset.pkl')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Define model, loss function, and optimizer
    model = BachNet()
    start_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    model.to(device)

    pos_weight_value = 25.0
    pos_weight = torch.tensor([pos_weight_value]).to(device)

    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    wandb.watch(model, log='all', log_freq=100)

    for epoch in range(0, epochs):
        total_loss = 0
        # Train on train_set
        model.train()
        best_val_loss = float('inf')
        for x, y_1, y_2, y_3 in train_loader:
            x = x.to(device)
            y_1, y_2, y_3 = y_1.to(device), y_2.to(device), y_3.to(device)
            
            optimizer.zero_grad()
            #Forward pass and loss for 3 heads individually, then combine and backpropagate
            output_1, output_2, output_3 = model(x)
            loss_1, loss_2, loss_3 = loss_function(output_1, y_1), loss_function(output_2, y_2), loss_function(output_3, y_3)
            final_loss = loss_1 + loss_2 + loss_3
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += final_loss.item()
        total_loss /= len(train_loader)
        
        # Get validation loss at intervals
        total_val_loss = 0
        if (epoch+1) % val_iters == 0:
            model.eval()
            with torch.no_grad():
                for x_val, y_val_1, y_val_2, y_val_3 in val_loader:    
                    x_val = x_val.to(device)
                    y_val_1, y_val_2, y_val_3 = y_val_1.to(device), y_val_2.to(device), y_val_3.to(device)
                    logits_val_1, logits_val_2, logits_val_3 = model(x_val)
                    loss_val_1, loss_val_2, loss_val_3 = loss_function(logits_val_1, y_val_1), loss_function(logits_val_2, y_val_2), loss_function(logits_val_3, y_val_3)
                    loss_val = loss_val_1 + loss_val_2 + loss_val_3
                    total_val_loss += loss_val.item()
            total_val_loss /= len(val_loader)
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Loss: {total_val_loss:.4f}")
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_loss,
                "val_loss": total_val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train_loss": total_loss})
        
        scheduler.step()
    wandb.finish()
    print(f"Config: {config}, Best Validation Loss: {best_val_loss:.4f}")

for config in config_list:
    config_train(config, epochs=20)