import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import BachNet
from ChoraleDataset import ChoraleDataset
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(8)



# Hyperparameters
epochs = 1000

checkpoint_iters = 50
val_iters = 5
increased_checkpoint_epoch = 600
increased_checkpoint_iters = 10

val_ratio = .1
batch_size = 1024

learning_rate = 0.004
dropout_rate = .1
weight_decay = 2e-5

from_checkpoint = True
checkpoint_path = '../content/drive/MyDrive/BachNet/checkpoints/'
best_checkpoint_path = '../content/drive/MyDrive/BachNet/checkpoints/best/'
checkpoint_epoch = 600

# Initialize Logging
wandb.init(
    project="Bach-Net",
    config={
        "epochs": epochs,
        "checkpoint_iters": checkpoint_iters,
        "val_iters": val_iters,
        "val_ratio": val_ratio,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay":  weight_decay,
        "dropout_rate": .1,
    }
)
config = wandb.config

# Create data loaders
t_set = ChoraleDataset('Bach-Net/dataset.pkl')
t_size = int(len(t_set)*(1-val_ratio))
v_size = int(len(t_set)-t_size)
train_set, val_set = random_split(t_set, [t_size, v_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Define model, loss function, and optimizer
model = BachNet()
start_epoch = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

if from_checkpoint:
    full_path = os.path.join(checkpoint_path, checkpoint_name)
    checkpoint = torch.load(full_path, map_location=device)
    if(isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        model.load_state_dict(checkpoint)
        start_epoch = checkpoint_epoch

model.to(device)
loss_function = torch.nn.BCEWithLogitsLoss()
wandb.watch(model, log='all', log_freq=100)

top5 = []
# Training loop
for epoch in range(start_epoch, epochs):
    total_loss = 0
    # Train on train_set
    model.train()
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
    # Save checkpoint at intervals
    if (epoch+1) % (checkpoint_iters) == 0:
        os.makedirs('checkpoints', exist_ok=True)
        path = f'checkpoints/checkpoint{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    
    if (epoch+1) % (increased_checkpoint_iters) == 0 and epoch+1 >= increased_checkpoint_epoch:
        top5last = float('inf')
        if len(top5) == 5:
            top5.sort(key=lambda x: x[0])
            top5last = top5[-1][0]
        if len(top5) < 5 or total_val_loss < top5last:
            best_path = f'checkpoints/best/best_model_epoch_{epoch+1}_loss_{total_val_loss:.4f}.pth'
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_loss': total_val_loss
            }, best_path)

            if len(top5) == 5:
                loss_to_remove, path_to_remove = top5.pop(-1)
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
            
            top5.append((total_val_loss, best_path))
    scheduler.step()
wandb.finish()