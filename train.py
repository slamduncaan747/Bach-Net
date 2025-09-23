import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import BachNet
from ChoraleDataset import ChoraleDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.manual_seed(8)
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Hyperparameters
epochs = 1000

checkpoint_iters = 100
val_iters = 5

val_ratio = .1
batch_size = 4096

learning_rate = 0.002
dropout_rate = .1
weight_decay = 1e-5

# Create data loaders
t_set = ChoraleDataset('dataset.pkl')
t_size = int(len(t_set)*(1-val_ratio))
v_size = int(len(t_set)-t_size)
train_set, val_set = random_split(t_set, [t_size, v_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Define model, loss function, and optimizer
model = BachNet()
model.to(device)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    # Train on train_set
    for x, y_1, y_2, y_3 in train_loader:
        model.train()
        x = x.to(device)
        y_1, y_2, y_3 = y_1.to(device), y_2.to(device), y_3.to(device)
        
        optimizer.zero_grad()
        #Forward pass and loss for 3 heads individually, then combine and backpropagate
        output_1, output_2, output_3 = model(x)
        loss_1, loss_2, loss_3 = loss_function(output_1, y_1), loss_function(output_2, y_2), loss_function(output_3, y_3)
        final_loss = loss_1 + loss_2 + loss_3
        final_loss.backward()
        optimizer.step()
        total_loss += final_loss.item()
    total_loss /= len(train_loader)

    # Save checkpoint at intervals
    if (epoch+1) % checkpoint_iters == 0:
        path = f'checkpoints/checkpoint{epoch+1}.pth'
        torch.save(model.state_dict(), path)
    
    # Get validation loss at intervals
    if (epoch+1) % val_iters == 0:
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val_1, y_val_2, y_val_3 in val_loader:
                mode.eval()
                x_val = x_val.to(device)
                y_val_1, y_val_2, y_val_3 = y_val_1.to(device), y_val_2.to(device), y_val_3.to(device)
                logits_val_1, logits_val_2, logits_val_3 = model(x_val)
                loss_val_1, loss_val_2, loss_val_3 = loss_function(logits_val_1, y_val_1), loss_function(logits_val_2, y_val_2), loss_function(logits_val_3, y_val_3)
                loss_val = loss_val_1 + loss_val_2 + loss_val_3
                total_val_loss += loss_val.item()
        total_val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Loss: {total_val_loss:.4f}")
    else:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")