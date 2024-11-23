from torch.utils.data import DataLoader
from dataset import ChessDataset
import torch.nn as nn
import torch.optim as optim
from model import ChessBotCNN
import torch


# Initialize the dataset
dataset = ChessDataset('chess_data.parquet')

# Create the DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=64,      
    shuffle=True,
    num_workers=4,      
    pin_memory=True     
)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize the model
model = ChessBotCNN()
model = model.to(device)  # Move model to GPU if available

# Loss functions
policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # For policy output
value_loss_fn = nn.MSELoss()                           # For value output

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Set the number of epochs
log_interval = 100  # Logging interval

for epoch in range(1, num_epochs + 1):
    model.train()
    for batch_idx, (board_tensor, policy_tensor, value_tensor) in enumerate(dataloader):
        board_tensor = board_tensor.to(device)
        policy_tensor = policy_tensor.to(device)
        value_tensor = value_tensor.to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_pred, value_pred = model(board_tensor)

        # Compute losses
        policy_loss = policy_loss_fn(policy_pred, policy_tensor)
        value_loss = value_loss_fn(value_pred.squeeze(), value_tensor)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                  f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")