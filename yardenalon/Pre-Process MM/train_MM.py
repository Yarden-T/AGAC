import torch.optim as optim
from model2 import FusionModel
from MM_options import parse_args
import torch.nn as nn
from train_loader import get_train_loader
import os
import torch

args = parse_args()

# Load data for training
real_json_path = os.path.join(args.base_path, args.real_JSON)
sim_json_path = os.path.join(args.base_path, args.sim_JSON)
train_loader = get_train_loader(real_json_path, sim_json_path)

# Create 'results' directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create model directory under 'results' if it doesn't exist
model_dir = os.path.join(results_dir, args.Model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create loss log file
loss_log_path = os.path.join(model_dir, 'Loss_log.txt')
with open(loss_log_path, 'w') as loss_log:
    loss_log.write("Epoch,Batch,Loss\n")

if __name__ == '__main__':
    # Define your FusionModel instance
    model = FusionModel(args.num_numerical_features)

    # Define a loss function (e.g., CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer (e.g., SGD or Adam)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Initialize iteration counter
    iteration = 0

    # Training loop
    for epoch in range(args.num_epochs):
        for batch_idx, (image, numerical_data, target) in enumerate(train_loader):
            # Move data to the device (CPU or GPU)
            image, numerical_data, target = image.to(args.device), numerical_data.to(args.device), target.to(args.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(image, numerical_data)
            
            # Compute the loss
            loss = criterion(output, target)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            iteration += 1
            
            # Print training progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                # Append training progress to loss log
                with open(loss_log_path, 'a') as loss_log:
                    loss_log.write(f"{epoch+1},{batch_idx+1},{loss.item()}\n")
            
            # Save the model every save_iter iterations
            if iteration % args.save_iter == 0:
                save_filename = f"{args.Model_name}_{iteration}.pt"
                save_path = os.path.join(model_dir, save_filename)
                torch.save(model.state_dict(), save_path)
                print(f"Saved model checkpoint at iteration {iteration}")
