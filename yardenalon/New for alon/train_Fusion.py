import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
from models.Fusion_model import FusionModel
from MM_options_fusion import parse_args
import os
import datetime
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from train_Utils import load_data_image,load_data_sim, lr_schedule
import wandb

torch.manual_seed(42)



# import the parser options
args = parse_args()



# Custom dataset to load images and their corresponding coordinates
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, numerical_input, numerical_output, transform=None):
        self.image_paths = image_paths
        self.numerical_output = numerical_output
        self.numerical_input = numerical_input
        self.transform = transform

        self.numerical_mean = np.mean(self.numerical_output, axis=0)
        print("Mean input: ", np.mean(self.numerical_output, axis=0))
        self.numerical_std = np.std(self.numerical_output, axis=0)
        print("STD input: ", np.std(self.numerical_output, axis=0))
        self.numerical_output = (self.numerical_output - self.numerical_mean) / self.numerical_std
        self.numerical_input = (self.numerical_input - self.numerical_mean) / self.numerical_std

        # Transformations for the images
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        numerical_output = torch.tensor(self.numerical_output[idx][:3], dtype=torch.float32)
        numerical_input = torch.tensor(self.numerical_input[idx][:3], dtype=torch.float32)
        
        return image, numerical_input, numerical_output



def train_model(model,
                dataset,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                learning_rate=args.lr,
                use_cuda=True,
                save_interval=args.save_iter):

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # current_time = datetime.datetime.now()
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    os.makedirs(f'./train history fusion/model_train_{current_time}', exist_ok=True)
    save_directory = f'./train history fusion/model_train_{current_time}'
    log_file = open(f'./train history fusion/model_train_{current_time}/log_loss.txt', 'w')

    # write parameters to log
    log_file.write("Training details for fusion model trained at: {}\n".format(current_time))
    log_file.write(f"Image Model Architecture: {args.model_type}\n")
    log_file.write(f"Epochs: {args.num_epochs}\n")
    log_file.write(f"Learning Rate: {args.lr}\n")
    log_file.write(f"Batch size: {args.batch_size}\n")
    log_file.write("====================\n")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimal_loss = np.Inf
    train_losses = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=0.1)

    plt.figure(figsize=(10, 5))
    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, numerical_input, numerical_output) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            numerical_output = numerical_output.to(device)
            numerical_input = numerical_input.to(device)

            # Forward pass
            outputs = model(numerical_input, images)
            loss = criterion(outputs, numerical_output)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress
            if batch_idx % int(args.batch_size/3) == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.8f}")


            if running_loss < optimal_loss:
                save_path_best = os.path.join(save_directory, f"model_epoch_{epoch + 1}_best.pth")
                torch.save(model.state_dict(), save_path_best)
                print(f"Model saved at epoch {epoch + 1} for new optimal loss at {optimal_loss}")
                optimal_loss = running_loss

        # Save the model at specified intervals
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_directory, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch + 1}")

        scheduler.step()
        train_losses.append(running_loss / len(dataloader))
        log_file.write(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.8f}\n")
        log_file.write(f"Epoch [{epoch + 1}/{epochs}], Running Loss: {running_loss:.8f}\n")

        wandb.log({"MSE_loss": running_loss, "loss": loss.item()})

        plt.plot(range(epochs)[:epoch+1], train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.grid(True)
        plt.savefig('./train history fusion/model_train_{}/loss_plot.png'.format(current_time))
            

    # Save the trained model
    # torch.save(model.state_dict(), "resnet18_regressor_1.pth")


if __name__ == "__main__":
    # Load the data
    image_paths , numerical_input = load_data_image(args.train_SRC_JSON)
    numerical_output = load_data_sim(args.train_real_JSON)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, criterion and optimizer
    model = FusionModel(input_size_fc = args.input_size_fc, hidden_size1_fc = args.hidden_size1_fc , hidden_size2_fc = args.hidden_size2_fc, hidden_size3_fc = args.hidden_size3_fc, output_size_fc = args.output_size_fc).to(device)

    # Load the dataset and dataloader
    dataset = CustomDataset(image_paths, numerical_input, numerical_output)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.Project_name,
        
        # track hyperparameters and run metadata
        config={
        "batch size": args.batch_size,
        "learning_rate": args.lr,
        "architecture": args.model_type,
        "dataset": "1000 generated paired",
        "epochs": args.num_epochs,
        "epoch to start lr decay": args.decay_epochs,
        "Fully connected output size": args.output_size_fc,
        "Resnet18 output size": args.output_size_image,
        }
    )

    train_model(model, dataset, batch_size=args.batch_size, epochs=args.num_epochs, learning_rate=args.lr, use_cuda=True, save_interval=args.save_iter)