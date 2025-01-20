import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model2 import FusionModel as model2 #VGG image model architecture
from model3 import FusionModel as model3 #EfficientNetV2 image model architecture
from torchvision import transforms
import time
from MM_options import parse_args
import os
import datetime

#import the parser options
args = parse_args()


# Paths to the JSON files
real_JSON = r"C:\Users\roblab20\Documents\yardenalon\Compression\datasets\paired_finger\json45\aligned_real_5_transformed.json"  # TODO: Fill in the path to the real JSON file
sim_JSON = r"C:\Users\roblab20\Documents\yardenalon\Compression\datasets\paired_finger\json45\aligned_Sim_5_transformed.json"  # TODO: Fill in the path to the sim JSON file

# Load and extract data from JSON files
with open(real_JSON, "r") as real_file:
    real_data = json.load(real_file)

with open(sim_JSON, "r") as sim_file:
    sim_data = json.load(sim_file)

image_paths = [entry["frame"] for entry in real_data.values()]
numerical_inputs = [entry["pose_transformed"][0][:3] for entry in sim_data.values()]
numerical_outputs = [entry["pose_transformed"][0][:3] for entry in real_data.values()]
images = [Image.open(path) for path in image_paths]


class CustomDatasetWithTensor(Dataset):
    def __init__(self, images, numerical_inputs, numerical_outputs, transform=None):
        self.images = images
        self.numerical_inputs = numerical_inputs
        self.numerical_outputs = numerical_outputs
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if image:
            image = self.transform(image)
        else:
            image = torch.zeros(3, 256, 256)  # Adjusted the placeholder for non-existent images to the new size
        num_input = torch.tensor(self.numerical_inputs[idx], dtype=torch.float32)
        num_output = torch.tensor(self.numerical_outputs[idx], dtype=torch.float32)

        return (image, num_input), num_output
        # return image, num_input, num_output #NEW way


# Define the data loader
def get_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataset, batch_size=32, epochs=50, learning_rate=0.001, use_cuda=True, save_interval=5):
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # current_time = datetime.datetime.now()
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    os.makedirs(f'./train history/model_train_{current_time}', exist_ok=True)
    save_directory = f'./train history/model_train_{current_time}'
    log_file = open(f'./train history/model_train_{current_time}/log_loss.txt', 'w')

    #write parameters to log
    log_file.write("Training details for model trained at: {}\n".format(current_time))
    log_file.write(f"Image Model Architecture: {args.model_type}\n")
    log_file.write(f"Epochs: {args.num_epochs}\n")
    log_file.write(f"Learning Rate: {args.lr}\n")
    log_file.write(f"Batch size: {args.batch_size}\n")
    log_file.write("====================\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = get_dataloader(dataset, batch_size)
    optimal_loss = np.Inf

    for epoch in range(epochs):
        start_epoch_time = time.time()

        model.train()
        running_loss = 0.0
        for i, ((images, num_inputs), num_outputs) in enumerate(dataloader):
            if i % 3 == 0:
                start_iter_time = time.time()

            num_inputs, num_outputs = num_inputs.to(device), num_outputs.to(device)
            images = images.to(device)

            outputs = model(images, num_inputs)  # Placeholder
            loss = criterion(outputs, num_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 3 == 0:
                print(f"Iteration {i+1}, Loss: {loss.item():.8f}, Time taken for the last 100 images: {time.time() - start_iter_time:.2f} seconds")

            running_loss += loss.item()

            if running_loss < optimal_loss:
                save_path_best = os.path.join(save_directory, f"model_epoch_{epoch + 1}_best.pth")
                torch.save(model.state_dict(), save_path_best)
                print(f"Model saved at epoch {epoch + 1} for new optimal loss at {optimal_loss}")
                optimal_loss = running_loss

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.8f}")
        log_file.write(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.8f}\n")

        # Save the model at specified intervals
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_directory, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    print("Training finished!")



# define the model
# model = model2(3)
model = model3(3)
# Create the updated dataset
dataset = CustomDatasetWithTensor(images, numerical_inputs, numerical_outputs)

#Run the training loop
train_model(model, dataset, batch_size=args.batch_size, epochs=args.num_epochs , learning_rate=args.lr)  # Uncomment when model2_instance is available


