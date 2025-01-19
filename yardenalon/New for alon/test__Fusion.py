import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import json
# from reg import ResNet18Regressor
from torch.utils.data import DataLoader
from models.Fusion_model import FusionModel
from torchvision import transforms
from MM_options_fusion import parse_args
from train_Utils import load_data_image,load_data_sim

torch.manual_seed(42)
args = parse_args()


# Custom dataset to load images and their corresponding coordinates
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, numerical_input, numerical_output, transform=None):
        self.image_paths = image_paths
        self.numerical_input = numerical_input
        self.transform = transform
        self.numerical_output = numerical_output
        
        self.numerical_mean = np.array([1.89911162e-05, 6.42717825e-05, 1.03481931e-02])
        self.numerical_std = np.array([0.00878651, 0.0086231, 0.0046479])
        self.numerical_input = (self.numerical_input - self.numerical_mean) / self.numerical_std
        self.numerical_output = (self.numerical_output - self.numerical_mean) / self.numerical_std

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
        numerical_input = torch.tensor(self.numerical_input[idx][:3], dtype=torch.float32)
        numerical_output = torch.tensor(self.numerical_output[idx][:3], dtype=torch.float32)

        return image, numerical_input, numerical_output



def test_model(model, dataset, batch_size=args.batch_size, use_cuda=True):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset and dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    # Evaluate the model
    error_x = 0
    error_y = 0
    error_z = 0

    mse_loss = 0
    with torch.no_grad():
        # for data, targets in dataloader:
        for batch_idx, (images, numerical_input, numerical_output) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            numerical_output = numerical_output.to(device)
            numerical_input = numerical_input.to(device)

            # Forward pass
            outputs = model(numerical_input, images)

            numerical_mean = torch.tensor([1.89911162e-05, 6.42717825e-05, 1.03481931e-02]).to(device)
            numerical_std = torch.tensor([0.00878651, 0.0086231, 0.0046479]).to(device)

            targets_unnorm = (numerical_output * numerical_std) + numerical_mean
            outputs_unnorm = (outputs * numerical_std) + numerical_mean

            errors = (outputs_unnorm - targets_unnorm).abs()
            error_x += errors[:, 0].sum().item()
            error_y += errors[:, 1].sum().item()
            error_z += errors[:, 2].sum().item()

            # Print the results for each image
            for i in range(images.size(0)):
                print(f"Image {i + 1}:")
                print(f"Model Output: {outputs_unnorm[i].tolist()}")
                print(f"Actual Output: {targets_unnorm[i].tolist()}")
                print(f"Error: {errors[i].tolist()}\n")

            mse_loss += ((outputs - numerical_output) ** 2).sum().item()
            

    # Calculate average MSE loss
    mse_loss /= len(dataset)
    print(f"Mean Squared Error (MSE) on Test Data: {mse_loss:.8f}")
    mean_error_x = error_x / len(dataset)
    mean_error_y = error_y / len(dataset)
    mean_error_z = error_z / len(dataset)
    MSE_error = np.sqrt(np.float_power(mean_error_x,2) + np.float_power(mean_error_y,2) + np.float_power(mean_error_z,2))

    print(f"Mean Error for X: {mean_error_x:.4f}")
    print(f"Mean Error for Y: {mean_error_y:.4f}")
    print(f"Mean Error for Z: {mean_error_z:.4f}")
    print(f'mean MSE error: {MSE_error:.4f}')


if __name__ == "__main__":
    # Load the data
    image_paths , numerical_input = load_data_image(args.test_gen_JSON)
    numerical_output = load_data_sim(args.train_real_JSON)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, criterion and optimizer
    model = FusionModel(input_size_fc = args.input_size_fc, hidden_size1_fc = args.hidden_size1_fc , hidden_size2_fc = args.hidden_size2_fc, hidden_size3_fc = args.hidden_size3_fc, output_size_fc = args.output_size_fc).to(device)
    model.load_state_dict(torch.load(r"C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\train history fusion\model_train_2023-08-31_18_13_48\model_epoch_100.pth"))
    model.to(device)
    

    # Load the dataset and dataloader
    dataset = CustomDataset(image_paths, numerical_input, numerical_output)

    test_model(model, dataset,batch_size=args.batch_size, use_cuda=True)