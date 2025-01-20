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
    def __init__(self, image_paths, numerical_input, transform=None):
        self.image_paths = image_paths
        self.numerical_input = numerical_input
        self.transform = transform

        self.numerical_mean = np.array([1.89911162e-05, 6.42717825e-05, 1.03481931e-02])
        self.numerical_std = np.array([0.00878651, 0.0086231, 0.0046479])
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
        numerical_input = torch.tensor(self.numerical_input[idx][:3], dtype=torch.float32)


        return image, numerical_input

def test_model(model, dataset, original_data, batch_size=5000, use_cuda=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the dataset and dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    # Dictionary to store updated JSON data
    updated_json_data = {}
    
    with torch.no_grad():
        for batch_idx, (images, numerical_input) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            numerical_input = numerical_input.to(device)

            # Forward pass
            outputs = model(numerical_input, images)

            numerical_mean = torch.tensor([1.89911162e-05, 6.42717825e-05, 1.03481931e-02]).to(device)
            numerical_std = torch.tensor([0.00878651, 0.0086231, 0.0046479]).to(device)

            outputs_unnorm = (outputs * numerical_std) + numerical_mean

            # Update JSON data and Print the results for each image
            for i, entry in enumerate(original_data.keys()):
                if i == 4999:
                    i = 0 
                    break

                elif batch_idx*batch_size + i >= 5000:
                    break


                # print(f"Image {i + 1}:")
                # print(f"Model Output: {outputs_unnorm[i].tolist()}")

                # image_path = dataset.image_paths[batch_idx * batch_size + i]
                # if entry["frame"] == image_path:
                # image_name = "frame_" + image_path.split("\\\\")[-1].split('.')[0]
                # img_num = image_path.split('\\')[-1][:-4]
                # image_name = f"frame_{img_num}"
                image_data = original_data[entry]

                new=image_data["pose_transformed"]
                new2= [outputs_unnorm[i].tolist(),new[1]]
                image_data["pose_transformed_old"] = image_data["pose_transformed"]
                image_data["pose_transformed"] = new2
                # image_data["pose_transformed"][0] = outputs_unnorm[i].tolist()
                updated_json_data[entry] = image_data

            else:
                continue


    # Save the updated JSON data
    with open("SRC_4_80epochs_Fusion2.json", "w") as json_file:
        json.dump(updated_json_data, json_file, indent=3)

if __name__ == "__main__":
    Json_path = r'C:\Users\yarde\Documents\GitHub\Hneg_SRC\Compression\datasets\Finger_4\json_data\SRC_4_80epochs_transformed.json'
    # Load the data
    image_paths , numerical_input = load_data_image(Json_path)
 

    # Load the original JSON data
    with open(Json_path, "r") as json_file:
        original_data = json.load(json_file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, criterion and optimizer
    model = FusionModel(input_size_fc = args.input_size_fc, hidden_size1_fc = args.hidden_size1_fc , hidden_size2_fc = args.hidden_size2_fc, hidden_size3_fc = args.hidden_size3_fc, output_size_fc = args.output_size_fc).to(device)
    model.load_state_dict(torch.load(r"C:\Users\yarde\Documents\GitHub\Hneg_SRC\Pre-Process_MM\train history fusion\model_train_2023-08-31_12_58_48\model_epoch_98_best.pth"))
    model.to(device)

    # Load the dataset and dataloader
    dataset = CustomDataset(image_paths, numerical_input)

    test_model(model, dataset, original_data, batch_size=5000, use_cuda=True)
