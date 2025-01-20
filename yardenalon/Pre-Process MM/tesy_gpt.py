# # Importing necessary libraries and modules
#
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import model3
#



# def test_model3_old(json_path, model_path, print_frequency=50):
#     # Check for CUDA availability
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Load the trained model
#     model = model3.FusionModel().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     # Define the transformations for images
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # Load and extract data from the JSON file
#     with open(json_path, "r") as file:
#         data = json.load(file)
#
#     # Create new JSON structure
#     new_data = []
#
#     # Counter for processed entries
#     processed_count = 0
#
#     for entry in data.values():
#         # Extract numerical data
#         numerical_data = torch.tensor(entry["pose_transformed"][0][:3], dtype=torch.float32).to(device)
#
#         # Load and transform the image
#         image_path = entry["frame"]
#         image = Image.open(image_path)
#         image_tensor = transform(image).unsqueeze(0).to(device)
#
#         # Get model prediction
#         with torch.no_grad():
#             numerical_output = model(image_tensor, numerical_data.unsqueeze(0))
#
#         # Update the JSON entry
#         entry["pose_transformed_old"] = entry["pose_transformed"]
#         entry["pose_transformed"] = numerical_output.cpu().squeeze(
#             0).tolist()  # Move data back to CPU for JSON serialization
#         new_data.append(entry)
#
#         # Increment the processed counter
#         processed_count += 1
#
#         # Print processed count at specified frequency
#         if processed_count % print_frequency == 0:
#             print(f"Processed {processed_count} entries.")
#
#             print("pose_transformed_old:", entry["pose_transformed_old"][0])
#             print("pose_transformed:", entry["pose_transformed"])
#
#         if processed_count >= 6999:
#             break
#     # Save the new JSON structure
#     new_json_path = json_path.replace(".json", "_updated.json")
#     with open(new_json_path, "w") as file:
#         json.dump(new_data, file)
#
#     return new_json_path


def test_model3(json_path, model_path, print_frequency=50, max_photos=None):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = model3.FusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define the transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and extract data from the JSON file
    with open(json_path, "r") as file:
        data = json.load(file)

    # Limit the number of photos if max_photos is specified
    if max_photos is not None:
        data = dict(list(data.items())[:max_photos])

    # Create new JSON structure (as a dictionary to preserve the original structure)
    new_data = {}

    # Counter for processed entries
    processed_count = 0

    for key, entry in data.items():
        # Extract numerical data
        numerical_data = torch.tensor(entry["pose_transformed"][0][:3], dtype=torch.float32).to(device)

        # Load and transform the image
        image_path = entry["frame"]
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            numerical_output = model(image_tensor, numerical_data.unsqueeze(0))

        # Update the JSON entry
        entry["pose_transformed_old"] = entry["pose_transformed"]
        entry["pose_transformed"] = numerical_output.cpu().squeeze(
            0).tolist()  # Move data back to CPU for JSON serialization

        # Update the new_data dictionary
        new_data[key] = entry

        # Increment the processed counter
        processed_count += 1

        # Print processed count at specified frequency
        if processed_count % print_frequency == 0:
            print(f"Processed {processed_count} entries.")

            print("pose_transformed_old:", entry["pose_transformed_old"][0])
            print("pose_transformed:", entry["pose_transformed"])

    # Save the new JSON structure with pretty formatting
    new_json_path = json_path.replace(".json", "_updated.json")
    with open(new_json_path, "w") as file:
        json.dump(new_data, file, indent=3)


    print('Finished TEST!')
    return new_json_path

#model path, path to trained model weights
model_path = r"C:\Users\roblab20\Documents\yardenalon\Pre-Process MM\train history\model_epoch_18_best.pth"
#JSON path to generated images with Simulation numerical data
JSON_path = r'C:\Users\roblab20\Documents\yardenalon\Compression\datasets\Finger\json_data\SRC_Gin_3_transformed.json'


new_json_path = test_model3(JSON_path, model_path, max_photos=7000)

