from MM_options_fusion import parse_args
import json
import torch

args = parse_args()

def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean

def load_data_image(Json_path):
    # Load the data
    image_paths = []
    coordinates = []

    # Load the JSON file
    with open(Json_path, "r") as file:
        data = json.load(file)

    # Extract image paths and coordinates
    for entry in data.values():
        image_paths.append(entry["frame"])
        coordinates.append(entry["pose_transformed"][0])

    return image_paths, coordinates

def load_data_sim(Json_path):
    # Load the data
    coordinates = []

    # Load the JSON file
    with open(Json_path, "r") as file:
        data = json.load(file)

    # Extract image paths and coordinates
    for entry in data.values():
        coordinates.append(entry["pose_transformed"][0])

    return coordinates

def json_to_array(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    coordinates = [entry["pose_transformed"][0] for entry in data.values()]
    return coordinates

def load_data_fc(Json_real_path = args.train_real_JSON, Json_sim_path = args.train_sim_JSON):
    # Load training data
    train_sim_data = json_to_array(Json_real_path)
    # train_real_data = json_to_array(Json_sim_path)

    # Convert the data to PyTorch tensors
    train_sim_tensor = torch.tensor(train_sim_data, dtype=torch.float32)
    # train_real_tensor = torch.tensor(train_real_data, dtype=torch.float32)

    #normalizing
    # Normalize the data
    mean_sim, std_sim = train_sim_tensor.mean(0), train_sim_tensor.std(0)
    # mean_real, std_real = train_real_tensor.mean(0), train_real_tensor.std(0)

    train_sim_tensor = normalize(train_sim_tensor, mean_sim, std_sim)
    # train_real_tensor = normalize(train_real_tensor, mean_real, std_real)

    return
    

def lr_schedule(epoch):
    initial_learning_rate = args.lr
    decay_epochs = args.decay_epochs
    total_epochs = args.num_epochs
    if epoch > decay_epochs:
        new_lr = initial_learning_rate * (1.0 - (epoch - decay_epochs) / (total_epochs - decay_epochs))
        print(new_lr)
        return new_lr 
    else:
        return initial_learning_rate
