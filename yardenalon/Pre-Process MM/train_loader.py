import torch
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, real_json_path, sim_json_path, transform=None):
        self.real_data = self._load_data(real_json_path)
        self.sim_data = self._load_data(sim_json_path)
        self.transform = transform

    def _load_data(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        # Assuming real_data and sim_data have the same length
        return len(self.real_data)

    def __getitem__(self, idx):
        real_item = self.real_data[idx]
        sim_item = self.sim_data[idx]
        
        real_image = Image.open(real_item['frame'])
        sim_pose = sim_item['pose_transformed']
        real_pose = real_item['pose_transformed']
        
        if self.transform:
            real_image = self.transform(real_image)
        
        return {'real_image': real_image, 'real_pose': real_pose, 'sim_pose': sim_pose}

def get_train_loader(real_json_path, sim_json_path, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to desired size
        transforms.ToTensor(),           # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    
    dataset = CustomDataset(real_json_path, sim_json_path, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader

if __name__ == '__main__':
    real_json_path = 'path_to_real_json.json'
    sim_json_path = 'path_to_sim_json.json'
    train_loader = get_train_loader(real_json_path, sim_json_path)
    
    for batch in train_loader:
        real_images = batch['real_image']
        real_poses = batch['real_pose']
        sim_poses = batch['sim_pose']
        # Train your model using the data in this batch
        break  # For demonstration, break after processing one batch
