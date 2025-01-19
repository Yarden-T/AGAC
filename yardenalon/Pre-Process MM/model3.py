import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super(EfficientNetEncoder, self).__init__()
        self.model = models.efficientnet_v2_l(weights = None)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the classification head

    def forward(self, x):
        x = self.model(x)
        return x



class FusionModel(nn.Module):
    def __init__(self, num_numerical_features = 3, hidden_units_image=[2048, 512], hidden_units_numerical=[128, 64],
                 fusion_hidden_units=[512, 128]):
        super(FusionModel, self).__init__()

        # EfficientNetV2  for Image processing
        self.encoder = EfficientNetEncoder()

        # Fully connected layers for image
        self.fc_image = nn.Sequential(
            nn.Linear(1280 * 1 * 1, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU()
        )

        # Fully Connected Network for Numerical Data
        self.fc_numerical = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_units_numerical[0]),
            nn.ReLU(),
            nn.Linear(hidden_units_numerical[0], hidden_units_numerical[1]),
            nn.ReLU()
        )

        # Fusion Network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_units_image[1] + hidden_units_numerical[1], fusion_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(fusion_hidden_units[0], fusion_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(fusion_hidden_units[1], 3)  # Output size of 3
        )

    def forward(self, img, numerical_data):
        img_features = self.encoder(img)
        img_features = img_features.view(-1, 1280)
        img_processed = self.fc_image(img_features)
        numerical_processed = self.fc_numerical(numerical_data)

        # Concatenate image and numerical processed data
        combined_data = torch.cat((img_processed, numerical_processed), dim=1)

        return self.fusion(combined_data)

