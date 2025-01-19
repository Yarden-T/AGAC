import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(VGG16, self).__init__()
        # Define the convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )




    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x



class FusionModel(nn.Module):
    def __init__(self, num_numerical_features = 3, hidden_units_image=[4096, 1024], hidden_units_numerical=[128, 64],
                 fusion_hidden_units=[512, 128]):
        super(FusionModel, self).__init__()

        # VGG16 for Image processing
        self.vgg16 = VGG16(input_channels=3, num_classes=1000)

        # Fully connected layers for image
        self.fc_image = nn.Sequential(
            nn.Linear(512 * 15 * 15, hidden_units_image[0]),  # 8x8 is the feature map size for 256x256 input images
            nn.ReLU(),
            nn.Linear(hidden_units_image[0], hidden_units_image[1]),
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
        img_features = self.vgg16(img)
        img_processed = self.fc_image(img_features)
        numerical_processed = self.fc_numerical(numerical_data)

        # Concatenate image and numerical processed data
        combined_data = torch.cat((img_processed, numerical_processed), dim=1)

        return self.fusion(combined_data)

