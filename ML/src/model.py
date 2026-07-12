from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch

class BodyFatMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes = [64, 32], output_size=1, dropout_rate=0.2):
        super(BodyFatMLP, self).__init__()
        
        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(previous_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class BodyM_EfficientNet(nn.Module):
    def __init__(self, in_channels=2, dropout_rate=0.2, output_size=14):
        super(BodyM_EfficientNet, self).__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        original_layer = self.model.features[0][0]

        self.model.features[0][0] = nn.Conv2d(
            in_channels, 
            original_layer.out_channels, 
            kernel_size=original_layer.kernel_size, 
            stride=original_layer.stride, 
            padding=original_layer.padding, 
            bias=original_layer.bias is not None
            )

        data = original_layer.weight.data

        self.model.features[0][0].weight.data = data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)

        gender_embedding_size = 8

        self.gender_embedding = nn.Linear(1, gender_embedding_size)

        in_features = self.model.classifier[1].in_features 
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(in_features + gender_embedding_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )

    def forward(self, x, gender):
        features = self.model.features(x)
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)

        gender_embedding = self.gender_embedding(gender.unsqueeze(1))

        features = torch.cat((features, gender_embedding), dim=1)

        return self.model.classifier[1](features)