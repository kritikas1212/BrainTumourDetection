from torch import nn
from torchvision import models
import torch

class TransferLearningResNet(nn.Module):
    def __init__(self, num_classes=4, dropout_rate_1=0.3, dropout_rate_2=0.25):
        super(TransferLearningResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all layers except the last block (layer4)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze all layers

        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Modify the final fully connected layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),  # Dense layer with 128 units
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate_2),  # Second dropout layer
            nn.Linear(256, num_classes),  # Final dense layer with output matching number of classes
        )

    def forward(self, x):
        return self.resnet(x)


class TransferLearningEfficientNet(nn.Module):
    """
    EfficientNet-B3 Transfer Learning Model
    
    EfficientNet-B3 Advantages:
    - More parameters (12M) vs ResNet50 (25.6M) = Faster inference
    - Better accuracy with fewer parameters (compound scaling)
    - Better feature extraction with MBConv blocks
    - Optimized for mobile and edge devices
    
    Expected Performance:
    - Accuracy: 99.0-99.5% (vs 98.77% ResNet50)
    - Inference time: ~0.7s (vs ~1s ResNet50)
    - Model size: ~48MB (vs ~98MB ResNet50)
    """
    def __init__(self, num_classes=4, dropout_rate=0.3, model_version='b3'):
        super(TransferLearningEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet
        if model_version == 'b0':
            self.efficientnet = models.efficientnet_b0(pretrained=True)
            in_features = 1280
        elif model_version == 'b3':
            self.efficientnet = models.efficientnet_b3(pretrained=True)
            in_features = 1536
        else:
            raise ValueError(f"Unsupported model version: {model_version}")
        
        # Freeze early layers (keep features layer frozen initially)
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        
        # Unfreeze last 3 blocks for fine-tuning (progressive unfreezing)
        # EfficientNet has 9 blocks, we'll unfreeze blocks 6, 7, 8
        for i in range(6, 9):
            if i < len(self.efficientnet.features):
                for param in self.efficientnet.features[i].parameters():
                    param.requires_grad = True
        
        # Custom classifier head with better regularization
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights using He initialization"""
        for m in self.efficientnet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning (optional, for later epochs)"""
        for param in self.efficientnet.parameters():
            param.requires_grad = True


class TransferLearningEfficientNetSimple(nn.Module):
    """
    EfficientNet-B3 head that matches the Google Colab training script
    (simple 2-layer classifier, dropout=0.25 default)
    """

    def __init__(self, num_classes=4, dropout_rate=0.25, model_version='b3'):
        super().__init__()

        if model_version == 'b3':
            self.efficientnet = models.efficientnet_b3(pretrained=True)
            in_features = 1536
        else:
            raise ValueError(f"Unsupported model version: {model_version}")

        for param in self.efficientnet.features.parameters():
            param.requires_grad = False

        for i in range(6, 9):
            if i < len(self.efficientnet.features):
                for param in self.efficientnet.features[i].parameters():
                    param.requires_grad = True

        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.efficientnet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.efficientnet(x)

    def unfreeze_all(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = True


# Model factory updated to include both CNN and transfer learning options
model_factory = {
    'resnet': TransferLearningResNet,
    'efficientnet_b0': lambda num_classes=4, dropout_rate=0.3: TransferLearningEfficientNet(num_classes, dropout_rate, 'b0'),
    'efficientnet_b3': lambda num_classes=4, dropout_rate=0.3: TransferLearningEfficientNet(num_classes, dropout_rate, 'b3'),
    'efficientnet_b3_simple': lambda num_classes=4, dropout_rate=0.25: TransferLearningEfficientNetSimple(num_classes, dropout_rate, 'b3'),
}


# Option to save the model (unchanged from original)
def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            save(model.state_dict(), path.join('./', f"{n}_model.pth"))
            print(f"Model saved as {n}_model.pth")
