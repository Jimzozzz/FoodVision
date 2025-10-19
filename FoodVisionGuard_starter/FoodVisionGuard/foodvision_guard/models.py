import torch
import torch.nn as nn
import timm

class Classifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool=True, dropout: float=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

class MultiHeadFish(nn.Module):
    def __init__(self, backbone_name: str, num_eye: int, num_gill: int, num_skin: int, pretrained: bool=True, dropout: float=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        def head(nc): 
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, nc))
        self.eye_head = head(num_eye)
        self.gill_head = head(num_gill)
        self.skin_head = head(num_skin)

    def forward(self, x):
        feats = self.backbone(x)
        return {
            "eye": self.eye_head(feats),
            "gill": self.gill_head(feats),
            "skin": self.skin_head(feats)
        }
