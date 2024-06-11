# Contains foundation model wrappers for use with `ProtoNet` classes.
# A `dict` is defined at the bottom for easy loading.

import torch
import torch.nn as nn
import torchvision.transforms as T
import clip

from abc import abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)  # Force square images
CLIP_DIM_MAPPING = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
DINOV2_DIM_MAPPING = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class FoundationModel(nn.Module):
    """Abstract class for Foundation Models.
    Standard interface for feature module in `MetaTemplate` or `ProtoNet`.
    """

    @abstractmethod
    def __init__(self, vision_variant: str):
        """Abstract __init__ method for FoundationModels.
        Required args:
            vision_variant (str): Vision backbone variant, e.g. 'ViT-B/32'.
        """
        super(FoundationModel, self).__init__()

        # Must set these attributes in child class
        self.model = None
        self.transform = None
        self.final_feat_dim = None

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor. Expected (B, C, H, W).
                              C = 3, H = W = 224.
        Returns:
            torch.Tensor: Feature tensor. Shape (B, D).
        """
        pass


class CLIP(FoundationModel):
    def __init__(self, vision_variant="ViT-B/32"):
        super(CLIP, self).__init__(vision_variant)
        model, preprocess = clip.load(vision_variant, DEVICE)

        self.transform = preprocess

        # Freeze CLIP backbone
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.final_feat_dim = CLIP_DIM_MAPPING[vision_variant]

    def forward(self, x):
        # x = self.transform(x)  # SetDataLoader yields transformed images
        # TODO: Note that clip usually normalises images
        x = self.model.encode_image(x)
        return x


class DINOv2(FoundationModel):
    def __init__(self, vision_variant="dinov2_vits14"):
        super(DINOv2, self).__init__(vision_variant)
        model = torch.hub.load("facebookresearch/dinov2", vision_variant)

        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        # Hard code transform for DINOv2
        self.transform = T.Compose(
            [
                T.Resize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

        # Freeze CLIP backbone
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.final_feat_dim = DINOV2_DIM_MAPPING[vision_variant]

    def forward(self, x):
        # x = self.transform(x)  # SetDataLoader yields transformed images
        # TODO: Note that clip usually normalises images
        x = self.model(x)
        return x


FOUNDATION_MODELS: dict[str, FoundationModel] = {"CLIP": CLIP, "DINOv2": DINOv2}
