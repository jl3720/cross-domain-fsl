# Contains foundation model wrappers for use with `ProtoNet` classes.
# A `dict` is defined at the bottom for easy loading.

import torch
import torch.nn as nn
import torchvision.transforms as T
import clip

from abc import abstractmethod

from vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_tiny
from vim.models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_small

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)  # Force square images

CLIP_DIM_MAPPING = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
DINOV2_DIM_MAPPING = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
VIM_DIM_MAPPING = {"vim_tiny": 192, "vim_small": 384}


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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


class FoundationFC(nn.Module):
    """Abstract class for Foundation Models with Fully Connected Adapter."""

    @abstractmethod
    def __init__(self, vision_variant: str, n_way: int):
        """Abstract __init__ method for FoundationModels.
        Required args:
            vision_variant (str): Vision backbone variant, e.g. 'ViT-B/32'.
            n_way (int): Number of classes for linear probe.
        """
        super(FoundationFC, self).__init__()

        # Must set these attributes in child class
        self.transform = None

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor. Expected (B, C, H, W).
                              C = 3, H = W = 224.
        Returns:
            torch.Tensor: Class logits. Shape (B, n_qway).
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


class CLIP_FC(FoundationFC):
    """CLIP model with a fully connected layer for fine-tuning.

    I.e. Linear probe.
    """

    def __init__(self, vision_variant="ViT-B/32", n_way=5):
        super(CLIP_FC, self).__init__(vision_variant, n_way)
        model, preprocess = clip.load(vision_variant, DEVICE)

        self.transform = preprocess

        # Freeze CLIP backbone
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.fc = nn.Linear(CLIP_DIM_MAPPING[vision_variant], n_way).half()
        self.norm = nn.BatchNorm1d(n_way).half()

        self.final_feat_dim = CLIP_DIM_MAPPING[vision_variant]

    def forward(self, x):
        # x = self.transform(x)  # SetDataLoader yields transformed images
        # TODO: Experiment with ReLU and/or BatchNorm
        self.model.eval()
        x = self.model.encode_image(x)
        x = self.fc(x)
        x = self.norm(x)
        return x


class FC(nn.Module):
    """Fully connected layer for fine-tuning."""

    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return x


class DINOv2(FoundationModel):
    def __init__(self, vision_variant="dinov2_vits14"):
        super(DINOv2, self).__init__(vision_variant)
        model = torch.hub.load("facebookresearch/dinov2", vision_variant)

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

class Vim(FoundationModel):
    def __init__(self, vision_variant: str="vim_tiny"):
        super().__init__(vision_variant)
        vision_variant = vision_variant.lower()
        if vision_variant == "vim_tiny":
            model = vim_tiny(pretrained=True)
        elif vision_variant == "vim_small":
            model = vim_small(pretrained=True)
        else:
            raise ValueError(f"Invalid vision_variant: {vision_variant}\
                             . Must be one of ['vim_tiny', 'vim_small']")
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.model = model
        self.transform = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)  # ImageNet defaults
        ])
        self.final_feat_dim = VIM_DIM_MAPPING[vision_variant]  # TODO: hard code for now
    
    def forward(self, x):
        return self.model(x, return_features=True)


FOUNDATION_MODELS: dict[str, FoundationModel] = {"CLIP": CLIP, "DINOv2": DINOv2, "Vim": Vim}
