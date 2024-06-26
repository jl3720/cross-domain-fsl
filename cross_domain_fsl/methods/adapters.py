import torch.nn as nn

# TODO: Create Base class and interface


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


class Adapter(nn.Module):
    """Inspired by CLIP Adapter module.
    Forces features through a bottleneck, and then to logits.
    Original maps features back to original dimension, and used with cosine sim.
    Adapted from https://github.com/gaopengcuhk/CLIP-Adapter/blob/08d07f8b2ecafc6f1479fe636b26d464d7a5574e/clip_adapter.py#L55
    """

    RATIO = 0.2

    def __init__(self, in_dim, out_dim, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Forward pass through adapter.

        Args:
            x: Foundation model image features (B, in_dim).

        Will be used as follows:
        ```
        image_features = clip_model.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        ```
        """
        bottlenecked = self.fc(x)
        x = self.RATIO * bottlenecked + (1 - self.RATIO) * x
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.head(x)
        return x


ADAPTERS = {"FC": FC, "Adapter": Adapter}
