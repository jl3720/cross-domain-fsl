# Directly take CLIP backbone, compute prototypes and evalate on novel classes

import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import clip

from functools import partial

# from cross_domain_fsl.options import parse_args

# from cross_domain_fsl.methods.backbone_multiblock import model_dict
from cross_domain_fsl.methods.protonet import ProtoNet

# from cross_domain_fsl.data.datamgr import SetDataManager
from cross_domain_fsl.data.managers import MANAGER_DICT

# the finetuning is very sensitive to lr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLIP_DIM_MAPPING = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="EuroSAT")
parser.add_argument(
    "--backbone", type=str, default="ViT-B/32", help="CLIP Vision backbone"
)
parser.add_argument("--n_support", type=int, default=5)
parser.add_argument("--n_query", type=int, default=15)
parser.add_argument("--n_way", type=int, default=5)

args = parser.parse_args()


class CLIP(nn.Module):
    def __init__(self, backbone="ViT-B/32"):
        super(CLIP, self).__init__()
        model, preprocess = clip.load(backbone, DEVICE)

        self.transform = preprocess

        # Freeze CLIP backbone
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.final_feat_dim = CLIP_DIM_MAPPING[backbone]

    def forward(self, x):
        # x = self.transform(x)  # SetDataLoader yields transformed images
        # TODO: Note that clip usually normalises images
        x = self.model.encode_image(x)
        return x


def clip_wrapper(backbone: str = "ViT-B/32", **kwargs):
    return CLIP(backbone)


def meta_test(model, test_dataloader):
    acc_all = []
    IS_FEATURE = False
    # for i, (x, y) in enumerate(test_dataloader):
    x, y = next(iter(test_dataloader))
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    y_query = y[:, model.n_support :].contiguous().view(-1)
    # x_var = torch.autograd.Variable(x)
    # 5x5 sets for testing
    scores = model.set_forward(x, IS_FEATURE)

    preds = nn.functional.softmax(scores, dim=1).argmax(1)
    print(f"Preds: {preds.shape}, y: {y.shape}, y_query: {y_query.shape}")
    acc = torch.sum(preds == y_query) / len(y_query)
    acc_all.append(acc)
    print(acc_all)

    with torch.no_grad():
        acc_all = torch.tensor(acc_all)
        acc_mean = torch.mean(acc_all)
        acc_std = torch.std(acc_all)
    print("Test Acc = %4.2f%% +- %4.2f%%" % (acc_mean * 100, acc_std * 100))
    return acc_mean, acc_std


def main(args: argparse.Namespace):
    mgr_cls = MANAGER_DICT[args.dataset]
    datamgr = mgr_cls(
        IMAGE_SIZE, n_way=args.n_way, n_query=args.n_query, n_support=args.n_support
    )

    test_loader = datamgr.get_data_loader(aug=False)

    x, y = next(iter(test_loader))
    print(x.size(), y.size())
    print(y)

    model_func = partial(clip_wrapper, backbone=args.backbone)
    print(model_func)
    protonet = ProtoNet(
        model_func, n_way=args.n_way, n_support=args.n_support, tf_path=None
    ).to(DEVICE)

    acc_mean, acc_std = meta_test(protonet, test_loader)


if __name__ == "__main__":
    print(args)
    main(args)
