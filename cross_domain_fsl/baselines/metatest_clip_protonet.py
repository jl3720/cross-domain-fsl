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
    "--vision_model", type=str, default="ViT-B/32", help="CLIP Vision backbone"
)
parser.add_argument("--n_support", type=int, default=5)
parser.add_argument("--n_query", type=int, default=15)
parser.add_argument("--n_way", type=int, default=5)

args = parser.parse_args()


class CLIP(nn.Module):
    def __init__(self, vision_model="ViT-B/32"):
        super(CLIP, self).__init__()
        model, preprocess = clip.load(vision_model, DEVICE)

        self.transform = preprocess

        # Freeze CLIP backbone
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.final_feat_dim = CLIP_DIM_MAPPING[vision_model]

    def forward(self, x):
        # x = self.transform(x)  # SetDataLoader yields transformed images
        # TODO: Note that clip usually normalises images
        x = self.model.encode_image(x)
        return x


def clip_wrapper(backbone: nn.Module, **kwargs):
    """Wrap CLIP backbone for a ProtoNet model"""
    return backbone


def meta_test(model, datamgr):
    test_dataloader = datamgr.get_data_loader(aug=False)

    n_classes = len(datamgr.dataset.sub_meta)  # Bit hacky
    n_episode = datamgr.n_episode
    IS_FEATURE = False
    acc_all = []
    class_acc_all = torch.zeros(n_classes, n_episode)
    # Mask for classes that appear in episode
    classes_mask = torch.zeros(n_classes, n_episode)

    model.eval()
    for i, (x, y) in enumerate(test_dataloader):
        # x, y = next(iter(test_dataloader))
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        classes = y[:, 0]
        classes_mask[classes, i] = 1  # Update mask

        y_query = y[:, model.n_support :].contiguous().view(-1)
        # x_var = torch.autograd.Variable(x)
        # 5x5 sets for testing
        scores = model.set_forward(x, IS_FEATURE)

        # raw_preds are indices of selected subset of classes
        raw_preds = nn.functional.softmax(scores, dim=1).argmax(1)
        preds = classes[raw_preds]  # map to actual class indices
        acc = torch.sum(preds == y_query) / len(y_query)
        acc_all.append(acc)
        print(f"Batch average accuracy: {acc.item()}")

        preds_by_class = preds.contiguous().view(model.n_way, model.n_query)
        y_query_by_class = y_query.contiguous().view(model.n_way, model.n_query)

        acc_by_class = (
            torch.sum(preds_by_class == y_query_by_class, dim=1) / model.n_query
        )
        for j, acc in enumerate(acc_by_class):
            print(f"Class {classes[j]} accuracy: {acc.item()}")

        for j, cl in enumerate(classes):
            class_acc_all[cl, i] = acc_by_class[j]

    with torch.no_grad():
        acc_all = torch.tensor(acc_all)
        acc_mean = torch.mean(acc_all)
        acc_std = torch.std(acc_all)

        # Use mask to ignore zero entries
        class_acc_mean = torch.sum(class_acc_all, dim=1) / classes_mask.sum(dim=1)
        class_acc_all[classes_mask == 0] = torch.nan  # Mask out unused classes
        class_acc_std = np.nanstd(class_acc_all.cpu().numpy(), axis=1)

    print("Average test Acc = %4.2f%% +- %4.2f%%" % (acc_mean * 100, acc_std * 100))
    for i in range(n_classes):  # Assumes classes are 0 - (N-1)
        print(
            "Class %d Test Acc = %4.2f%% +- %4.2f%%"
            % (i, class_acc_mean[i] * 100, class_acc_std[i] * 100)
        )
    return acc_mean, acc_std


def main(args: argparse.Namespace):

    clip_backbone = CLIP(vision_model=args.vision_model).to(DEVICE)
    transform = clip_backbone.transform
    model_func = partial(clip_wrapper, backbone=clip_backbone)
    # print(model_func)

    mgr_cls = MANAGER_DICT[args.dataset]
    datamgr = mgr_cls(
        IMAGE_SIZE,
        transform,
        n_way=args.n_way,
        n_query=args.n_query,
        n_support=args.n_support,
    )

    protonet = ProtoNet(
        model_func, n_way=args.n_way, n_support=args.n_support, tf_path=None
    ).to(DEVICE)

    acc_mean, acc_std = meta_test(protonet, datamgr)


if __name__ == "__main__":
    print(args)
    main(args)
