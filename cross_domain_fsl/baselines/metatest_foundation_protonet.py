# Directly take CLIP backbone, compute prototypes and evalate on novel classes

import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import clip

from functools import partial

# from cross_domain_fsl.options import parse_args

# from cross_domain_fsl.methods.backbone_multiblock import model_dict
from cross_domain_fsl.methods.protonet import ProtoNet
from cross_domain_fsl.methods.foundation_models import FOUNDATION_MODELS
from cross_domain_fsl.data.managers import MANAGER_DICT

# the finetuning is very sensitive to lr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)  # Force square images
CLIP_DIM_MAPPING = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="EuroSAT",
    help="Valid options: {'EuroSAT', 'CropDisease', 'ISIC', 'ChestX'}",
)
parser.add_argument(
    "--foundation_model",
    type=str,
    default="CLIP",
    help="Foundation model to adapt.\
        Options (Case sensitive): {'CLIP', 'DINOv2', 'Vim'}",
)
parser.add_argument(
    "--vision_variant", type=str, default="ViT-B/32", help="Vision backbone variant"
)
parser.add_argument("--n_support", type=int, default=5)
parser.add_argument("--n_query", type=int, default=16)
parser.add_argument("--n_way", type=int, default=5)
parser.add_argument("--n_episode", type=int, default=1000)

args = parser.parse_args()


def foundation_model_wrapper(foundation_model: nn.Module, **kwargs):
    """Wrap foundation model backbone for a `ProtoNet` module"""
    return foundation_model


def meta_test(model, datamgr):
    t0 = time.time()
    test_dataloader = datamgr.get_data_loader(aug=False)
    t1 = time.time()
    print(f"Time to init Dataloader: {t1 - t0}")

    n_classes = len(datamgr.dataset.sub_meta)  # Bit hacky
    n_episode = datamgr.n_episode
    IS_FEATURE = False
    acc_all = []
    class_acc_all = torch.zeros(n_classes, n_episode)
    # Mask for classes that appear in episode
    classes_mask = torch.zeros(n_classes, n_episode)

    time_list = []
    model.eval()
    for i, (x, y) in enumerate(test_dataloader):
        # x, y = next(iter(test_dataloader))
        t0 = time.time()
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
        time_list.append(time.time() - t0)
        acc = torch.sum(preds == y_query) / len(y_query)
        acc_all.append(acc)
        print(f"Batch ({i}/{n_episode}) average accuracy: {acc.item()}")

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
    time_mean = np.mean(time_list)
    print(f"Average time per episode: {time_mean}")
    return acc_mean, acc_std


def main(args: argparse.Namespace):

    foundation_model = FOUNDATION_MODELS[args.foundation_model](args.vision_variant)
    transform = foundation_model.transform
    model_func = partial(foundation_model_wrapper, foundation_model=foundation_model)
    # print(model_func)

    mgr_cls = MANAGER_DICT[args.dataset]
    datamgr = mgr_cls(
        IMAGE_SIZE,
        transform,
        n_way=args.n_way,
        n_query=args.n_query,
        n_support=args.n_support,
        n_episode=args.n_episode,
    )

    protonet = ProtoNet(
        model_func, n_way=args.n_way, n_support=args.n_support, tf_path=None
    ).to(DEVICE)

    acc_mean, acc_std = meta_test(protonet, datamgr)


if __name__ == "__main__":
    print(args)
    main(args)
