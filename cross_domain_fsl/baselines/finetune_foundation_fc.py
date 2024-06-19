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
from cross_domain_fsl.methods.foundation_models import FOUNDATION_FC_MODELS
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
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--n_support", type=int, default=5)
parser.add_argument("--n_query", type=int, default=16)
parser.add_argument("--n_way", type=int, default=5)
parser.add_argument("--n_episode", type=int, default=1000)

args = parser.parse_args()


def meta_test(model: nn.Module, datamgr, args: argparse.Namespace):
    """Fine tune on support set, evaluate on query set.

    Args:
        model: Foundation model + trainable linear probe. `forward()` should
            accept input of shape (B, 3, 224, 224) and return (B, n_classes).
    """
    t0 = time.time()
    test_dataloader = datamgr.get_data_loader(aug=False)
    t1 = time.time()
    print(f"Time to init Dataloader: {t1 - t0}")

    n_classes = len(datamgr.dataset.sub_meta)  # Bit hacky
    n_episode = datamgr.n_episode

    support_acc_all = []
    acc_all = []
    class_acc_all = torch.zeros(n_classes, n_episode)
    # Mask for classes that appear in episode
    classes_mask = torch.zeros(n_classes, n_episode)

    criterion = nn.CrossEntropyLoss()

    time_list = []
    # model.eval()
    for i, (x, y) in enumerate(test_dataloader):
        # x, y = next(iter(test_dataloader))
        t0 = time.time()
        x = x.to(DEVICE)
        y = y.to(DEVICE)  # global label ids

        n_way = args.n_way
        n_support = args.n_support
        n_query = args.n_query

        # TODO: Create augmented support set.
        x_support = x[:, : args.n_support, :, :, :].contiguous()
        x_query = x[:, args.n_support :, :, :, :].contiguous()

        # print(f"x_support shape: {x_support.shape}")
        # print(f"x_query shape: {x_query.shape}")

        classes = y[:, 0]
        classes_mask[classes, i] = 1  # Update mask

        # Train on support set

        # Each episode, we need to reinitialize the model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Forward pass
        # x: (n_way * n_support, 3, 224, 224) -> support_scores: (n_way * n_support, n_way)
        support_scores = model(x_support.view(n_way * n_support, *x_support.shape[2:]))
        # y: (n_way) -> (n_way, n_support) -> (n_way * n_support)
        # Use local label ids for backprop
        y_support = (
            torch.arange(n_way)
            .expand(n_way, n_support)
            .contiguous()
            .view(-1)
            .to(DEVICE)
        )
        with torch.no_grad():
            support_preds = nn.functional.softmax(support_scores, dim=-1).argmax(-1)
            support_acc = torch.sum(support_preds == y_support) / len(y_support)
            support_acc_all.append(support_acc)
            print(
                f"Batch ({i}/{n_episode}) average support accuracy: {support_acc.item()}"
            )  # Before any finetuning

        # Backward pass. TODO: Multiple epochs of fine tuning on support set?
        loss_val = criterion(support_scores, y_support)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Evaluate on query set
        y_query = y[:, n_support:].contiguous().view(-1)

        query_scores = model(x_query.view(n_way * n_query, *x_query.shape[2:]))

        # raw_preds are indices of selected subset of classes
        raw_preds = nn.functional.softmax(query_scores, dim=-1).argmax(-1)
        preds = classes[raw_preds]  # map to actual class indices
        time_list.append(time.time() - t0)
        acc = torch.sum(preds == y_query) / len(y_query)
        acc_all.append(acc)
        print(f"Batch ({i}/{n_episode}) average query accuracy: {acc.item()}")

        preds_by_class = preds.contiguous().view(n_way, n_query)
        y_query_by_class = y_query.contiguous().view(n_way, n_query)

        acc_by_class = torch.sum(preds_by_class == y_query_by_class, dim=1) / n_query
        for j, acc in enumerate(acc_by_class):
            print(f"Class {classes[j]} accuracy: {acc.item()}")

        for j, cl in enumerate(classes):
            class_acc_all[cl, i] = acc_by_class[j]

    with torch.no_grad():
        support_acc_all = torch.tensor(support_acc_all)
        support_acc_mean = torch.mean(support_acc_all)
        support_acc_std = torch.std(support_acc_all)
        acc_all = torch.tensor(acc_all)
        acc_mean = torch.mean(acc_all)
        acc_std = torch.std(acc_all)

        # Use mask to ignore zero entries
        class_acc_mean = torch.sum(class_acc_all, dim=1) / classes_mask.sum(dim=1)
        class_acc_all[classes_mask == 0] = torch.nan  # Mask out unused classes
        class_acc_std = np.nanstd(class_acc_all.cpu().numpy(), axis=1)

    print(
        "Average support Acc = %4.2f%% +- %4.2f%%"
        % (support_acc_mean * 100, support_acc_std * 100)
    )
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

    foundation_fc = FOUNDATION_FC_MODELS[args.foundation_model](
        args.vision_variant, args.n_way
    ).to(DEVICE)
    transform = foundation_fc.transform
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

    acc_mean, acc_std = meta_test(foundation_fc, datamgr, args)


if __name__ == "__main__":
    print(args)
    main(args)
