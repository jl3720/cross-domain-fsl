import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import math
import torch.nn.functional as F
from torch.nn import Parameter

from cross_domain_fsl.methods.text_encoder import TextEncoder

clip_dim_mapping = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}
class_num_mapping = {"PACS": 7, "OfficeHome": 65, "VLCS": 5, "DomainNet": 345}


# Define a custom dataset that generates K*N style-content features and class labels.
# Replace this with your own data loading code.
class StyleContentDataset(torch.utils.data.Dataset):
    def __init__(self, K, N, LEN_prompt, DIM):
        # K*N style-content features 80*N*77*512
        # self.input_data_1 = torch.randn(K * N, LEN_prompt, DIM).tolist()
        # self.input_data_1 = torch.randn(K * N, LEN_prompt, DIM).float()
        self.input_data_1 = torch.randn(
            size=(K * N, LEN_prompt, DIM), dtype=torch.float16
        )
        self.input_data_1 = [
            self.input_data_1[i] for i in range(self.input_data_1.shape[0])
        ]
        # self.input_data_2 = torch.randn(K * N, LEN_prompt).tolist()
        # self.input_data_2 = torch.randn(K * N, LEN_prompt).float()
        self.input_data_2 = torch.randn(size=(K * N, LEN_prompt), dtype=torch.float16)
        self.input_data_2 = [
            self.input_data_2[i] for i in range(self.input_data_2.shape[0])
        ]
        # Class labels for each feature (N classes)
        self.labels = torch.randint(0, N, (K * N,)).tolist()
        # Generate or load your style-content features and labels here.

        # print("Type of Style-Content Features:", type(self.input_data_1), len(self.input_data_1))
        # print("Type of Style-Content Features 2:", type(self.input_data_2), len(self.input_data_2))
        # print("Type of Class Labels:", type(self.labels), len(self.labels), self.labels[0])

    def __len__(self):
        return len(self.input_data_1)

    def __getitem__(self, idx):
        data1 = self.input_data_1[idx]
        data2 = self.input_data_2[idx]
        label = self.labels[idx]
        return data1, data2, label


class StyleContentDatasetV2(torch.utils.data.Dataset):
    def __init__(self, input_data_1, input_data_2):
        self.input_data_1 = input_data_1
        self.input_data_2 = input_data_2
        K = 80
        N = input_data_1.size()[0] // K
        self.labels = [id % N for id in range(K * N)]
        print("label:", self.labels)

    def __len__(self):
        return len(self.input_data_1)

    def __getitem__(self, idx):
        data1 = self.input_data_1[idx]
        data2 = self.input_data_2[idx]
        label = self.labels[idx]
        return data1, data2, label


# Define a custom linear classifier with L2 norm
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x, L2_flag=True):
        # ('x:', x.size())

        # Apply L2 normalization
        if L2_flag:
            x = F.normalize(x, p=2, dim=1)

        # forward fc
        x = self.fc(x)

        return x


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.weight = Parameter(
            torch.FloatTensor(out_features, in_features).to(torch.float16)
        )
        # self.weight = self.weight.cuda()
        # self.weight = Parameter(torch.FloatTensor(size=(out_features, in_features), dtype=float16))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def forward2(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine


def train_classifier(input_data_1, input_data_2, clip_model_name, dataset_name, args):
    # Initialize your dataset, text encoder, and linear classifier.
    K = 80  # Number of styles
    N = class_num_mapping[dataset_name]  # Number of classes
    LEN_prompt = 77
    DIM = clip_dim_mapping[clip_model_name]

    # dataset = StyleContentDataset(K, N, LEN_prompt, DIM)
    dataset = StyleContentDatasetV2(input_data_1, input_data_2)

    # Training parameters.
    num_epochs = 50
    batch_size = 128

    # Define a data loader with the specified batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # define CLIP model
    device = "cuda"
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    text_encoder = TextEncoder(clip_model)

    # define classifier with L2 norm
    # classifier = LinearClassifier(input_dim=512, num_classes=N)  # 512 is the dimension of CLIP's text encoder output.
    easy_margin = False
    classifier = ArcMarginProduct(DIM, N, s=5, m=0.5, easy_margin=easy_margin).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer with specified parameters.
    optimizer = optim.SGD(classifier.parameters(), lr=0.005, momentum=0.9)

    # Training loop.
    for epoch in range(num_epochs):
        for style_content_vec, tokenized_prompts_content, labels in data_loader:
            style_content_vec = torch.tensor(style_content_vec).cuda()
            tokenized_prompts_content = torch.tensor(tokenized_prompts_content).cuda()
            labels = labels.cuda()
            # print('style_content_vec:', style_content_vec.size(), style_content_vec.dtype, 'tokenized_prompts_content:', tokenized_prompts_content.size(), tokenized_prompts_content.dtype)
            optimizer.zero_grad()

            # Forward pass through the text encoder.
            # text_features = model.encode_text(stylepromptContent)
            text_features = text_encoder(style_content_vec, tokenized_prompts_content)
            # print('text_features:', text_features.size())  #128, 77, 512

            # Apply L2 normalization
            L2_flag = True
            if L2_flag:
                text_features = F.normalize(text_features, p=2, dim=1)

            # Forward pass through the linear classifier.
            # outputs = classifier(text_features)
            output = classifier(text_features, labels)
            # print('output:', output.size())

            # Calculate ArcFace loss.
            # loss = arcface_loss(outputs, labels)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")

    classifier_dir = Path(args.classifier_dir)
    os.makedirs(classifier_dir, exist_ok=True)
    save_path = classifier_dir / "{}_{}_classifier.pth".format(
        clip_model_name.replace("/", "_"), dataset_name
    )
    # Save the trained model if needed.
    torch.save(classifier.state_dict(), save_path)
    print(f"Classifier saved to {save_path}")


def read_style_data(file_name):
    import pickle

    with open(file_name, "rb") as file:
        data = pickle.load(file)

        input_data_1 = torch.cat([p[0] for p in data], dim=0)
        input_data_2 = torch.cat([p[1] for p in data], dim=0)

        print(len(input_data_1), input_data_1[0].size())  # 560 torch.Size([77, 512])
        print(len(input_data_2), input_data_2[0].size())  # 560 torch.Size([77])

        return input_data_1, input_data_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-vision-model",
        type=str,
        default="RN50",
        help="Valid options (case sensitive): RN50, RN101, RN50x4, ViT-B/32, ViT-B/16, ViT-L/14",
    )
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument(
        "--style-vecs-dir",
        type=str,
        default="./pth/style_vecs_out",
        help="Directory containing style vectors",
    )
    parser.add_argument(
        "--classifier-dir",
        type=str,
        default="./pth/classifiers",
        help="Directory to save classifier",
    )

    args = parser.parse_args()
    print("##################################################")
    print(
        f"Training classifier with {args.clip_vision_model} CLIP backbone on {args.dataset} dataset"
    )
    print("##################################################")
    print(f"args: {args}")

    # clip_model_name = 'RN50'
    # dataset_name = 'PACS'
    clip_model_name = args.clip_vision_model
    dataset_name = args.dataset
    style_vecs_dir = Path(args.style_vecs_dir)
    print(f"style_vecs_dirs exists: {(style_vecs_dir, style_vecs_dir.exists())}")

    file_name = style_vecs_dir / "{}_{}_style_vecs_out".format(
        clip_model_name.replace("/", "_"), dataset_name
    )
    print(f"style vecs file exists: {(file_name, file_name.exists())}")
    input_data_1, input_data_2 = read_style_data(file_name)
    train_classifier(input_data_1, input_data_2, clip_model_name, dataset_name, args)

    print("Classifier training completed.\n")
    # train_classifier()
