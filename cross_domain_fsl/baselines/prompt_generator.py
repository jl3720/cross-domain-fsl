import argparse
import torch
import clip
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import torch.nn as nn
from pathlib import Path
from cross_domain_fsl.methods.prompt_learner import PromptLearner
from cross_domain_fsl.methods.text_encoder import TextEncoder

from cross_domain_fsl.utils.configs import CLIP_DIM_MAPPING, CLASS_NAMES_MAPPING

device = "cuda"

PROMPT_LEARNER_CHKP_DIR = Path("./pth/prompt_learner")
print(
    f"Prompt learner checkpoint directory: {PROMPT_LEARNER_CHKP_DIR}, {PROMPT_LEARNER_CHKP_DIR.exists()}"
)

if not PROMPT_LEARNER_CHKP_DIR.exists():
    PROMPT_LEARNER_CHKP_DIR.mkdir(parents=True, exist_ok=True)

STYLE_VECS_OUT_DIR = Path("./pth/style_vecs_out")
print(
    f"Style vectors output directory: {STYLE_VECS_OUT_DIR}, {STYLE_VECS_OUT_DIR.exists()}"
)
if not STYLE_VECS_OUT_DIR.exists():
    STYLE_VECS_OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_diversity_loss(style_feature, previous_style_features):
    """
    Computes the style diversity loss.
    """
    if previous_style_features is not None:
        N = previous_style_features.shape[0]
        style_feature = style_feature.repeat(N, 1)

        cosine_similarity = F.normalize(style_feature, p=2, dim=-1) * F.normalize(
            previous_style_features, p=2, dim=-1
        )
        loss = torch.mean(torch.abs(cosine_similarity.sum(dim=-1)))
        # print('version1:', loss)
        # cosine_similarity = F.cosine_similarity(F.normalize(style_feature, p=2, dim=-1), F.normalize(previous_style_features, p=2, dim=-1))
        # loss = torch.mean(torch.abs(cosine_similarity.sum(dim=-1)))
        # print('version2:', loss)
    else:
        loss = None
    return loss


def content_consistency_loss(style_content_feature, content_feature):
    """
    Computes the content consistency loss.
    """
    # style_content_feature = content_feature
    style_content_feature = style_content_feature / style_content_feature.norm(
        dim=-1, keepdim=True
    )
    content_feature = content_feature / content_feature.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_style_content_feature = style_content_feature @ content_feature.t()
    logits_content_feature = logits_style_content_feature.t()
    ground_truth = torch.arange(
        style_content_feature.shape[0], dtype=torch.long, device="cuda"
    )
    loss = (
        F.cross_entropy(logits_style_content_feature, ground_truth)
        + F.cross_entropy(logits_content_feature, ground_truth)
    ) / 2
    return loss


def total_prompt_loss(
    style_feature, previous_style_features, style_content_feature, content_feature
):
    """
    Computes the total prompt loss.
    """
    Lstyle = style_diversity_loss(style_feature, previous_style_features)
    Lcontent = content_consistency_loss(style_content_feature, content_feature)
    if Lstyle is not None:
        Lprompt = Lstyle + Lcontent
    else:
        Lprompt = Lcontent
    return Lprompt


def prompt_generator(text_encoder, pre_style_feas_list, prompt_learner, dim=512):
    if len(pre_style_feas_list):
        pre_style_feas = torch.zeros([len(pre_style_feas_list), dim])
        for index, pre_style_fea in enumerate(pre_style_feas_list):
            pre_style_feas[index, :] = pre_style_fea
    else:
        pre_style_feas = None
    style_vec, style_content_vec = prompt_learner.forward()  # N, 77, 512
    tokenized_prompts_style = prompt_learner.tokenized_prompts_style  # N, 77
    tokenized_prompts_content = prompt_learner.tokenized_prompts_content
    style_fea = text_encoder(style_vec, tokenized_prompts_style)
    style_content_feas = text_encoder(style_content_vec, tokenized_prompts_content)
    return style_fea, pre_style_feas, style_content_feas


def train(args: argparse.Namespace):  # return
    # clip_model_name = 'RN50'
    # clip_model_name = 'ViT-B/16'
    clip_model_name = args.clip_vision_model
    dim = CLIP_DIM_MAPPING[clip_model_name]
    clip_model, _ = clip.load(clip_model_name, device=device)
    dataset_name = args.dataset
    # dataset_name = 'PACS'
    # dataset_name = 'OfficeHome'
    # dataset_name = 'VLCS'
    # dataset_name = 'DomainNet'

    lr = 0.002
    # class_names = os.listdir('D:\work_code\dg\miro-main\miro-main\domainbed\scripts\PACS\cartoon')
    # class_names = sorted(os.listdir('/sdc1/fuyuqian/datasets/PACS/cartoon/')) # PACS
    # class_names = sorted(os.listdir('/sdc1/fuyuqian/datasets/OfficeHome/Art/'))  #officehome

    class_names = CLASS_NAMES_MAPPING[dataset_name]
    print("class_names:", class_names, len(class_names))
    class_tokens = clip.tokenize(class_names).cuda()
    with torch.no_grad():
        content_feas = clip_model.encode_text(class_tokens)
    style_num = 80

    style_feas_out = []
    style_content_feas_out = []
    text_encoder = TextEncoder(clip_model)
    text_encoder.cuda()
    # text_encoder.eval()
    L = 100
    # pkl.dump(style_content_feas_out, open('./{}_style_vecs_out'.format(dataset_name), 'wb'))
    for i in range(style_num):
        prompt_learner = PromptLearner(class_names, clip_model)
        prompt_learner.train()
        prompt_learner.cuda()
        optimizer = optim.SGD(prompt_learner.parameters(), lr=lr, momentum=0.9)
        for j in range(L):
            style_fea, pre_style_feas, style_content_feas = prompt_generator(
                text_encoder, style_feas_out, prompt_learner, dim
            )
            style_fea = style_fea.cuda()
            if pre_style_feas is not None:
                pre_style_feas = pre_style_feas.cuda()
            style_content_feas = style_content_feas.cuda()
            content_feas = content_feas.cuda()
            loss = total_prompt_loss(
                style_fea, pre_style_feas, style_content_feas, content_feas
            )
            print(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            # print(prompt_learner.ctx.grad)
            optimizer.step()

        style_vec, style_content_vec = prompt_learner.forward()
        tokenized_prompts_style = prompt_learner.tokenized_prompts_style
        tokenized_prompts_content = prompt_learner.tokenized_prompts_content
        style_fea = text_encoder(style_vec, tokenized_prompts_style)
        style_feas_out.append(style_fea.detach())
        style_content_feas_out.append((style_content_vec, tokenized_prompts_content))
        torch.save(
            prompt_learner.state_dict(),
            PROMPT_LEARNER_CHKP_DIR
            / "{}_{}th_style.pth".format(clip_model_name.replace("/", "_"), str(i)),
        )
    style_vecs_path = STYLE_VECS_OUT_DIR / "{}_{}_style_vecs_out".format(
        clip_model_name.replace("/", "_"), dataset_name
    )
    pkl.dump(style_content_feas_out, open(style_vecs_path, "wb"))
    print(f"Style vectors saved to {style_vecs_path}.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-vision-model",
        type=str,
        default="RN50",
        help="Valid options (case sensitive): RN50, RN101, RN50x4, ViT-B/32, ViT-B/16, ViT-L/14",
    )
    parser.add_argument("--dataset", type=str, default="PACS")

    args = parser.parse_args()
    print("##################################################")
    print(
        f"Training prompt generator for {args.dataset} dataset using {args.clip_vision_model} model."
    )
    print("##################################################")
    train(args)
    print("Prompt generator training completed.\n")
