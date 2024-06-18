# Basic script to load a model and run inference on an image

import torch
# from timm.models.vision_transformer import _cfg
from pathlib import Path

from models_mamba import VisionMamba
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as vim_t
# TODO: Add checkpoints to repo or instructions.
# VIM_T_76 = Path("./vim/checkpoints/vim_t_midclstok_76p1acc.pth")
# VIM_T_FT_78 = Path("./vim/checkpoints/vim_t_midclstok_ft_78p3acc.pth")

# print(VIM_T_76, ": ", VIM_T_76.exists())
# print(VIM_T_FT_78, ": ", VIM_T_FT_78.exists())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
#     model = VisionMamba(
#         patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(VIM_T_76)
#         model.load_state_dict(checkpoint["model"])
#     model.to(device)
#     return model

def main():
    model = vim_t(pretrained=True)
    model.eval()
    print("Model loaded successfully")
    print(model)

    x = torch.randn(1, 3, 224, 224).to(device)
    print("x: ", x)
    with torch.no_grad():
        y = model(x, return_features=True)
        print(f"y shape: {y.shape}")



if __name__ == "__main__":
    # print("PyTorch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA version:", torch.version.cuda)
    # print("Devices:", torch.cuda.device_count())
    # print("Current device:", torch.cuda.current_device())
    # print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    # x = torch.randn(1, 3, 224, 224).to(device)
    # layer = torch.nn.Conv2d(3, 64, 3).to(device)
    # y = layer(x)
    # print(y.shape)

    main()