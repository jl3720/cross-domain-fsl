# 1 StyleAdv-CDFSL
Repository for the CVPR-2023 paper : StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning

[[Paper](https://arxiv.org/pdf/2302.09309)], [[Presentation Video on Bilibili](https://www.bilibili.com/video/BV1th4y1s78H/?spm_id_from=333.999.0.0&vd_source=668a0bb77d7d7b855bde68ecea1232e7)], [[Presentation Video on Youtube](https://youtu.be/YB-S2YF22mc)]

<img width="470" alt="image" src="https://github.com/lovelyqian/StyleAdv-CDFSL/assets/49612387/133c5248-1728-4f6e-a49c-6a7767f3a7ea">


# 2 Setup 
## 2.1 venv & code
- For Euler / Slurm machines, uses `module` loading and `conda` isn't available.
- Vim requires Python 3.10 and CUDA 11.6+.
```
module load python/3.10.4
module load cuda/11.8.0
module remove cudnn  # hack to ensure cudnn shipped with pytorch is used
python -V

# Create venv
python -m venv PATH/TO/VENV
source PATH/TO/VENV/bin/activate
```
- Follow the [readme](Vim/README.md) for the Vim dependencies setup.
- Then install `cross-domain-fsl` as editable package:
```
pip install -e cross-domain-fsl/
```

## 2.1 conda env & code
```
# conda env
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
conda install pandas
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
pip3 install tensorboard
pip3 install timm
pip3 install opencv-python==4.5.5.62
pip3 install ml-collections
# code
git clone https://github.com/lovelyqian/StyleAdv-CDFSL
cd StyleAdv-CDFSL
```

## 2.2 datasets
We use the mini-Imagenet as the single source dataset, and use cub, cars, places, plantae, ChestX, ISIC, EuroSAT, and CropDisease as novel target datasets. 

For the mini-Imagenet, cub, cars, places, and plantae, we refer to the [FWT](https://github.com/hytseng0509/CrossDomainFewShot) repo.

For the ChestX, ISIC, EuroSAT, and CropDisease, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo.

If you can't find the Plantae dataset, we provide at [here](https://drive.google.com/file/d/1e3TklMlVBCG0XRfEw6DKStJGdmmXgvq5/view?usp=drive_link), please cite its paper. 

### EuroSAT dataset
- After downloading and unzipping, you should have a folder structure like this: 
```sh
$ tree --filelimit 20  # in EuroSAT/
>>>
.
└── 2750
    ├── AnnualCrop [3000 entries exceeds filelimit, not opening dir]
    ├── Forest [3000 entries exceeds filelimit, not opening dir]
    ├── HerbaceousVegetation [3000 entries exceeds filelimit, not opening dir]
    ├── Highway [2500 entries exceeds filelimit, not opening dir]
    ├── Industrial [2500 entries exceeds filelimit, not opening dir]
    ├── Pasture [2000 entries exceeds filelimit, not opening dir]
    ├── PermanentCrop [2500 entries exceeds filelimit, not opening dir]
    ├── Residential [3000 entries exceeds filelimit, not opening dir]
    ├── River [2500 entries exceeds filelimit, not opening dir]
    └── SeaLake [3000 entries exceeds filelimit, not opening dir]

11 directories, 0 files
```

### ISIC dataset
- After downloading and unzipping, you should have a folder structure like this: 

```sh
$ tree --filelimit 20
>>>
.
├── ISIC2018_Task3_Test_GroundTruth
│   ├── ATTRIBUTION.txt
│   ├── ISIC2018_Task3_Test_GroundTruth.csv
│   └── LICENSE.txt
├── ISIC2018_Task3_Test_Input [1514 entries exceeds filelimit, not opening dir]
├── ISIC2018_Task3_Training_GroundTruth
│   ├── ATTRIBUTION.txt
│   ├── ISIC2018_Task3_Training_GroundTruth.csv
│   └── LICENSE.txt
├── ISIC2018_Task3_Training_Input [10017 entries exceeds filelimit, not opening dir]
├── ISIC2018_Task3_Validation_GroundTruth
│   ├── ATTRIBUTION.txt
│   ├── ISIC2018_Task3_Validation_GroundTruth.csv
│   └── LICENSE.txt
└── ISIC2018_Task3_Validation_Input [195 entries exceeds filelimit, not opening dir]

6 directories, 9 files
```

### CropDisease dataset
- After downloading and unzipping, you should have a folder structure like this: 

```sh
$ tree --filelimit 40 -h -d  # in CropDisease/
.
├── [4.0K]  dataset
│   ├── [4.0K]  test
│   │   ├── [ 20K]  Apple___Apple_scab
│   │   ├── [ 20K]  Apple___Black_rot
│   │   ├── [4.0K]  Apple___Cedar_apple_rust
│   │   ├── [ 36K]  Apple___healthy
│   │   ├── [ 32K]  Blueberry___healthy
│   │   ├── [ 24K]  Cherry_(including_sour)___Powdery_mildew
│   │   ├── [ 20K]  Cherry_(including_sour)___healthy
│   │   ├── [ 12K]  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
│   │   ├── [ 12K]  Corn_(maize)___Common_rust_
│   │   ├── [ 20K]  Corn_(maize)___Northern_Leaf_Blight
│   │   ├── [ 28K]  Corn_(maize)___healthy
│   │   ├── [ 28K]  Grape___Black_rot
│   │   ├── [ 36K]  Grape___Esca_(Black_Measles)
│   │   ├── [ 20K]  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
│   │   ├── [ 12K]  Grape___healthy
│   │   ├── [108K]  Orange___Haunglongbing_(Citrus_greening)
│   │   ├── [ 40K]  Peach___Bacterial_spot
│   │   ├── [ 12K]  Peach___healthy
│   │   ├── [ 20K]  Pepper,_bell___Bacterial_spot
│   │   ├── [ 32K]  Pepper,_bell___healthy
│   │   ├── [ 20K]  Potato___Early_blight
│   │   ├── [ 20K]  Potato___Late_blight
│   │   ├── [4.0K]  Potato___healthy
│   │   ├── [ 12K]  Raspberry___healthy
│   │   ├── [100K]  Soybean___healthy
│   │   ├── [ 40K]  Squash___Powdery_mildew
│   │   ├── [ 24K]  Strawberry___Leaf_scorch
│   │   ├── [ 12K]  Strawberry___healthy
│   │   ├── [ 52K]  Tomato___Bacterial_spot
│   │   ├── [ 24K]  Tomato___Early_blight
│   │   ├── [ 36K]  Tomato___Late_blight
│   │   ├── [ 20K]  Tomato___Leaf_Mold
│   │   ├── [ 36K]  Tomato___Septoria_leaf_spot
│   │   ├── [ 36K]  Tomato___Spider_mites Two-spotted_spider_mite
│   │   ├── [ 36K]  Tomato___Target_Spot
│   │   ├── [116K]  Tomato___Tomato_Yellow_Leaf_Curl_Virus
│   │   ├── [ 12K]  Tomato___Tomato_mosaic_virus
│   │   └── [ 36K]  Tomato___healthy
│   └── [4.0K]  train
│       ├── [ 52K]  Apple___Apple_scab
│       ├── [ 56K]  Apple___Black_rot
│       ├── [ 24K]  Apple___Cedar_apple_rust
│       ├── [136K]  Apple___healthy
│       ├── [120K]  Blueberry___healthy
│       ├── [ 84K]  Cherry_(including_sour)___Powdery_mildew
│       ├── [ 64K]  Cherry_(including_sour)___healthy
│       ├── [ 48K]  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
│       ├── [ 36K]  Corn_(maize)___Common_rust_
│       ├── [ 76K]  Corn_(maize)___Northern_Leaf_Blight
│       ├── [ 92K]  Corn_(maize)___healthy
│       ├── [ 96K]  Grape___Black_rot
│       ├── [104K]  Grape___Esca_(Black_Measles)
│       ├── [ 92K]  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
│       ├── [ 36K]  Grape___healthy
│       ├── [436K]  Orange___Haunglongbing_(Citrus_greening)
│       ├── [184K]  Peach___Bacterial_spot
│       ├── [ 32K]  Peach___healthy
│       ├── [ 76K]  Pepper,_bell___Bacterial_spot
│       ├── [104K]  Pepper,_bell___healthy
│       ├── [ 80K]  Potato___Early_blight
│       ├── [ 76K]  Potato___Late_blight
│       ├── [ 16K]  Potato___healthy
│       ├── [ 32K]  Raspberry___healthy
│       ├── [376K]  Soybean___healthy
│       ├── [144K]  Squash___Powdery_mildew
│       ├── [ 88K]  Strawberry___Leaf_scorch
│       ├── [ 40K]  Strawberry___healthy
│       ├── [184K]  Tomato___Bacterial_spot
│       ├── [ 76K]  Tomato___Early_blight
│       ├── [140K]  Tomato___Late_blight
│       ├── [ 76K]  Tomato___Leaf_Mold
│       ├── [140K]  Tomato___Septoria_leaf_spot
│       ├── [128K]  Tomato___Spider_mites Two-spotted_spider_mite
│       ├── [112K]  Tomato___Target_Spot
│       ├── [444K]  Tomato___Tomato_Yellow_Leaf_Curl_Virus
│       ├── [ 36K]  Tomato___Tomato_mosaic_virus
│       └── [124K]  Tomato___healthy
├── [4.0K]  test
│   ├── [ 20K]  Apple___Apple_scab
│   ├── [ 20K]  Apple___Black_rot
│   ├── [4.0K]  Apple___Cedar_apple_rust
│   ├── [ 36K]  Apple___healthy
│   ├── [ 32K]  Blueberry___healthy
│   ├── [ 24K]  Cherry_(including_sour)___Powdery_mildew
│   ├── [ 20K]  Cherry_(including_sour)___healthy
│   ├── [ 12K]  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
│   ├── [ 12K]  Corn_(maize)___Common_rust_
│   ├── [ 20K]  Corn_(maize)___Northern_Leaf_Blight
│   ├── [ 28K]  Corn_(maize)___healthy
│   ├── [ 28K]  Grape___Black_rot
│   ├── [ 36K]  Grape___Esca_(Black_Measles)
│   ├── [ 20K]  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
│   ├── [ 12K]  Grape___healthy
│   ├── [108K]  Orange___Haunglongbing_(Citrus_greening)
│   ├── [ 40K]  Peach___Bacterial_spot
│   ├── [ 12K]  Peach___healthy
│   ├── [ 20K]  Pepper,_bell___Bacterial_spot
│   ├── [ 32K]  Pepper,_bell___healthy
│   ├── [ 20K]  Potato___Early_blight
│   ├── [ 20K]  Potato___Late_blight
│   ├── [4.0K]  Potato___healthy
│   ├── [ 12K]  Raspberry___healthy
│   ├── [100K]  Soybean___healthy
│   ├── [ 40K]  Squash___Powdery_mildew
│   ├── [ 24K]  Strawberry___Leaf_scorch
│   ├── [ 12K]  Strawberry___healthy
│   ├── [ 52K]  Tomato___Bacterial_spot
│   ├── [ 24K]  Tomato___Early_blight
│   ├── [ 36K]  Tomato___Late_blight
│   ├── [ 20K]  Tomato___Leaf_Mold
│   ├── [ 36K]  Tomato___Septoria_leaf_spot
│   ├── [ 36K]  Tomato___Spider_mites Two-spotted_spider_mite
│   ├── [ 36K]  Tomato___Target_Spot
│   ├── [116K]  Tomato___Tomato_Yellow_Leaf_Curl_Virus
│   ├── [ 12K]  Tomato___Tomato_mosaic_virus
│   └── [ 36K]  Tomato___healthy
└── [4.0K]  train
    ├── [ 52K]  Apple___Apple_scab
    ├── [ 56K]  Apple___Black_rot
    ├── [ 24K]  Apple___Cedar_apple_rust
    ├── [136K]  Apple___healthy
    ├── [120K]  Blueberry___healthy
    ├── [ 84K]  Cherry_(including_sour)___Powdery_mildew
    ├── [ 64K]  Cherry_(including_sour)___healthy
    ├── [ 48K]  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
    ├── [ 36K]  Corn_(maize)___Common_rust_
    ├── [ 76K]  Corn_(maize)___Northern_Leaf_Blight
    ├── [ 92K]  Corn_(maize)___healthy
    ├── [ 96K]  Grape___Black_rot
    ├── [104K]  Grape___Esca_(Black_Measles)
    ├── [ 92K]  Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
    ├── [ 36K]  Grape___healthy
    ├── [436K]  Orange___Haunglongbing_(Citrus_greening)
    ├── [184K]  Peach___Bacterial_spot
    ├── [ 32K]  Peach___healthy
    ├── [ 76K]  Pepper,_bell___Bacterial_spot
    ├── [104K]  Pepper,_bell___healthy
    ├── [ 80K]  Potato___Early_blight
    ├── [ 76K]  Potato___Late_blight
    ├── [ 16K]  Potato___healthy
    ├── [ 32K]  Raspberry___healthy
    ├── [376K]  Soybean___healthy
    ├── [144K]  Squash___Powdery_mildew
    ├── [ 88K]  Strawberry___Leaf_scorch
    ├── [ 40K]  Strawberry___healthy
    ├── [184K]  Tomato___Bacterial_spot
    ├── [ 76K]  Tomato___Early_blight
    ├── [140K]  Tomato___Late_blight
    ├── [ 76K]  Tomato___Leaf_Mold
    ├── [140K]  Tomato___Septoria_leaf_spot
    ├── [128K]  Tomato___Spider_mites Two-spotted_spider_mite
    ├── [112K]  Tomato___Target_Spot
    ├── [444K]  Tomato___Tomato_Yellow_Leaf_Curl_Virus
    ├── [ 36K]  Tomato___Tomato_mosaic_virus
    └── [124K]  Tomato___healthy

157 directories
```

### ChestX dataset
- After downloading and unzipping, you may have a folder structure like this: 

```sh
$ tree --filelimit 20  # in ChestX/
>>>
.  # ChestX/
├── ARXIV_V5_CHESTXRAY.pdf
├── BBox_List_2017.csv
├── Data_Entry_2017.csv
├── FAQ_CHESTXRAY.pdf
├── LOG_CHESTXRAY.pdf
├── README_CHESTXRAY.pdf
├── images_001
│   └── images [4999 entries exceeds filelimit, not opening dir]
├── images_002
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_003
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_004
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_005
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_006
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_007
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_008
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_009
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_010
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_011
│   └── images [10000 entries exceeds filelimit, not opening dir]
├── images_012
│   └── images [7121 entries exceeds filelimit, not opening dir]
├── test_list.txt
└── train_val_list.txt

24 directories, 8 files
```

The `SetDataset` classes in `cross_domain_fsl/data/Chest_few_shot.py` expect only a single folder containing all the images. To achieve this, you can go into the run the following commands:

```sh
# Assuming you are in the ChestX/ directory
mkdir images
find . -name "*.png" -exec mv -t images/ {} +
```
This should result in a file structure like this:
```sh
$ tree --filelimit 20  # in ChestX/
>>>
.
├── ARXIV_V5_CHESTXRAY.pdf
├── BBox_List_2017.csv
├── Data_Entry_2017.csv
├── FAQ_CHESTXRAY.pdf
├── LOG_CHESTXRAY.pdf
├── README_CHESTXRAY.pdf
├── images [112120 entries exceeds filelimit, not opening dir]
├── test_list.txt
└── train_val_list.txt

1 directory, 8 files
```

# 3 StyleAdv based on ResNet
## 3.1 meta-train StyleAdv
Our method aims at improving the generalization ability of models, we apply the style attack and adversarial training during the meta-train stage. Once the model is meta-trained, it can be used for inference on different novel target datasets directly. 

Taking 5-way 1-shot as an example, the meta-train can be done as,
```
python3 metatrain_StyleAdv_RN.py --dataset miniImagenet --name exp-name --train_aug --warmup baseline --n_shot 1 --stop_epoch 200
```

- We integrate the testing into the training, and the testing results can be found on `output/checkpoints/exp-name/acc*.txt`;

- We set a probability `$p_{skip}$` for randomly skipping the attack, the value of it can be modified in `methods/tool_func.py`;

- We also provide our meta-trained ckps in `output/checkpoints/StyleAdv-RN-1shot` and `output/checkpoints/StyleAdv-RN-5shot`;

## 3.2 fine-tune the meta-trained StyleAdv
Though not necessary, for better performance, you may further fine-tune the meta-trained models on the target sets.

Taking 5-way 1-shot as an example, the fine-tuning on cars can be done as,
```
python3 finetune_StyleAdv_RN.py --testset cars --name exp-FT --train_aug --n_shot 1 --finetune_epoch 10 --resume_dir StyleAdv-RN-1shot --resume_epoch -1
```

- The finetuning is very sensitive to the `fintune_epoch` and `finetune_lr`;

- The value of `finetune_lr` can be modified in `finetune_StyleAdv_RN.py` :(sorry for not organizing the code very well;

- As attached in the [supplementary materials of paper](https://arxiv.org/pdf/2302.09309), we set the `finetune_epoch` and `finetune_lr` as:

  | Backbone 	| Task 	| Optimizer 	| finetune_epoch 	| finetune_lr 	|
  |:---:	|---	|---	|:---:	|---	|
  | RN10 	| 5-way 5-shot 	| Adam 	| 50 	| {0,0.001} 	|
  | RN10 	| 5-way 1-shot 	| Adam 	| 10 	| {0,0.005} 	|

# 4 StyleAdv based on ViT
coming soon

# 5 Citing
If you find our paper or this code useful for your research, please considering cite us (●°u°●)」:
```
@inproceedings{fu2023styleadv,
  title={StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning},
  author={Fu, Yuqian and Xie, Yu and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24575--24584},
  year={2023}
}
```

# 6 Acknowledge
Our code is built upon the implementation of [FWT](https://github.com/hytseng0509/CrossDomainFewShot), [ATA](https://github.com/Haoqing-Wang/CDFSL-ATA), and [PMF](https://github.com/hushell/pmf_cvpr22). Thanks for their work.
