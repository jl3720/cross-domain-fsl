from cross_domain_fsl.data import (
    ISIC_few_shot,
    EuroSAT_few_shot,
    CropDisease_few_shot,
    Chest_few_shot,
)

MANAGER_DICT = {
    "ISIC": ISIC_few_shot.SetDataManager,
    "EuroSAT": EuroSAT_few_shot.SetDataManager,
    "CropDisease": CropDisease_few_shot.SetDataManager,
    "ChestX": Chest_few_shot.SetDataManager,
}
