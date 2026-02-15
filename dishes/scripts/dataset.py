# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train"):
        dish_df = pd.read_csv(config.DISH_CSV_PATH)
        ingr_df = pd.read_csv(config.INGREDIENTS_CSV_PATH)

        # фильтрация по split
        self.df = dish_df[dish_df["split"] == ds_type].reset_index(drop=True)

        # id -> название ингредиента
        self.ingr_map = dict(zip(ingr_df["id"], ingr_df["ingr"]))

        self.image_root = config.IMAGE_ROOT
        self.transforms = transforms

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # ⚠️ можно заранее нормализовать массу
        # if config.NORMALIZE_MASS:
        #     self.mass_mean = self.df["total_mass"].mean()
        #     self.mass_std = self.df["total_mass"].std()
        # else:
        self.mass_mean = None
        self.mass_std = None

    def __len__(self):
        return len(self.df)

    def _build_text_from_ingredients(self, ingredients_str):
        ids = ingredients_str.split(";")
        names = [self.ingr_map[i] for i in ids if i in self.ingr_map]
        return " ".join(names)

    def _normalize_mass(self, mass):
        if self.mass_mean is None:
            return mass
        return (mass - self.mass_mean) / (self.mass_std + 1e-6)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # ----- TEXT -----
        text = self._build_text_from_ingredients(row["ingredients"])

        # ----- IMAGE -----
        dish_id = row["dish_id"]
        img_path = os.path.join(self.image_root, str(dish_id), "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        # ----- LABEL -----
        calories = torch.tensor(
            float(row["total_calories"]), dtype=torch.float32
        )

        # ----- MASS -----
        mass = self._normalize_mass(float(row["total_mass"]))
        mass = torch.tensor(mass, dtype=torch.float32)

        return {
            "label": calories,
            "image": image,
            "text": text,
            "mass": mass
        }

def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    masses = torch.stack([item["mass"] for item in batch]).unsqueeze(1)

    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    return {
        "label": labels,
        "image": images,
        "mass": masses,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        # transforms = A.Compose(
        #     [
        #         A.SmallestMaxSize(
        #             max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
        #         # A.RandomCrop(
        #         #     height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
        #         # A.Affine(scale=(0.8, 1.2),
        #         #         rotate=(-15, 15),
        #         #         translate_percent=(-0.1, 0.1),
        #         #         shear=(-10, 10),
        #         #         fill=0,
        #         #         p=0.8),
        #         # A.CoarseDropout(num_holes_range=(2, 8),
        #         #                 hole_height_range=(int(0.07 * cfg.input_size[1]),
        #         #                                 int(0.15 * cfg.input_size[1])),
        #         #                 hole_width_range=(int(0.1 * cfg.input_size[2]),
        #         #                                 int(0.15 * cfg.input_size[2])),
        #         #                 fill=0,
        #         #                 p=0.5),
        #         A.ColorJitter(
        #              brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        #         A.Normalize(mean=cfg.mean, std=cfg.std),
        #         A.ToTensorV2(p=1.0)
        #     ],
        #     seed=42,
        # )
        transforms = A.Compose([
            # Resize so smallest side matches target scale
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),

            # Zoom into food & remove plate/background bias
            A.RandomResizedCrop(
                size=(cfg.input_size[1], cfg.input_size[2]),
                scale=(0.6, 1.0),
                ratio=(0.9, 1.1),
                p=1.0
            ),

            # Optional: mild color robustness (safe for food)
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.5
            ),

            # Normalize for pretrained backbone
            A.Normalize(
                mean=cfg.mean,
                std=cfg.std,
            ),

            ToTensorV2(p=1.0),
        ],
            seed=123)
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    return transforms