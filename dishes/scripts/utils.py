import os
from functools import partial

import re

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW

from scripts.dataset import MultimodalDataset, collate_fn, get_transforms

from tqdm import tqdm

def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False

class MultimodalRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # --------------------------
        # Текстовая модель
        # --------------------------
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)

        # --------------------------
        # Визуальная модель
        # --------------------------
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        # --------------------------
        # Numeric features (mass)
        # --------------------------
        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1)
        )

        # --------------------------
        # Fusion + регрессия
        # --------------------------
        fusion_dim = config.HIDDEN_DIM * 2 + 32
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        # Text features (CLS)
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_features)

        # Image features
        image_features = self.image_model(image)
        image_emb = self.image_proj(image_features)

        # Numeric features
        numeric_emb = self.numeric_mlp(mass)

        # Fusion
        fused = torch.cat([text_emb, image_emb, numeric_emb], dim=1)
        out = self.regressor(fused)
        return out.squeeze(1)  # [B]


def train(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------
    # Инициализация модели и токенайзера
    # --------------------------
    model = MultimodalRegressor(config).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Разморозка слоёв
    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE)

    # !!!
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer = AdamW([
        {'params': model.text_model.parameters(), 'lr': config.TEXT_LR},
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.numeric_mlp.parameters(), 'lr': config.CLASSIFIER_LR},
        {'params': model.regressor.parameters(), 'lr': config.CLASSIFIER_LR},
    ])

    criterion = nn.SmoothL1Loss()

    # --------------------------
    # Загрузка данных
    # --------------------------
    train_transforms = get_transforms(config, ds_type="train")
    val_transforms = get_transforms(config, ds_type="test")

    train_dataset = MultimodalDataset(config, train_transforms, ds_type="train")
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="test")

    # !!!+
    from torch.utils.data import random_split

    subset_size = int(0.5 * len(train_dataset))
    rest_size = len(train_dataset) - subset_size

    subset_dataset, _ = random_split(train_dataset, [subset_size, rest_size])

    # visualize+
    import matplotlib.pyplot as plt
    import numpy as np
    from random import sample
    from torch.utils.data import Subset

        # val:
    subset_sizeV = int(0.5 * len(val_dataset))
    rest_sizeV = len(val_dataset) - subset_sizeV

    subset_datasetV, _ = random_split(val_dataset, [subset_sizeV, rest_sizeV])

    # !!!-
    train_loader = DataLoader(
        subset_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        subset_datasetV,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    
    subset_sizeVIZ = int(0.5 * len(train_dataset))
    indices = sample(range(len(train_dataset)), subset_sizeVIZ)
    subset_datasetVIZ = Subset(train_dataset, indices)

    # Access original dataset for image_cfg
    orig_dataset = subset_datasetVIZ.dataset

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices[:10]):  # pick 10 random from subset
        sample_item = orig_dataset[idx]
        image = sample_item['image']

        # Convert to [H,W,C] and unnormalize
        image = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array(orig_dataset.image_cfg.mean)
        std = np.array(orig_dataset.image_cfg.std)
        # image = (image * std + mean).clip(0, 1)

        ax.imshow(image)
        ax.set_title(f"Calories: {sample_item['label']:.0f}\nMass: {sample_item['mass'].item():.0f}g")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    # visualize-

    # --------------------------
    # Тренировка
    # --------------------------
    best_loss = float("inf")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)
    
        for batch in loop:
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'image': batch['image'].to(DEVICE),
                'mass': batch['mass'].to(DEVICE)
            }

            labels = batch['label'].to(DEVICE)
            outputs = model(**inputs)

            loss = criterion(outputs, labels)  # SmoothL1Loss или MSELoss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_mae = validate(model, val_loader, DEVICE)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f}")

        # Сохраняем лучшую модель по MAE
        if val_mae < best_loss:
            best_loss = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)



def validate(model, val_loader, device):
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            labels = batch['label'].to(device)

            outputs = model(**inputs)
            mae = torch.abs(outputs - labels).sum()
            total_mae += mae.item()
            total_samples += labels.size(0)

    return total_mae / total_samples  # среднее по всем образцам
