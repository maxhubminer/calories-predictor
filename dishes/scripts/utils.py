import os
from functools import partial

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW

from dataset import MultimodalDataset, collate_fn, get_transforms


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
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        )

        # --------------------------
        # Fusion + регрессия
        # --------------------------
        fusion_dim = config.HIDDEN_DIM * 2 + 32
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    # --------------------------
    # Тренировка
    # --------------------------
    best_loss = float("inf")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
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
        for batch in val_loader:
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
