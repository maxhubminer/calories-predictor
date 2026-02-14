
class Config:
    # для воспроизводимости
    SEED = 123

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"
    
    # Какие слои размораживаем - совпадают с неймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 5e-4
    EPOCHS = 10
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 5

    NORMALIZE_MASS = True
    
    # Пути
    DISH_CSV_PATH = "data/dish.csv"
    INGREDIENTS_CSV_PATH = "data/ingredients.csv"
    IMAGE_ROOT = "data/images"
    SAVE_PATH = "best_model.pth"