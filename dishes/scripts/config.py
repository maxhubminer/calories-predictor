
class Config:
    # для воспроизводимости
    SEED = 123

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b2"
    
    # Какие слои размораживаем - совпадают с неймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"
    
    # Гиперпараметры
    BATCH_SIZE = 8
    #TEXT_LR = 3e-5
    TEXT_LR = 3e-3
    #IMAGE_LR = 1e-4    
    IMAGE_LR = 1e-2
    #CLASSIFIER_LR = 5e-4
    CLASSIFIER_LR = 5e-2
    EPOCHS = 10
    DROPOUT = 0.2
    HIDDEN_DIM = 256

    NORMALIZE_MASS = True
    
    # Пути
    DISH_CSV_PATH = "data/dish.csv"
    INGREDIENTS_CSV_PATH = "data/ingredients.csv"
    IMAGE_ROOT = "data/images"
    SAVE_PATH = "best_model.pth"