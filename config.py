# config.py
import torch


class Config:
    BATCH_SIZE = 4
    EPOCHS = 2

    # Tokenizer configuration
    TOKENIZER_MODEL = "bert-base-uncased"
    MAX_INPUT_LEN = 512
    MAX_SUMMARY_LEN = 100
    TEXT_DIM = 768
    IMAGE_DIM = 768

    TOKENIZER_FOLDER = "TOKENIZER"
    TEXT_EMBED_FOLDER = "TEXT_EMBEDDING_MODEL"

    # Image-related configurations
    MAX_IMAGES_RL = 10  # Set to None for dynamic image padding based on batch, or set a specific number like 5 which means only 5 images will be passed to RL as selection set
    MAX_IMAGES_SUMMARIZATION = (
        5  # Number of images to be selected by RL for summary generation
    )

    # Paths to datasets
    TRAIN_CSV_PATH = "dataset/train_dataset.csv"

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # TIMESTEPS for RL
    RL_TIMESTEPS = 10
    RL_SEED = 1234

    # Folder to save checkpoints
    CHECKPOINT_FOLDER = "MODEL_OUTPUTS/CHECKPOINTS"
    RL_CHECKPOINT_FOLDER = "MODEL_OUTPUTS/RL_CHECKPOINTS"

    # Folder to save loss values for training and validation
    LOSS_FOLDER = "MODEL_OUTPUTS/LOSS_LOGS"

    LMDB_PATH = "image_features.lmdb"
