import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import os

from config import Config
def get_tokenizer(model_name, folder_path):
    """
    Load the tokenizer from the local folder if it exists, otherwise download and save it.
    If the folder doesn't exist, create it.

    Args:
    - model_name (str): Name of the model tokenizer (e.g., 'bert-base-uncased')
    - folder_path (str): Path to the folder where the tokenizer will be saved or loaded from.

    Returns:
    - tokenizer: The loaded or downloaded tokenizer.
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    # Check if tokenizer files exist in the folder
    if os.path.exists(os.path.join(folder_path, "tokenizer_config.json")):
        # print(f"Loading tokenizer from {folder_path}")
        tokenizer = AutoTokenizer.from_pretrained(folder_path)
    else:
        print(
            f"Tokenizer not found in {folder_path}. Downloading and saving tokenizer..."
        )
        # Download tokenizer and save it locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(folder_path)

    return tokenizer


def get_embedding_model(model_name, folder_path):
    """
    Load the model from the local folder if it exists, otherwise download and save it.
    If the folder doesn't exist, create it.

    Args:
    - model_name (str): Name of the pre-trained model (e.g., 'bert-base-uncased')
    - folder_path (str): Path to the folder where the model will be saved or loaded from.

    Returns:
    - model: The loaded or downloaded model.
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    # List of essential files for the safetensors format
    essential_files = ["config.json", "model.safetensors"]

    # Check if all required files exist
    if all(os.path.exists(os.path.join(folder_path, file)) for file in essential_files):
        # print(f"Loading model from {folder_path}")
        model = AutoModel.from_pretrained(
            folder_path, trust_remote_code=True
        )  # `trust_remote_code` for safetensors
    else:
        print(
            f"Model not found or incomplete in {folder_path}. Downloading and saving model..."
        )
        # Download model and save it locally
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(
            folder_path, safe_serialization=True
        )  # Save as safetensors

    return model

config = Config()


class MultimodalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        max_images,
        image_feature_dim=768,
        text_feature_dim=768,
        transform=None,
    ):
        """
        :param csv_file: Path to the CSV containing dataset information.
        :param max_images: Maximum number of images to pad/truncate to.
        :param image_feature_dim: Dimension of image features (e.g., 768 for ViT or 2048 for ResNet).
        :param text_feature_dim: Dimension of text features (e.g., 768 for BERT).
        :param transform: Optional transformations for images.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = get_tokenizer(config.TOKENIZER_MODEL, config.TOKENIZER_FOLDER)
        self.embedding_model = get_embedding_model(
            config.TOKENIZER_MODEL, config.TEXT_EMBED_FOLDER
        )
        self.max_images = max_images  # Use None or a number as set in the config
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.transform = transform
        self.max_input_len = config.MAX_INPUT_LEN
        self.max_summary_len = config.MAX_SUMMARY_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extracting text, summary, and image paths
        text = self.data.iloc[idx]["Body"]
        summary = self.data.iloc[idx]["Summary"]
        images_paths = self.data.iloc[idx]["Image_Feature_Paths"]

        # Preprocess text (input) using BERT tokenizer and return BERT last hidden state (full sequence)
        encoded_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
        )
        attention_mask = encoded_text[
            "attention_mask"
        ]  # Attention mask to differentiate real tokens from padding
        with torch.no_grad():
            text_embeddings = self.embedding_model(**encoded_text)[
                "last_hidden_state"
            ]  # Shape: [batch_size, sequence_length, text_feature_dim]

        # Preprocess summary (target) with padding and attention mask
        encoded_summary = self.tokenizer(
            summary,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_summary_len,
        )
        summary_ids = encoded_summary["input_ids"].squeeze(0)
        summary_attention_mask = encoded_summary["attention_mask"].squeeze(0)

        return {
            "text_embeddings": text_embeddings.squeeze(
                0
            ),  # Return full sequence embeddings
            "attention_mask": attention_mask.squeeze(
                0
            ),  # Attention mask for real vs. padded tokens
            "summary": summary_ids,
            "summary_attention_mask": summary_attention_mask,  # Attention mask for summary
            "text": text,
            "image_paths": images_paths,
        }
