import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
import lmdb
import pickle


# Open the LMDB environment
def open_lmdb(lmdb_path):
    return lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
    )


# Retrieve image features by key
def get_feature_from_lmdb(env, key):
    key = key.replace("preprocessed_images/", "", 1)
    with env.begin() as txn:
        value = txn.get(key.encode())
        if value is None:
            raise KeyError(f"Key '{key}' not found in LMDB.")
        feature = pickle.loads(value)
    return torch.tensor(feature, dtype=torch.float32)


class MultimodalSummarizationEnv(gym.Env):
    def __init__(
        self,
        device,
        max_images_to_select=5,
        text_feature_dim=768,
        image_feature_dim=768,
        lmdb_path=None,  # Add LMDB path to the constructor
    ):
        super(MultimodalSummarizationEnv, self).__init__()
        self.max_selected_images = max_images_to_select
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.device = device
        self.lmdb_env = open_lmdb(lmdb_path)

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Select or discard an image
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.image_feature_dim + self.text_feature_dim,),
            dtype=np.float32,
        )

    def set_sample_data(self, text_features, text_attention_mask, image_paths, text):
        """
        Set the data for a single sample to be processed by RL.
        :param text_features: Encoded text features
        :param text_attention_mask: Mask for the text
        :param image_paths: List of paths to image feature files
        :param text: Raw text content
        :param image_mask: Mask indicating valid vs. padded images
        """
        self.text_features = text_features
        self.text_attention_mask = text_attention_mask
        self.image_paths = json.loads(image_paths)
        self.text = text

        self.selected_image_paths = []  # Initialize list to store selected image paths
        self.current_step = 0
        self.done = False

    def reset(self, seed=None, *args, **kwargs):
        """
        Reset the environment for a single sample.
        Returns the initial state for the sample.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.selected_image_paths = []  # Reset selected images
        self.current_step = 0
        self.done = False

        # Return the initial state for the sample
        state = self._get_state()
        return state, {}

    def step(self, action):
        """
        Process a single action for the current sample.
        """
        truncated = False
        info = {}
        reward = 0  # Initialize reward

        similarity_weight = 0.5
        diversity_weight = 0.3

        if not self.done:
            if action == 1:
                # Add current image path to selected paths
                self.selected_image_paths.append(self.image_paths[self.current_step])

                # Calculate rewards (e.g., similarity and diversity)
                text_feature = self._apply_text_attention_mask().to(self.device)
                image_feature = self._load_image_feature(
                    self.image_paths[self.current_step]
                ).to(self.device)

                similarity_reward = F.cosine_similarity(
                    text_feature, image_feature, dim=0
                ).item()

                # Diversity reward based on previous selected images
                if len(self.selected_image_paths) > 1:
                    diversity_reward = self._calculate_diversity_reward(image_feature)
                    reward += (similarity_weight * similarity_reward) + (
                        diversity_weight * diversity_reward
                    )
                else:
                    reward += similarity_weight * similarity_reward

            # Check if selection is complete
            if (
                self.current_step + 1 >= len(self.image_paths)
                or len(self.selected_image_paths) >= self.max_selected_images
            ):
                self.done = True

            self.current_step += 1

        # Prepare the next state
        next_state = (
            self._get_state()
            if not self.done
            else np.zeros(self.image_feature_dim + self.text_feature_dim)
        )

        # Include selected image paths in info
        info["selected_image_paths"] = self.selected_image_paths

        return next_state, reward, self.done, truncated, info

    def _get_state(self):
        """
        Get the current state combining text and image features for a single sample.
        """
        # Ensure both features are on the specified device
        pooled_text_feature = self._apply_text_attention_mask().to(self.device)
        current_image_feature = self._load_image_feature(
            self.image_paths[self.current_step]
        ).to(self.device)

        # Concatenate the features
        state = torch.cat((pooled_text_feature, current_image_feature), dim=-1)

        return (
            state.cpu().numpy()
        )  # Convert to numpy to ensure compatibility with the environment

    def _apply_text_attention_mask(self):
        """
        Apply attention mask to the text embeddings to ignore padded tokens.
        """
        expanded_mask = self.text_attention_mask.unsqueeze(-1)
        masked_text = self.text_features * expanded_mask
        non_padded_tokens = expanded_mask.sum(dim=0).unsqueeze(0)
        non_padded_tokens = torch.clamp(non_padded_tokens, min=1.0)
        pooled_text_feature = masked_text.sum(dim=0) / non_padded_tokens
        return pooled_text_feature.squeeze(0)

    def _load_image_feature(self, image_path):
        """
        Load the image feature from a given path.
        """
        image_path = os.path.normpath(image_path).replace("\\", "/")
        # key = os.path.basename(image_path)
        return get_feature_from_lmdb(self.lmdb_env, image_path).to(self.device)

    def _calculate_diversity_reward(self, new_image_feature):
        """
        Calculate diversity reward based on similarity with previously selected images.
        """
        previous_images = [
            self._load_image_feature(path) for path in self.selected_image_paths
        ]
        previous_images_tensor = torch.stack(previous_images).to(self.device)
        diversity_reward = (
            (
                1
                - F.cosine_similarity(
                    new_image_feature.unsqueeze(0),
                    previous_images_tensor.mean(dim=0),
                    dim=1,
                )
            )
            .mean()
            .item()
        )
        return diversity_reward
