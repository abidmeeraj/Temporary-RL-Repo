from config import Config
from dataset import MultimodalDataset
from torch.utils.data import DataLoader
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import MultimodalSummarizationEnv
import os

config = Config()

def make_env(config, batch_sample, lmdb_path):
    env = MultimodalSummarizationEnv(
        max_images_to_select=config.MAX_IMAGES_SUMMARIZATION,
        text_feature_dim=config.TEXT_DIM,
        image_feature_dim=config.IMAGE_DIM,
        device=config.RL_DEVICE,
        lmdb_path=lmdb_path,
    )
    env.set_sample_data(
        batch_sample["text_embeddings"],
        batch_sample["attention_mask"],
        batch_sample["image_paths"],
        batch_sample["text"],
    )
    return env

def train_rl_model(
    config,
    train_loader,
    rl_model,
    current_epoch,
):
    checkpoint_folder = config.RL_CHECKPOINT_FOLDER

    # Load pre-trained RL model if not already initialized
    if rl_model is None:
        rl_model = load_checkpoint_rl(
            envs=None,  # Dummy environment; it will be reset dynamically
            device=config.RL_DEVICE,
            folder=checkpoint_folder,
            epoch=current_epoch - 1,
        )

    for batch_idx, batch in enumerate(train_loader):
        num_envs = len(batch["text_embeddings"])  # Number of instances in the batch
        timesteps_per_batch = num_envs * config.MAX_IMAGES_RL

        # Create environments for the current batch
        envs = DummyVecEnv(
            [
                lambda: make_env(
                    config,
                    {
                        "text_embeddings": batch["text_embeddings"][i],
                        "attention_mask": batch["attention_mask"][i],
                        "image_paths": batch["image_paths"][i],
                        "text": batch["text"][i],
                    },
                    config.LMDB_PATH,
                )
                for i in range(num_envs)
            ]
        )

        # Dynamically reload RL model if environment size changes or no model exists
        if rl_model is None or rl_model.n_envs != num_envs:
            if rl_model is not None:
                # Save the current RL model
                print(f"Saving RL model before resizing for batch {batch_idx + 1}.")
                save_checkpoint_rl(
                    rl_model,
                    checkpoint_folder,
                    epoch=current_epoch,
                    prefix="temp_a2c_model",
                )

            # Check if a checkpoint exists to reload
            checkpoint_path = os.path.join(
                checkpoint_folder, f"temp_a2c_model_epoch_{current_epoch}.pth"
            )
            if os.path.exists(checkpoint_path):
                print(f"Reloading RL model from {checkpoint_path}.")
                rl_model = A2C.load(checkpoint_path, env=envs, device=config.RL_DEVICE)
            else:
                # Initialize a new RL model if no checkpoint exists
                print("No checkpoint found. Initializing a new RL model.")
                rl_model = A2C("MlpPolicy", envs, verbose=1, device=config.RL_DEVICE)
        else:
            rl_model.set_env(envs)

        # Train RL model on the current batch
        rl_model.learn(total_timesteps=timesteps_per_batch)

        # Close the environments after processing the batch
        envs.close()

    # Save the RL model for the current epoch
    save_checkpoint_rl(rl_model, checkpoint_folder, epoch=current_epoch)

    return rl_model

def load_checkpoint_rl(envs, device, folder, prefix="a2c_image_selection", epoch=None):
    """
    Load the RL model checkpoint by epoch or the latest checkpoint.
    """
    if epoch is not None:
        filename = f"{prefix}_epoch_{epoch}.pth"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            print(f"Loading RL model checkpoint from {filepath}")
            return A2C.load(filepath, env=envs, device=device)
        else:
            print(f"No RL model checkpoint found at {filepath}")
            return None
    else:
        # Load the latest checkpoint based on modification time
        checkpoint_files = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.startswith(prefix) and f.endswith(".pth")
            ],
            key=lambda x: os.path.getmtime(os.path.join(folder, x)),
            reverse=True,
        )
        if checkpoint_files:
            latest_checkpoint = os.path.join(folder, checkpoint_files[0])
            print(f"Loading latest RL model checkpoint from {latest_checkpoint}")
            return A2C.load(latest_checkpoint, env=envs, device=device)
        else:
            print(f"No RL model checkpoints found in folder: {folder}")
            return None
        

def save_checkpoint_rl(rl_model, folder, epoch, prefix="a2c_image_selection"):
    """
    Save the RL model checkpoint with epoch numbers.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{prefix}_epoch_{epoch}.pth"
    filepath = os.path.join(folder, filename)
    rl_model.save(filepath)
    print(f"RL model checkpoint saved: {filepath}")

# Initialize dataset and DataLoader
train_dataset = MultimodalDataset(
    csv_file=config.TRAIN_CSV_PATH, max_images=config.MAX_IMAGES_RL
)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

rl_model = None

for epoch in range(config.EPOCHS):
    rl_model = train_rl_model(
                        config,
                        train_loader,
                        rl_model,
                        current_epoch=epoch,
                    )