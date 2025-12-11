"""
Training script for APNet on NNK dataset.

Usage:
    python script/train_apnet.py <excel_path> [options]
    
Example:
    python script/train_apnet.py data/nnk_data.xlsx --max_epochs 50 --batch_size 64
"""

import os
import sys
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import lightning as L
from model.APNet.data_module import DataModule
from model.APNet.lightning_module import LightningModule
from utils.config import parse_config, log_config
from utils.lightning import get_pl_trainer, seed_everything
from load_nnk_data import load_nnk_data


def train_apnet_nnk(excel_path, config_overrides=None):
    """
    Train APNet on NNK dataset.
    
    Args:
        excel_path: Path to Excel file containing NNK data
        config_overrides: Dictionary of config overrides or list of command-line style arguments
    """
    
    # Load NNK dataset
    print("="*70)
    print("STEP 1: Loading NNK Dataset")
    print("="*70)

    # Parse configuration
    print("\n" + "="*70)
    print("STEP 2: Configuring Training")
    print("="*70)
    
    # Start with default config
    config_args = []
    
    # Add dataset-specific config
    # config_args.append('utils/config/default_config/dataset/NNKDataset.yaml')
    
    # Apply any overrides
    if config_overrides:
        if isinstance(config_overrides, dict):
            for key, value in config_overrides.items():
                config_args.append(f'{key}={value}')
        elif isinstance(config_overrides, list):
            config_args.extend(config_overrides)
    
    config = parse_config(*config_args)
    
    # Set dataset path
    # config.dataset.dataset = dataset
    
    log_config(config)
    
    # Set random seed
    seed_everything(config.seed)
    
    # Initialize DataModule
    print("\n" + "="*70)
    print("STEP 3: Preparing Data")
    print("="*70)
    data_module = DataModule(config, log=True)
    data_module.dataset = load_nnk_data(excel_path, output_dir='./data/NNK')
    data_module.max_len = data_module.dataset.metadata.length.max()
    data_module.prepare_data()
    
    # Initialize Model
    print("\n" + "="*70)
    print("STEP 4: Initializing Model")
    print("="*70)
    model = LightningModule(config)
    
    # Initialize Trainer
    print("\n" + "="*70)
    print("STEP 5: Setting up Trainer")
    print("="*70)
    trainer = get_pl_trainer(config)
    
    # Train
    print("\n" + "="*70)
    print("STEP 6: Training")
    print("="*70)
    trainer.fit(model, data_module)
    
    # Test on validation set
    print("\n" + "="*70)
    print("STEP 7: Validation")
    print("="*70)
    trainer.validate(model, data_module)
    
    # Test on test set
    print("\n" + "="*70)
    print("STEP 8: Testing")
    print("="*70)
    trainer.test(model, data_module)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Logs and checkpoints saved to: {trainer.logger.log_dir}")
    
    return trainer, model, data_module


def main():
    parser = argparse.ArgumentParser(description='Train APNet on NNK dataset')
    
    # Required arguments
    parser.add_argument('excel_path', type=str, 
                        help='Path to Excel file containing NNK data')
    
    # Training hyperparameters
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum sequence length (default: None, no filtering)')
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension (default: 256)')
    parser.add_argument('--esm_freeze', action='store_true', default=True,
                        help='Freeze ESM model weights (default: True)')
    
    # Training configuration
    parser.add_argument('--devices', type=str, default='auto',
                        help='Devices to use for training (default: auto)')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator to use (default: auto)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--project_name', type=str, default='APNet_NNK',
                        help='Project name for logging (default: APNet_NNK)')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload dataset even if already cached')
    
    args = parser.parse_args()
    
    # Build config overrides
    config_overrides = [
        f'project={args.project_name}',
        f'seed={args.seed}',
        f'trainer.max_epochs={args.max_epochs}',
        f'trainer.devices={args.devices}',
        f'trainer.accelerator={args.accelerator}',
        f'train_dataloader.batch_size={args.batch_size}',
        f'valid_dataloader.batch_size={args.batch_size}',
        f'test_dataloader.batch_size={args.batch_size}',
        f'train_dataloader.num_workers={args.num_workers}',
        f'valid_dataloader.num_workers={args.num_workers}',
        f'test_dataloader.num_workers={args.num_workers}',
        f'optimizer.args.lr={args.learning_rate}',
        f'hparams.embed_dim={args.embed_dim}',
        f'hparams.esm_freeze={args.esm_freeze}',
    ]
    
    if args.max_len is not None:
        config_overrides.append(f'dataset.max_len={args.max_len}')
    
    # Train model
    train_apnet_nnk(args.excel_path, config_overrides)


if __name__ == '__main__':
    main()
