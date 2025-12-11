# NNK Dataset Training Guide

This guide explains how to train APNet on the NNK aggregation dataset.

## Dataset Overview

The NNK dataset contains ~127K protein sequences with binary aggregation labels:
- **Training data**: NNK1-3 combined (119,669 sequences)
  - Train set: 113,685 sequences (95%)
  - Test set: 5,984 sequences (5%)
- **Validation data**: NNK4 (7,394 sequences)

### Data Processing
- Sequences are split on `*` and only the first part is used
- The `nucleator` column provides binary labels (0/1)
- Empty `nucleator` values are treated as 0 (non-aggregating)
- All sequences are 1-20 amino acids long

## Quick Start

### 1. Load the Dataset

```bash
# Activate your Python environment
source .venv/bin/activate

# Load the NNK data from Excel
python script/load_nnk_data.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx \
    --output_dir ./data/NNK
```

This creates:
- `data/NNK/metadata.csv` - Sequence metadata with labels and splits
- `data/NNK/sequence.fasta` - FASTA format sequences

### 2. Train APNet

```bash
# Basic training with default settings
python script/train_apnet.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx

# Training with custom hyperparameters
python script/train_apnet.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx \
    --max_epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --embed_dim 256 \
    --num_workers 8
```

## Training Configuration

### Available Command-Line Options

```bash
python script/train_apnet.py <excel_path> [OPTIONS]

Required:
  excel_path              Path to Excel file with NNK data

Training Hyperparameters:
  --max_epochs INT        Number of training epochs (default: 50)
  --batch_size INT        Batch size (default: 32)
  --learning_rate FLOAT   Learning rate (default: 1e-3)
  --max_len INT           Max sequence length filter (default: None)

Model Hyperparameters:
  --embed_dim INT         Embedding dimension (default: 256)
  --esm_freeze            Freeze ESM weights (default: True)

Training Configuration:
  --devices STR           Devices to use (default: auto)
  --accelerator STR       Accelerator type (default: auto)
  --num_workers INT       Data loading workers (default: 8)

Other Options:
  --seed INT              Random seed (default: 42)
  --project_name STR      Project name for logging (default: APNet_NNK)
  --force_reload          Force reload dataset cache
```

### Example Commands

```bash
# Train on GPU with larger batch size
python script/train_apnet.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx \
    --batch_size 128 \
    --max_epochs 100 \
    --devices 1

# Train on CPU for testing
python script/train_apnet.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx \
    --batch_size 16 \
    --max_epochs 5 \
    --accelerator cpu

# Full training with custom settings
python script/train_apnet.py \
    ./data/100K/massive_exp__aggregation__thompson.xlsx \
    --max_epochs 100 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --embed_dim 512 \
    --num_workers 16 \
    --project_name APNet_NNK_v1
```

## Model Architecture

APNet uses the following architecture for NNK training:

1. **ESM-2 Embeddings** (esm2_t33_650M_UR50D, frozen)
   - Extracts protein language model features from layer 33
   - 1280-dimensional per-token embeddings

2. **Down MLP**
   - Projects ESM embeddings: 1280 → 512 → 256
   - Average pools across sequence length

3. **Feature MLP**
   - Processes AAIndex features: 300 → 256
   - Selected physicochemical descriptors

4. **Feature Fusion**
   - Combines ESM and AAIndex: `f(x,y) = x + x*y`
   - Residual-like connection

5. **Classification Head**
   - Binary classifier: 256 → 128 → 2
   - Outputs aggregation probability

## Training Outputs

Training results are saved to `./lightning_logs/APNet_NNK/`:

```
lightning_logs/APNet_NNK/
├── version_0/
│   ├── checkpoints/          # Model checkpoints
│   │   ├── epoch=XX-*.ckpt   # Best checkpoints
│   │   └── last.ckpt         # Last checkpoint
│   ├── hparams.yaml          # Hyperparameters
│   ├── metrics.csv           # Training metrics
│   └── events.out.tfevents.* # TensorBoard logs
└── ...
```

### View Training Progress

```bash
# View metrics CSV
cat lightning_logs/APNet_NNK/version_0/metrics.csv

# Launch TensorBoard
tensorboard --logdir lightning_logs/APNet_NNK
```

## Using Trained Models

After training, load the model for inference:

```python
from model.APNet.data_module import DataModule
from model.APNet.lightning_module import LightningModule
from utils.lightning import LitModelInference

# Load trained model
checkpoint = './lightning_logs/APNet_NNK/version_0/checkpoints/last.ckpt'
model = LitModelInference(LightningModule, DataModule, checkpoint)
model.set_batch_size(batch_size=256, num_workers=1)

# Predict on new sequences
sequences = ['STVIIE', 'GGVVIA', 'KVVVLK']
predictions = model.predict(sequences)
```

## Dataset Statistics

**NNK Training Set (113,685 sequences)**
- Non-aggregating (label 0): 91,027 (80.1%)
- Aggregating (label 1): 22,658 (19.9%)

**NNK Test Set (5,984 sequences)**
- Non-aggregating (label 0): 4,791 (80.1%)
- Aggregating (label 1): 1,193 (19.9%)

**NNK Validation Set (7,394 sequences)**
- Non-aggregating (label 0): 5,634 (76.2%)
- Aggregating (label 1): 1,760 (23.8%)

**Sequence Length Distribution**
- Min length: 1 amino acid
- Max length: 20 amino acids
- All sequences are short peptides

## Configuration Files

Configuration is managed through YAML files:

- `utils/config/default_config/dataset/NNKDataset.yaml` - Dataset settings
- `utils/config/default_config/lightning/LitData.yaml` - Data loading
- `utils/config/default_config/lightning/LitModel.yaml` - Trainer settings
- `utils/config/default_config/model/MyModel.yaml` - Model architecture
- `utils/config/default_config/project/project.yaml` - Project settings

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python script/train_apnet.py <excel_path> --batch_size 16

# Filter longer sequences
python script/train_apnet.py <excel_path> --max_len 15
```

### Data Loading Issues
```bash
# Force reload dataset
python script/load_nnk_data.py <excel_path> --force_reload

# Or in training script
python script/train_apnet.py <excel_path> --force_reload
```

### Slow Training
```bash
# Increase workers (if you have more CPU cores)
python script/train_apnet.py <excel_path> --num_workers 16

# Use GPU if available
python script/train_apnet.py <excel_path> --accelerator gpu --devices 1
```

## Notes

- The ESM-2 model weights are frozen by default to reduce memory usage
- AAIndex features are computed on-the-fly during training
- The dataset uses stratified splitting to maintain class balance
- Random seed is set for reproducibility (default: 42)
