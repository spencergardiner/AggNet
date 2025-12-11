# AggNet Project Structure

## Overview

AggNet is a deep learning framework for predicting protein aggregation propensity using ESM (Evolutionary Scale Modeling) protein language models. The project provides two main models:

1. **APNet** - Amyloidogenic Peptide Network: Predicts if individual peptides are amyloidogenic
2. **APRNet** - Aggregation-Prone Region Network: Profiles full-length proteins to identify aggregation-prone regions

---

## 📁 Directory Structure

```
AggNet/
├── README.md                    # Main project documentation
├── NNK_TRAINING_GUIDE.md       # Detailed guide for training on NNK dataset
├── example.ipynb                # Quick start Jupyter notebook
├── CPAD2.ipynb                 # CPAD2 dataset analysis notebook
│
├── checkpoint/                  # Model checkpoints
│   ├── APNet.ckpt              # Pre-trained APNet model
│   └── selected_aaindex.pt     # AA biochemical property indices
│
├── data/                        # Datasets
│   ├── AmyHex/                 # Amyloid hexapeptide benchmarks
│   │   ├── Hex142.fasta
│   │   └── Hex1279.fasta
│   ├── CPAD2/                  # CPAD2.0 dataset
│   ├── NNK/                    # NNK aggregation dataset (cached)
│   ├── TPBLA/                  # TPBLA benchmark
│   ├── WALTZ_DB_2/             # WALTZ database
│   └── 100K/                   # Large-scale NNK data
│
├── model/                       # Model architectures
│   ├── APNet/                  # Peptide-level classification
│   │   ├── APNet.py           # Core model architecture
│   │   ├── lightning_module.py # PyTorch Lightning wrapper
│   │   └── data_module.py     # Data loading and preprocessing
│   └── APRNet/                 # Protein-level profiling
│       └── APRNet.py
│
├── script/                      # Executable scripts
│   ├── predict_amyloid.py      # Predict amyloid propensity
│   ├── predict_APR.py          # Profile proteins for APRs
│   ├── train_apnet.py          # Train APNet model
│   ├── load_nnk_data.py        # Load and preprocess NNK data
│   └── sbatch_train_apnet.sh   # SLURM training script
│
├── utils/                       # Utility modules
│   ├── bio/                    # Bioinformatics utilities
│   ├── config/                 # Configuration management
│   │   ├── config_utils.py
│   │   └── default_config/    # Default YAML configs
│   │       ├── lightning/     # Training configs
│   │       ├── model/         # Model hyperparameters
│   │       ├── dataset/       # Dataset configs
│   │       └── project/       # Project settings
│   ├── dataset/               # Dataset handling
│   │   ├── ProteinDataset.py # Core dataset class
│   │   └── Tokenizer.py      # Sequence tokenization
│   ├── datastructure/         # Data structures
│   ├── file/                  # File I/O utilities
│   ├── lightning/             # PyTorch Lightning utils
│   │   ├── trainer_utils.py  # Trainer configuration
│   │   └── predict_utils.py  # Inference utilities
│   ├── loss/                  # Loss functions
│   ├── metric/                # Evaluation metrics
│   │   ├── bin_cls.py        # Binary classification metrics
│   │   └── sov.py            # Segment overlap scores
│   ├── model/                 # Model components
│   │   └── Layer.py          # Neural network layers (MLP, etc.)
│   ├── optim/                 # Optimizers and schedulers
│   ├── parallel/              # Multi-GPU utilities
│   └── wrapper/               # External model wrappers
│       ├── ESM/              # ESM embeddings
│       └── ESMFold/          # ESMFold structures
│
└── lightning_logs/             # Training logs and checkpoints
    └── version_*/             # Experiment versions
```

---

## 🧩 Core Components

### 1. Model Architecture (`model/`)

#### APNet (`model/APNet/`)
**Purpose**: Binary classification of peptides as amyloidogenic or non-amyloidogenic.

**Key Files**:
- **`APNet.py`**: Core neural network architecture
  - Uses ESM2 (650M parameters) as feature extractor
  - Down-projection MLP: ESM embeddings (1280D) → compressed representation (embed_dim)
  - Feature MLP: Processes additional features (AAindex, one-hot encoding)
  - Task heads: Classification/regression heads for different tasks
  - Supports freezing ESM weights for efficient training

- **`lightning_module.py`**: PyTorch Lightning wrapper
  - Implements training/validation/test loops
  - Computes binary classification metrics (ACC, AUC, MCC, F1, etc.)
  - Manages optimizer and scheduler configuration
  - Logs metrics per step and per epoch

- **`data_module.py`**: Data loading and batching
  - Handles train/val/test splits
  - Custom collate function for variable-length sequences
  - Tokenization with ESM alphabet
  - Feature extraction: ESM embeddings, one-hot encoding, AAindex properties

#### APRNet (`model/APRNet/`)
**Purpose**: Sliding window analysis to identify aggregation-prone regions in full proteins.

**Key Features**:
- Uses APNet backbone with sliding window approach
- Generates per-residue aggregation scores
- Identifies contiguous high-scoring regions (APRs)

---

### 2. Data Pipeline (`utils/dataset/`)

#### ProteinDataset (`ProteinDataset.py`)
**Purpose**: Unified dataset class for protein sequences and features.

**Features**:
- Stores sequences, metadata, and computed features
- Lazy loading of features from HDF5 for memory efficiency
- Train/test splitting with stratification
- Feature caching system
- Subset and indexing support

**Structure**:
```python
ProteinDataset/
├── metadata.csv          # Sequence info (name, sequence, length, labels)
├── sequence.fasta        # FASTA format sequences
└── features/             # Cached feature files (HDF5)
    ├── esm_embedding.h5
    ├── onehot.h5
    └── aaindex.h5
```

#### Tokenizer (`Tokenizer.py`)
**Purpose**: Convert sequences to model inputs.

**Capabilities**:
- ESM tokenization (with special tokens: `<cls>`, `<eos>`, `<pad>`)
- One-hot encoding (20 amino acids)
- AAindex biochemical properties (selected subset of 566 properties)
- Handles variable-length sequences with padding

---

### 3. Training Scripts (`script/`)

#### `train_apnet.py`
**Purpose**: Train APNet on custom datasets (e.g., NNK).

**Workflow**:
1. Load dataset (from Excel or cached ProteinDataset)
2. Parse configuration (YAML + command-line overrides)
3. Initialize DataModule (handles data splits, augmentation)
4. Initialize LightningModule (model + training logic)
5. Configure Trainer (callbacks, logging, checkpointing)
6. Train model
7. Evaluate on test set

**Key Features**:
- Config-driven: All hyperparameters from YAML files
- Checkpointing: Saves top-k models based on validation loss
- Early stopping: Monitors validation loss with patience
- Logging: TensorBoard and CSV metrics

#### `load_nnk_data.py`
**Purpose**: Load and preprocess NNK aggregation dataset from Excel.

**Data Processing**:
- Reads 4 sheets: NNK1, NNK2, NNK3 (training), NNK4 (validation)
- Parses sequences: splits on `*`, takes first part
- Binary labels: `nucleator` column (empty → 0)
- Filters empty sequences
- Caches processed data to `./data/NNK/`

**Output**: ProteinDataset with ~127K sequences

---

### 4. Prediction Scripts (`script/`)

#### `predict_amyloid.py`
**Purpose**: Predict amyloid propensity for peptides in FASTA file.

**Usage**:
```bash
python script/predict_amyloid.py \
    --fasta data/AmyHex/Hex142.fasta \
    --checkpoint checkpoint/APNet.ckpt \
    --batch_size 256 \
    --output results.csv
```

**Output**: CSV with columns:
- `peptide`: Sequence
- `probability`: Amyloid probability (0-1)
- `label`: "amyloid" or "non-amyloid"

#### `predict_APR.py`
**Purpose**: Profile full-length protein to identify aggregation-prone regions.

**Usage**:
```bash
python script/predict_APR.py \
    --sequence QVQLVQSGAE... \
    --checkpoint checkpoint/APNet.ckpt \
    --output apr_profile.csv
```

**Output**: Per-residue aggregation scores and identified APRs.

---

### 5. Configuration System (`utils/config/`)

#### Config Architecture
Uses OmegaConf for hierarchical YAML configuration.

**Default Configs** (`utils/config/default_config/`):

**`lightning/LitModel.yaml`**:
```yaml
trainer:
  max_epochs: 50
  accelerator: auto
  devices: auto
  gradient_clip_val: 2.0
  
ckpt_callback:
  monitor: valid/loss_epoch
  save_top_k: 10
  mode: min
  
early_stop_callback:
  monitor: valid/loss_epoch
  patience: 200
  min_delta: 0.01
```

**`model/`**: Hyperparameters for APNet/APRNet
**`dataset/`**: Dataset-specific settings
**`project/`**: Paths, logging, seeds

**Overriding**:
- Command-line: `python train_apnet.py --trainer.max_epochs=100`
- Programmatic: `config_overrides={'trainer.max_epochs': 100}`

---

### 6. Utilities (`utils/`)

#### Loss Functions (`loss/`)
- Binary cross-entropy
- Focal loss (for imbalanced data)
- Custom aggregation-specific losses

#### Metrics (`metric/`)
- **`bin_cls.py`**: Binary classification metrics
  - ACC, AUC, MCC, F1, F0.5, F2
  - Sensitivity (SE), Specificity (SP)
  - Precision (PPV), NPV
  - Confusion matrix elements (TP, FP, TN, FN)

#### Lightning Utilities (`lightning/`)
- **`trainer_utils.py`**: Constructs PyTorch Lightning Trainer
  - Configures callbacks (checkpoint, early stopping, progress bar)
  - Sets up loggers (TensorBoard, CSV)
  - Handles multi-GPU training

- **`predict_utils.py`**: Inference wrapper
  - `LitModelInference`: Loads checkpoint and runs predictions
  - Batch prediction with efficient data loading
  - Result aggregation

#### Wrappers (`wrapper/`)
- **`ESM/`**: ESM2 embedding extraction
  - Batch processing
  - GPU acceleration
  - Caching to HDF5
  
- **`ESMFold/`**: Structure prediction (optional)

---

## 🔄 Data Flow

### Training Workflow

```
Excel/FASTA File
    ↓
[load_nnk_data.py] → Parse sequences, labels
    ↓
ProteinDataset
    ├── metadata.csv (name, sequence, label)
    └── features/ (cached)
    ↓
DataModule
    ├── Train/Val/Test Split
    ├── Tokenization (ESM)
    ├── Feature Extraction
    └── DataLoader (batching)
    ↓
LightningModule
    ├── APNet Model
    │   ├── ESM2 Encoder (frozen)
    │   ├── Down-projection MLP
    │   └── Classification Head
    ├── Loss Computation
    └── Metric Logging
    ↓
Trainer
    ├── Training Loop
    ├── Validation
    ├── Checkpointing
    └── Early Stopping
    ↓
Trained Model (checkpoint.ckpt)
```

### Prediction Workflow

```
FASTA File
    ↓
[predict_amyloid.py] → Read sequences
    ↓
LitModelInference
    ├── Load checkpoint
    └── Create DataModule
    ↓
Batch Prediction
    ├── Tokenization
    ├── ESM Embeddings
    └── Forward Pass
    ↓
Results
    ├── Probabilities
    └── Labels
    ↓
CSV Output
```

---

## 🎯 Key Design Patterns

### 1. Feature Caching
- Expensive operations (ESM embeddings) cached to HDF5
- Lazy loading: features loaded only when accessed
- Index mapping for subsets

### 2. Config-Driven Training
- All hyperparameters in YAML
- Easy to reproduce experiments
- Command-line overrides for quick tweaks

### 3. PyTorch Lightning
- Separates model logic from training boilerplate
- Automatic multi-GPU support
- Built-in checkpointing and logging

### 4. Modular Architecture
- Models, datasets, and utilities are independent
- Easy to swap ESM models or add new features
- Reusable components across projects

---

## 📊 Datasets

### NNK Dataset
- **Source**: Thompson lab, ~127K sequences
- **Format**: Excel with 4 sheets
- **Task**: Binary classification (aggregating/non-aggregating)
- **Sequences**: 1-20 amino acids
- **Labels**: 0 (non-nucleator), 1 (nucleator)

### Benchmark Datasets
- **AmyHex**: Hexapeptide benchmarks (142, 1279 sequences)
- **CPAD2**: Curated amyloid database
- **TPBLA**: Beta-lactamase aggregation
- **WALTZ_DB_2**: WALTZ database sequences

---

## 🛠️ Common Tasks

### Train APNet on NNK
```bash
python script/train_apnet.py \
    data/100K/massive_exp__aggregation__thompson.xlsx \
    --trainer.max_epochs=50 \
    --train_dataloader.batch_size=64
```

### Predict Amyloid Propensity
```bash
python script/predict_amyloid.py \
    --fasta data/AmyHex/Hex142.fasta \
    --checkpoint checkpoint/APNet.ckpt \
    --output results.csv
```

### Profile Protein for APRs
```bash
python script/predict_APR.py \
    --sequence MVLSPADKTNVKAAW... \
    --checkpoint checkpoint/APNet.ckpt \
    --output apr_profile.csv
```

### Force Reload Dataset (Clear Cache)
```bash
rm -rf data/NNK/
python script/load_nnk_data.py <excel_path> --force_reload
```

---

## 🔧 Technical Details

### ESM Integration
- **Model**: ESM2 (650M parameters)
- **Embeddings**: 1280-dimensional per-residue representations
- **Frozen**: ESM weights frozen during training (computational efficiency)
- **Tokenization**: Special tokens for sequence boundaries

### Features Used
1. **ESM Embeddings**: Contextual protein language model representations
2. **One-Hot Encoding**: Binary encoding of 20 amino acids
3. **AAindex**: Biochemical properties (hydrophobicity, charge, etc.)

### Model Architecture
```
Input Sequence
    ↓
ESM2 (frozen)
    ↓
[CLS] Token Embedding (1280D)
    ↓
Down-projection MLP (1280D → embed_dim)
    ↓
[Optional] Feature MLP (AAindex + One-hot)
    ↓
Concatenate
    ↓
Task Head (Classification)
    ↓
Softmax → [P(non-amyloid), P(amyloid)]
```

### Training Configuration
- **Optimizer**: AdamW (typical)
- **Scheduler**: Cosine annealing / ReduceLROnPlateau
- **Loss**: Binary cross-entropy
- **Batch Size**: 64-256
- **Epochs**: 50-200 with early stopping
- **Gradient Clipping**: 2.0

---

## 🐛 Troubleshooting

### Issue: `AttributeError: 'float' object has no attribute 'strip'`
**Cause**: Corrupted data in cache (NaN values in sequences)
**Fix**: Delete cache and reload
```bash
rm -rf data/NNK/
```

### Issue: `RuntimeError: Early stopping conditioned on metric 'valid_loss_epoch'`
**Cause**: Metric name mismatch (underscores vs. slashes)
**Fix**: Update config to use `valid/loss_epoch`

### Issue: Out of memory during training
**Solutions**:
- Reduce batch size: `--train_dataloader.batch_size=32`
- Use gradient accumulation
- Freeze ESM model (should already be frozen)

### Issue: Slow ESM embedding extraction
**Solutions**:
- Embeddings are cached after first run
- Use GPU: `--trainer.accelerator=gpu`
- Reduce batch size if GPU memory is limited

---

## 📚 References

- **ESM**: [Evolutionary Scale Modeling (Meta AI)](https://github.com/facebookresearch/esm)
- **PyTorch Lightning**: [lightning.ai](https://lightning.ai/)
- **AAindex**: [Amino acid indices database](https://www.genome.jp/aaindex/)

---

## 🔮 Future Extensions

- Support for additional protein language models (ProtTrans, ESM3)
- Multi-task learning (aggregation + structure + function)
- Active learning for dataset expansion
- Real-time prediction API
- Integration with AlphaFold structures

---

**Last Updated**: December 2025  
**Maintainer**: AggNet Development Team
