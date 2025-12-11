"""
Script to load and preprocess NNK dataset from Excel file.

The Excel file should contain sheets: NNK1, NNK2, NNK3, NNK4
- NNK1-3: Training data (95/5 train/test split)
- NNK4: Validation set

Columns:
- aa_seq: Amino acid sequence (split on '*', take first part)
- nucleator: Binary aggregation label (0, 1, or empty -> 0)
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import ProteinDataset


def parse_nnk_sheet(df, sheet_name):
    """
    Parse a single NNK sheet from the Excel file.
    
    Args:
        df: DataFrame containing the sheet data
        sheet_name: Name of the sheet (for logging)
    
    Returns:
        Processed DataFrame with columns: name, sequence, label
    """
    print(f"Processing sheet: {sheet_name}")
    print(f"  Original rows: {len(df)}")
    
    # Check required columns
    if 'aa_seq' not in df.columns:
        raise ValueError(f"Sheet {sheet_name} missing 'aa_seq' column")
    if 'nucleator' not in df.columns:
        raise ValueError(f"Sheet {sheet_name} missing 'nucleator' column")
    
    # Process in a copy to avoid issues
    df_copy = df.copy()
    
    # Process sequences: split on '*' and take first part
    df_copy['sequence'] = df_copy['aa_seq'].apply(
        lambda x: str(x).split('*')[0] if pd.notna(x) else ''
    )
    
    # Process labels: convert nucleator to binary (empty -> 0)
    df_copy['label'] = df_copy['nucleator'].apply(
        lambda x: int(x) if pd.notna(x) and x != '' else 0
    )
    
    # Filter out invalid sequences (empty, NaN, or problematic strings like 'NA')
    # Use notna() to remove NaN and check length > 0 for empty strings
    valid_mask = df_copy['sequence'].notna() & (df_copy['sequence'].str.len() > 0)
    df_copy = df_copy[valid_mask].reset_index(drop=True)
    
    # Ensure sequence column is string type (prevents 'NA' from becoming NaN)
    df_copy['sequence'] = df_copy['sequence'].astype(str)
    
    print(f"  Valid sequences: {len(df_copy)}")
    print(f"  Label distribution: {df_copy['label'].value_counts().to_dict()}")
    
    # Create result DataFrame with explicit dtypes to prevent 'NA' string from becoming NaN
    result_df = pd.DataFrame({
        'name': [f'{sheet_name}_{i}' for i in range(len(df_copy))],
        'sequence': pd.array(df_copy['sequence'].values, dtype='string'),  # Use pandas string dtype
        'label': df_copy['label'].values
    })
    
    return result_df


def load_nnk_data(excel_path, output_dir='./data/NNK', force_reload=False):
    """
    Load NNK dataset from Excel file and create ProteinDataset.
    
    Args:
        excel_path: Path to Excel file containing NNK data
        output_dir: Directory to save processed dataset
        force_reload: If True, reload even if dataset already exists
    
    Returns:
        ProteinDataset object with all NNK data
    """
    dataset_path = output_dir
    metadata_csv = os.path.join(dataset_path, 'metadata.csv')
    
    # Check if dataset already exists
    if os.path.exists(metadata_csv) and not force_reload:
        print(f"Loading existing dataset from {dataset_path}")
        dataset = ProteinDataset('NNK', path=dataset_path)
        return dataset
    
    print(f"Loading NNK data from {excel_path}")
    
    # Read Excel file
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Load all sheets
    excel_data = pd.read_excel(excel_path, sheet_name=['NNK1', 'NNK2', 'NNK3', 'NNK4(ValidationSet)'])
    
    # Process training sheets (NNK1-3)
    train_dfs = []
    for sheet_name in ['NNK1', 'NNK2', 'NNK3']:
        if sheet_name in excel_data:
            df = parse_nnk_sheet(excel_data[sheet_name], sheet_name)
            train_dfs.append(df)
        else:
            print(f"Warning: Sheet {sheet_name} not found in Excel file")
    
    # Combine training data
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\nCombined training data: {len(train_df)} sequences")
    
    # Remove duplicates from training data
    n_before = len(train_df)
    train_df = train_df.drop_duplicates(subset=['sequence'], keep='first').reset_index(drop=True)
    n_after = len(train_df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after} duplicate sequences from training data")
        print(f"  Training data after deduplication: {n_after} sequences")
    
    # Process validation sheet (NNK4)
    test_sheet = 'NNK4(ValidationSet)'
    if test_sheet in excel_data:
        test_df = parse_nnk_sheet(excel_data[test_sheet], 'NNK4')
        print(f"Validation data: {len(test_df)} sequences")
        
        # Remove duplicates from validation data
        n_before = len(test_df)
        test_df = test_df.drop_duplicates(subset=['sequence'], keep='first').reset_index(drop=True)
        n_after = len(test_df)
        if n_before != n_after:
            print(f"  Removed {n_before - n_after} duplicate sequences from validation data")
            print(f"  Validation data after deduplication: {n_after} sequences")
    else:
        raise ValueError("NNK4(TestSet) not found in Excel file")
    
    # Combine all data
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Remove any duplicates between training and validation sets
    # Keep validation set sequences if they appear in both
    n_before = len(all_data)
    # Mark which came from validation
    all_data['from_validation'] = False
    all_data.loc[len(train_df):, 'from_validation'] = True
    
    # Sort so validation sequences come first, then drop duplicates
    all_data = all_data.sort_values('from_validation', ascending=False)
    all_data = all_data.drop_duplicates(subset=['sequence'], keep='first').reset_index(drop=True)
    all_data = all_data.drop(columns=['from_validation'])
    
    n_after = len(all_data)
    if n_before != n_after:
        print(f"\nRemoved {n_before - n_after} sequences duplicated between train and validation sets")
        print(f"Total unique sequences: {n_after}")
    
    # Add length column
    all_data['length'] = all_data['sequence'].apply(len)
    
    # Assign splits: NNK1-3 get train/test, NNK4 gets valid
    # We'll mark NNK1-3 as 'train' temporarily and split later
    all_data['split'] = 'train'
    all_data.loc[len(train_df):, 'split'] = 'test'
    
    print(f"\nTotal sequences: {len(all_data)}")
    print(f"Sequence length range: {all_data['length'].min()} - {all_data['length'].max()}")
    print(f"Overall label distribution: {all_data['label'].value_counts().to_dict()}")
    
    # Create ProteinDataset
    dataset = ProteinDataset('NNK', path=output_dir, metadata=all_data)
    
    # Perform 95/5 train/test split on NNK1-3 data only
    train_indices = dataset.metadata[dataset.metadata['split'] == 'train'].index
    if len(train_indices) > 0:
        from sklearn.model_selection import train_test_split
        
        # Get labels for stratification
        train_labels = dataset.metadata.loc[train_indices, 'label']
        
        # Split into train (95%) and test (5%)
        actual_train_idx, test_idx = train_test_split(
            train_indices,
            test_size=0.05,
            random_state=42,
            stratify=train_labels
        )
        
        # Update splits
        dataset.metadata.loc[test_idx, 'split'] = 'valid'
        
        print(f"\nSplit summary:")
        print(f"  Train: {len(actual_train_idx)} sequences")
        print(f"  Validate: {len(test_idx)} sequences")
        print(f"  test: {len(dataset.metadata[dataset.metadata['split'] == 'valid'])} sequences")
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset.save(overwrite=True)

    df = dataset.metadata
    print(df.head())
    print()
    print(df.tail())
    print(f"\nDataset saved to {dataset_path}")
    
    return dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load NNK dataset from Excel file')
    parser.add_argument('excel_path', type=str, help='Path to Excel file containing NNK data')
    parser.add_argument('--output_dir', type=str, default='./data/NNK',
                        help='Directory to save processed dataset (default: ./data/NNK)')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload even if dataset already exists')
    
    args = parser.parse_args()
    
    dataset = load_nnk_data(args.excel_path, args.output_dir, args.force_reload)
    
    print("\n" + "="*70)
    print("Dataset loaded successfully!")
    print("="*70)
    print(f"\nDataset summary:")
    print(dataset.metadata.groupby('split')['label'].value_counts())
