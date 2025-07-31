"""
Data loading utilities for Seq2Seq training
Loads preprocessed data created by preprocessing_and_split.ipynb
"""

import os
import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.model_selection import train_test_split


def load_preprocessed_data(base_path=None):
    """
    Load preprocessed data created by the notebook
    Returns the same variables that the notebook creates

    Prerequisites: Run preprocessing_and_split.ipynb first!
    """

    print("Loading preprocessed data...")

    # Get the path to the data folder relative to the project root
    if base_path is None:
        # Get the directory of this script (src/models/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to project root, then into data
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, "data")
    
    print(f"  → Looking for data in: {base_path}")

    # Check if preprocessed data exists
    processed_file = os.path.join(base_path, "DATASET_PROCESSED.csv")
    vocab_word2index = os.path.join(base_path, "word2index.pkl")
    vocab_index2word = os.path.join(base_path, "index2word.pkl")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            f"❌ ERROR: Preprocessed data not found!\n\n"
            f"Missing: {processed_file}\n\n"
            f"Please run the 'preprocessing_and_split.ipynb' notebook first to create the preprocessed data.\n"
            f"The notebook will create:\n"
            f"  - DATASET_PROCESSED.csv\n"
            f"  - word2index.pkl\n"
            f"  - index2word.pkl\n\n"
            f"Then you can run this training script."
        )

    if not os.path.exists(vocab_word2index) or not os.path.exists(vocab_index2word):
        raise FileNotFoundError(
            f"❌ ERROR: Vocabulary files not found!\n\n"
            f"Missing: {vocab_word2index} or {vocab_index2word}\n\n"
            f"Please run the 'preprocessing_and_split.ipynb' notebook first."
        )

    print("  → Loading vocabulary...")
    with open(vocab_word2index, "rb") as f:
        word2index = pickle.load(f)
    with open(vocab_index2word, "rb") as f:
        index2word = pickle.load(f)

    print("  → Loading processed dataset...")
    qa_df = pd.read_csv(processed_file)
    qa_df["question_padded"] = qa_df["question_padded"].apply(ast.literal_eval)
    qa_df["answer_padded"] = qa_df["answer_padded"].apply(ast.literal_eval)

    print("  → Creating train/val/test splits...")
    # Recreate train/val/test split with same random state as notebook/evals
    X = np.array(qa_df["question_padded"].tolist())
    y = np.array(qa_df["answer_padded"].tolist())
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"  ✓ Data loaded successfully!")
    print(f"    - Vocabulary size: {len(word2index)}")
    print(f"    - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, word2index, index2word


def get_vocab_size(word2index):
    """Get vocabulary size"""
    return len(word2index)


def print_data_summary(X_train, y_train, X_val, y_val, X_test, y_test, word2index):
    """Print summary of loaded data"""
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"Vocabulary size: {len(word2index)}")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(
        f"Sequence length: {X_train.shape[1]} (questions), {y_train.shape[1]} (answers)"
    )
    print("Special tokens:", ["<pad>", "<unk>", "<sos>", "<eos>"])
    print("=" * 50 + "\n")
