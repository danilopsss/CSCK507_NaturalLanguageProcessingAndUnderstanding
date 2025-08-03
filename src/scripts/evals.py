"""
Evaluation script for Seq2Seq chatbot models
Evaluates both models (with and without attention) on the test set using BLEU,
accuracy, and BERTScore metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
from typing import List, Dict, Tuple
import os
import sys
from tqdm import tqdm
from datetime import datetime
import csv

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import from consolidated model utilities
# Must happen after after path setup!
from utils.model_utils import (
    load_preprocessed_data,
    device,
    tokenize,
    SOS_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
except:
    pass

# BERTScore import
try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True
    print("BERTScore library loaded successfully")
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("BERTScore not available. Install with: uv add bert-score")

# MODEL ARCHITECTURES


# Seq2Seq without attention (from Seq2Seq_without_attention.py)
class EncoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden


class DecoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


class Seq2SeqNoAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

        hidden = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs


# Seq2Seq with attention (from Seq2Seq_with_attention.py)
class EncoderWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        return outputs, hidden


class LuongAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.squeeze(0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.attn(encoder_outputs)
        scores = torch.bmm(energy, hidden.unsqueeze(2))
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        return context.transpose(0, 1)


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim + enc_hid_dim * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(0)
        embedded = self.embedding(input_token)
        context = self.attn(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(torch.cat((output, context), dim=2).squeeze(0))
        return output, hidden


# EVALUATION UTILITIES


class EvaluationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    """Load and prepare test data using consolidated utilities"""
    print("Loading data using consolidated utilities...")

    # Use consolidated data loading function
    X_train, y_train, X_val, y_val, X_test, y_test, word2idx, idx2word = load_preprocessed_data()

    print(f"Data loaded successfully!")
    print(f" - Vocabulary size: {len(word2idx)}")
    print(f" - Test set size: {len(X_test)} samples")

    return X_test, y_test, word2idx, idx2word


def indices_to_sentence(indices: List[int], idx2word: Dict[int, str]) -> str:
    """Convert list of indices to sentence, removing special tokens"""
    words = []
    for idx in indices:
        word = idx2word.get(idx, UNK_TOKEN)
        if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            words.append(word)
        elif word == EOS_TOKEN:
            break
    return " ".join(words)


def calculate_bleu_score(references: List[List[str]], hypothesis: List[str]) -> float:
    """Calculate BLEU score for a single prediction"""
    if not hypothesis or not any(references):
        return 0.0

    smoothing_function = SmoothingFunction().method1
    try:
        return sentence_bleu(references, hypothesis, smoothing_function=smoothing_function)
    except:
        return 0.0


def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BERTScore metrics for predictions vs references

    Args:
        predictions: List of model-generated text
        references: List of ground truth text

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not BERTSCORE_AVAILABLE:
        print("BERTScore not available. Install with: uv add bert-score")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    if not predictions or not references:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    print("Calculating BERTScore metrics...")

    # Calculate BERTScore using default BERT model
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False, device=str(device))

    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item(),
    }


def evaluate_model_predictions(model, test_loader, word2idx, idx2word, model_name="Model"):
    """Generate predictions and calculate metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_sources = []

    print(f"\nGenerating predictions for {model_name}...")

    # Calculate total number of samples for progress bar
    total_samples = len(test_loader.dataset)

    with torch.no_grad():
        # Create progress bar for batches
        batch_pbar = tqdm(test_loader, desc=f"{model_name} Batches", unit="batch")

        for src, tgt in batch_pbar:
            src, tgt = src.to(device), tgt.to(device)
            batch_size = src.size(0)
            max_len = tgt.size(1)

            # Create progress bar for samples within batch
            sample_pbar = tqdm(range(batch_size), desc="Samples", unit="sample", leave=False)

            # Generate predictions
            for i in sample_pbar:
                src_single = src[i : i + 1]
                tgt_single = tgt[i : i + 1]

                # Generate prediction based on model type
                if (
                    hasattr(model, "encoder")
                    and hasattr(model.encoder, "rnn")
                    and model.encoder.rnn.bidirectional
                ):
                    # Model with attention
                    prediction = generate_prediction_with_attention(
                        model, src_single, word2idx, idx2word, max_len
                    )
                else:
                    # Model without attention
                    prediction = generate_prediction_no_attention(
                        model, src_single, word2idx, idx2word, max_len
                    )

                # Convert target to sentence
                target_indices = tgt_single[0].cpu().numpy()
                target_sentence = indices_to_sentence(target_indices, idx2word)

                # Convert source to sentence
                source_indices = src_single[0].cpu().numpy()
                source_sentence = indices_to_sentence(source_indices, idx2word)

                all_predictions.append(prediction)
                all_targets.append(target_sentence)
                all_sources.append(source_sentence)

                # Update sample progress bar
                sample_pbar.set_postfix(
                    {
                        "Predictions": len(all_predictions),
                        "Last_pred": prediction[:30] + "..."
                        if len(prediction) > 30
                        else prediction,
                    }
                )

            # Update batch progress bar
            batch_pbar.set_postfix(
                {
                    "Total_predictions": len(all_predictions),
                    "Progress": f"{len(all_predictions)}/{total_samples}",
                }
            )

    return all_predictions, all_targets, all_sources


def generate_prediction_no_attention(model, src, word2idx, idx2word, max_len=20):
    """Generate prediction for model without attention"""
    hidden = model.encoder(src)
    input_token = torch.tensor([word2idx[SOS_TOKEN]], device=device)

    prediction = []
    for _ in range(max_len):
        output, hidden = model.decoder(input_token, hidden)
        next_token = output.argmax(1).item()

        if next_token == word2idx[EOS_TOKEN]:
            break

        word = idx2word.get(next_token, UNK_TOKEN)
        if word not in [PAD_TOKEN, SOS_TOKEN]:
            prediction.append(word)

        input_token = torch.tensor([next_token], device=device)

    return " ".join(prediction)


def generate_prediction_with_attention(model, src, word2idx, idx2word, max_len=20):
    """Generate prediction for model with attention"""
    # Transpose for attention model (expects seq_len, batch_size)
    src_transposed = src.transpose(0, 1)
    enc_outputs, enc_hidden = model.encoder(src_transposed)

    dec_input = torch.tensor([word2idx[SOS_TOKEN]], device=device)
    dec_hidden = enc_hidden

    prediction = []
    for _ in range(max_len):
        dec_output, dec_hidden = model.decoder(dec_input, dec_hidden, enc_outputs)
        next_token = dec_output.argmax(1).item()

        if next_token == word2idx[EOS_TOKEN]:
            break

        word = idx2word.get(next_token, UNK_TOKEN)
        if word not in [PAD_TOKEN, SOS_TOKEN]:
            prediction.append(word)

        dec_input = torch.tensor([next_token], device=device)

    return " ".join(prediction)


def calculate_metrics(
    predictions: List[str], targets: List[str], sources: List[str] = None, model_name: str = ""
) -> Dict[str, float]:
    """Calculate evaluation metrics including BLEU, accuracy, and BERTScore"""
    bleu_scores = []
    exact_matches = 0
    empty_predictions = 0

    print("\nCalculating evaluation metrics...")

    # Setup error logging for empty predictions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    error_log_path = os.path.join(project_root, "results", "questions_errors.txt")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

    # Create progress bar for metric calculation
    if sources:
        metric_pbar = tqdm(
            zip(predictions, targets, sources),
            total=len(predictions),
            desc="Computing BLEU & Accuracy",
            unit="sample",
        )
    else:
        metric_pbar = tqdm(
            zip(predictions, targets),
            total=len(predictions),
            desc="Computing BLEU & Accuracy",
            unit="sample",
        )

    for i, items in enumerate(metric_pbar):
        if sources:
            pred, target, source = items
        else:
            pred, target = items
            source = f"Question {i + 1}"  # fallback if no sources provided

        # Count empty predictions and log problematic questions
        if not pred.strip():
            empty_predictions += 1
            # Log the question that resulted in empty answer
            with open(error_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Model: {model_name}\n")
                f.write(f"Question: {source}\n")
                f.write(f"Expected: {target}\n")
                f.write(f"Got: <EMPTY>\n")
                f.write("-" * 50 + "\n")

        # BLEU score
        pred_tokens = pred.split() if pred else []
        target_tokens = target.split() if target else []

        if target_tokens:  # Only calculate if target is not empty
            bleu_score = calculate_bleu_score([target_tokens], pred_tokens)
            bleu_scores.append(bleu_score)

        # Exact match
        if pred.strip().lower() == target.strip().lower():
            exact_matches += 1

        # Update progress bar with current metrics
        current_avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        current_accuracy = exact_matches / (metric_pbar.n + 1)
        metric_pbar.set_postfix(
            {
                "Avg_BLEU": f"{current_avg_bleu:.4f}",
                "Accuracy": f"{current_accuracy:.4f}",
                "Empty": empty_predictions,
            }
        )

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    accuracy = exact_matches / len(predictions) if predictions else 0.0

    # Calculate BERTScore metrics
    bert_metrics = calculate_bert_score(predictions, targets)

    return {
        "bleu_score": avg_bleu,
        "accuracy": accuracy,
        "total_samples": len(predictions),
        "empty_predictions": empty_predictions,
        "empty_prediction_rate": empty_predictions / len(predictions) if predictions else 0.0,
        "bert_precision": bert_metrics["bert_precision"],
        "bert_recall": bert_metrics["bert_recall"],
        "bert_f1": bert_metrics["bert_f1"],
    }


def load_model_no_attention(vocab_size, word2idx):
    """Load and initialize model without attention"""
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512

    # Check if model file exists first - use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "weights", "chatbot_model_no_attention.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: Model file '{model_path}' not found!\n\n"
            f"The Seq2Seq model without attention has not been trained yet.\n"
            f"Please train the model first by running the training script:\n"
            f"'src/models/Seq2Seq_without_attention.py'\n\n"
            f"Cannot evaluate an untrained model - evaluation stopped."
        )

    encoder = EncoderNoAttention(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE).to(device)
    decoder = DecoderNoAttention(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE).to(device)
    model = Seq2SeqNoAttention(encoder, decoder).to(device)

    # Load the saved model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model without attention from {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Could not load model from '{model_path}': {e}\n\n"
            f"The model file exists but appears to be corrupted or incompatible.\n"
            f"Please retrain the model or check the model architecture."
        )

    return model


def load_model_with_attention(vocab_size, word2idx):
    """Load and initialize model with attention"""
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256

    # Check if model file exists first - use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "weights", "chatbot_model_with_attention.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: Model file '{model_path}' not found!\n\n"
            f"The Seq2Seq model with attention has not been trained yet.\n"
            f"Please train the model first by running the training script:\n"
            f"'src/models/Seq2Seq_with_attention.py'\n\n"
            f"Cannot evaluate an untrained model - evaluation stopped."
        )

    encoder = EncoderWithAttention(vocab_size, ENC_EMB_DIM, ENC_HID_DIM).to(device)
    decoder = DecoderWithAttention(vocab_size, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM).to(device)

    # Load the saved model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        print(f"Loaded model with attention from {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Could not load model from '{model_path}': {e}\n\n"
            f"The model file exists but appears to be corrupted or incompatible.\n"
            f"Please retrain the model or check the model architecture."
        )

    # Create a wrapper class to mimic the model interface
    class Seq2SeqWithAttention:
        def __init__(self, encoder, decoder):
            self.encoder = encoder
            self.decoder = decoder

        def eval(self):
            self.encoder.eval()
            self.decoder.eval()

    return Seq2SeqWithAttention(encoder, decoder)


def print_sample_predictions(
    predictions: List[str],
    targets: List[str],
    sources: List[str],
    model_name: str,
    n_samples: int = 20,
):
    """Print sample predictions for manual evaluation"""
    print(f"\n--- Sample Predictions for {model_name} ---")
    for i in range(min(n_samples, len(predictions))):
        print(f"Question:   {sources[i]}")
        print(f"Target:     {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 50)


def main():
    print("=== Seq2Seq Chatbot Model Evaluation ===")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading test data...")
    X_test, y_test, word2idx, idx2word = load_data()
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    # Create test dataset and dataloader
    test_dataset = EvaluationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load models
    print("\nLoading models...")
    model_no_attention = load_model_no_attention(vocab_size, word2idx)
    model_with_attention = load_model_with_attention(vocab_size, word2idx)

    # Evaluate model without attention
    print("\n" + "=" * 60)
    print("EVALUATING SEQ2SEQ MODEL WITHOUT ATTENTION")
    print("=" * 60)

    predictions_no_att, targets_no_att, sources_no_att = evaluate_model_predictions(
        model_no_attention, test_loader, word2idx, idx2word, "Seq2Seq without Attention"
    )

    metrics_no_att = calculate_metrics(
        predictions_no_att, targets_no_att, sources_no_att, "Seq2Seq without Attention"
    )

    print(f"\n--- Results for Seq2Seq without Attention ---")
    print(f"BLEU Score:      {metrics_no_att['bleu_score']:.4f}")
    print(f"Accuracy:        {metrics_no_att['accuracy']:.4f}")
    print(f"BERT Precision:  {metrics_no_att['bert_precision']:.4f}")
    print(f"BERT Recall:     {metrics_no_att['bert_recall']:.4f}")
    print(f"BERT F1:         {metrics_no_att['bert_f1']:.4f}")
    print(
        f"Empty Responses: {metrics_no_att['empty_predictions']} ({metrics_no_att['empty_prediction_rate']:.1%})"
    )
    print(f"Total Samples:   {metrics_no_att['total_samples']}")

    print_sample_predictions(
        predictions_no_att, targets_no_att, sources_no_att, "Seq2Seq without Attention"
    )

    # Evaluate model with attention
    print("\n" + "=" * 60)
    print("EVALUATING SEQ2SEQ MODEL WITH LUONG ATTENTION")
    print("=" * 60)

    predictions_with_att, targets_with_att, sources_with_att = evaluate_model_predictions(
        model_with_attention,
        test_loader,
        word2idx,
        idx2word,
        "Seq2Seq with Attention",
    )

    metrics_with_att = calculate_metrics(
        predictions_with_att, targets_with_att, sources_with_att, "Seq2Seq with Attention"
    )

    print(f"\n--- Results for Seq2Seq with Luong Attention ---")
    print(f"BLEU Score:      {metrics_with_att['bleu_score']:.4f}")
    print(f"Accuracy:        {metrics_with_att['accuracy']:.4f}")
    print(f"BERT Precision:  {metrics_with_att['bert_precision']:.4f}")
    print(f"BERT Recall:     {metrics_with_att['bert_recall']:.4f}")
    print(f"BERT F1:         {metrics_with_att['bert_f1']:.4f}")
    print(
        f"Empty Responses: {metrics_with_att['empty_predictions']} ({metrics_with_att['empty_prediction_rate']:.1%})"
    )
    print(f"Total Samples:   {metrics_with_att['total_samples']}")

    print_sample_predictions(
        predictions_with_att,
        targets_with_att,
        sources_with_att,
        "Seq2Seq with Attention",
    )

    # Comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'No Attention':<15} {'With Attention':<15} {'Improvement':<15}")
    print("-" * 65)
    print(
        f"{'BLEU Score':<20} {metrics_no_att['bleu_score']:<15.4f} {metrics_with_att['bleu_score']:<15.4f} {metrics_with_att['bleu_score'] - metrics_no_att['bleu_score']:<15.4f}"
    )
    print(
        f"{'Accuracy':<20} {metrics_no_att['accuracy']:<15.4f} {metrics_with_att['accuracy']:<15.4f} {metrics_with_att['accuracy'] - metrics_no_att['accuracy']:<15.4f}"
    )
    print(
        f"{'BERT Precision':<20} {metrics_no_att['bert_precision']:<15.4f} {metrics_with_att['bert_precision']:<15.4f} {metrics_with_att['bert_precision'] - metrics_no_att['bert_precision']:<15.4f}"
    )
    print(
        f"{'BERT Recall':<20} {metrics_no_att['bert_recall']:<15.4f} {metrics_with_att['bert_recall']:<15.4f} {metrics_with_att['bert_recall'] - metrics_no_att['bert_recall']:<15.4f}"
    )
    print(
        f"{'BERT F1':<20} {metrics_no_att['bert_f1']:<15.4f} {metrics_with_att['bert_f1']:<15.4f} {metrics_with_att['bert_f1'] - metrics_no_att['bert_f1']:<15.4f}"
    )

    # Save results
    results = {
        "seq2seq_no_attention": metrics_no_att,
        "seq2seq_with_attention": metrics_with_att,
        "comparison": {
            "bleu_improvement": metrics_with_att["bleu_score"] - metrics_no_att["bleu_score"],
            "accuracy_improvement": metrics_with_att["accuracy"] - metrics_no_att["accuracy"],
            "bert_precision_improvement": metrics_with_att["bert_precision"]
            - metrics_no_att["bert_precision"],
            "bert_recall_improvement": metrics_with_att["bert_recall"]
            - metrics_no_att["bert_recall"],
            "bert_f1_improvement": metrics_with_att["bert_f1"] - metrics_no_att["bert_f1"],
        },
    }

    # Generate timestamp for results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.txt"
    csv_filename = f"evaluation_results_{timestamp}.csv"

    # Get absolute path to results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, results_filename)
    csv_path = os.path.join(results_dir, csv_filename)

    print(f"\nEvaluation completed! Results saved to {results_path}")
    print(f"CSV results saved to {csv_path}")

    # Check if any empty predictions were logged
    error_log_path = os.path.join(results_dir, "questions_errors.txt")
    if os.path.exists(error_log_path):
        print(f"Questions resulting in empty answers logged to: {error_log_path}")

    # Save results to CSV for easy analysis and plotting
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "model",
                "bleu_score",
                "accuracy",
                "bert_precision",
                "bert_recall",
                "bert_f1",
                "empty_predictions",
                "empty_prediction_rate",
                "timestamp",
            ]
        )
        writer.writerow(
            [
                "no_attention",
                metrics_no_att["bleu_score"],
                metrics_no_att["accuracy"],
                metrics_no_att["bert_precision"],
                metrics_no_att["bert_recall"],
                metrics_no_att["bert_f1"],
                metrics_no_att["empty_predictions"],
                metrics_no_att["empty_prediction_rate"],
                timestamp,
            ]
        )
        writer.writerow(
            [
                "with_attention",
                metrics_with_att["bleu_score"],
                metrics_with_att["accuracy"],
                metrics_with_att["bert_precision"],
                metrics_with_att["bert_recall"],
                metrics_with_att["bert_f1"],
                metrics_with_att["empty_predictions"],
                metrics_with_att["empty_prediction_rate"],
                timestamp,
            ]
        )

    # Save detailed results
    with open(results_path, "w") as f:
        f.write("=== Seq2Seq Chatbot Model Evaluation Results ===\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device Used: {device}\n")
        f.write(f"Test set size: {len(X_test)} samples\n")
        f.write(f"Vocabulary size: {vocab_size}\n\n")

        f.write("Seq2Seq without Attention:\n")
        f.write(f"  BLEU Score: {metrics_no_att['bleu_score']:.4f}\n")
        f.write(f"  Accuracy: {metrics_no_att['accuracy']:.4f}\n")
        f.write(f"  BERT Precision: {metrics_no_att['bert_precision']:.4f}\n")
        f.write(f"  BERT Recall: {metrics_no_att['bert_recall']:.4f}\n")
        f.write(f"  BERT F1: {metrics_no_att['bert_f1']:.4f}\n")
        f.write(
            f"  Empty Responses: {metrics_no_att['empty_predictions']} ({metrics_no_att['empty_prediction_rate']:.1%})\n\n"
        )

        f.write("Seq2Seq with Luong Attention:\n")
        f.write(f"  BLEU Score: {metrics_with_att['bleu_score']:.4f}\n")
        f.write(f"  Accuracy: {metrics_with_att['accuracy']:.4f}\n")
        f.write(f"  BERT Precision: {metrics_with_att['bert_precision']:.4f}\n")
        f.write(f"  BERT Recall: {metrics_with_att['bert_recall']:.4f}\n")
        f.write(f"  BERT F1: {metrics_with_att['bert_f1']:.4f}\n")
        f.write(
            f"  Empty Responses: {metrics_with_att['empty_predictions']} ({metrics_with_att['empty_prediction_rate']:.1%})\n\n"
        )

        f.write("Improvements (With Attention - Without Attention):\n")
        f.write(f"  BLEU Score: {results['comparison']['bleu_improvement']:.4f}\n")
        f.write(f"  Accuracy: {results['comparison']['accuracy_improvement']:.4f}\n")
        f.write(f"  BERT Precision: {results['comparison']['bert_precision_improvement']:.4f}\n")
        f.write(f"  BERT Recall: {results['comparison']['bert_recall_improvement']:.4f}\n")
        f.write(f"  BERT F1: {results['comparison']['bert_f1_improvement']:.4f}\n")


if __name__ == "__main__":
    main()
