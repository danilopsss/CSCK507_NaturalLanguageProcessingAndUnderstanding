# CSCK507 Natural Language Processing and Understanding

A Question-Answering chatbot system built for the CSCK507 Natural Language Processing course, featuring data preprocessing pipelines and a web-based chat interface.

## Overview

This project implements a chatbot system that processes question-answer datasets and provides a web interface for interaction. The system includes:

- Dataset: Question-Answer Dataset from: https://www.cs.cmu.edu/~ark/QA-data/
- Data preprocessing pipeline for Q&A dataset
- Vocabulary building and text tokenization
- Train/validation/test data splitting
- Seq2Seq model implementations (with and without attention)
- Consolidated model utilities for training and inference
- Model evaluation with BLEU scores
- FastAPI-based web application with WebSocket support
- Simple chat interface

## Installation

### Prerequisites
- Python 3.12 or higher
- uv package manager

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Download the spaCy English language model:
   ```bash
   uv run -- spacy download en_core_web_lg
   ```

## Usage

### Device Configuration

The project supports multiple compute devices for optimal performance:

- **Apple Silicon (M1/M2/M3/M4)**: Automatically uses MPS (Metal Performance Shaders) for GPU acceleration
- **NVIDIA GPUs**: Automatically uses CUDA when available
- **CPU**: Fallback option, works on all systems

**Device Selection Options:**

Note: Using evals.py as an example.

1. **Auto-detection (recommended)**: The system automatically selects the best available device
   ```bash
   uv run python src/evals.py
   ```

2. **Manual device selection**:
   ```bash
   # Force MPS (Apple Silicon)
   export TORCH_DEVICE=mps
   uv run python src/evals.py

   # Force CUDA (NVIDIA GPUs)
   export TORCH_DEVICE=cuda
   uv run python src/evals.py

   # Force CPU (debugging/compatibility)
   export TORCH_DEVICE=cpu
   uv run python src/evals.py
   ```

### Training Models

#### Train Seq2Seq Model Without Attention
```bash
uv run python src/models/Seq2Seq_without_attention.py
```

This will:
- Load preprocessed data using consolidated utilities
- Train a basic Seq2Seq model with GRU encoder/decoder
- Save the trained model to `models/chatbot_model_no_attention.pth`

#### Train Seq2Seq Model With Luong Attention
```bash
uv run python src/models/Seq2Seq_with_attention.py
```

This will:
- Load preprocessed data using consolidated utilities
- Train a Seq2Seq model with bidirectional encoder and Luong attention
- Save the trained model to `models/chatbot_model_with_attention.pth`
- Start an interactive chat session after training

### Running Model Evaluation
Evaluate both Seq2Seq models (with and without attention) on the test set:
```bash
uv run python src/evals.py
```

This will:
- Load the test dataset and vocabulary mappings using consolidated utilities
- Initialize both model architectures
- Generate predictions using BLEU scores and accuracy metrics
- Display sample predictions for manual evaluation
- Save detailed results to `results/evaluation_results_[timestamp].txt`

### Running the Web Application
Start the chatbot web server:
```bash
uv run uvicorn src.chatbot.main:app --host 0.0.0.0 --reload
```

The application will be available at `http://localhost:8000`

### Data Processing
The data preprocessing pipeline is available in the Jupyter notebook:
```
notebooks/preprocessing_and_split.ipynb
```

This notebook processes Q&A datasets from three sources (S08, S09, S10) and prepares them for model training. The processed data is saved to the `data/` directory as:
- `DATASET_PROCESSED.csv` - Combined and processed dataset
- `word2index.pkl` - Word to index mapping
- `index2word.pkl` - Index to word mapping

**Prerequisites:** Run this notebook first before training any models!

### Consolidated Model Utilities
The project uses consolidated model utilities located in `src/utils/model_utils.py`:

- **Data Loading**: `load_preprocessed_data()` - Centralized data loading function
- **Model Architectures**: `Encoder`, `Decoder`, `LuongAttention` classes
- **Training Functions**: `train()`, `chat()`, `top_k_sampling()`
- **Model Management**: `save_model()`, `load_model()`, `create_model_components()`
- **Utilities**: `tokenize()`, `collate_fn()`, device configuration

All training scripts now import from this consolidated module for consistency.

## Project Structure

```
├── README.md
├── pyproject.toml
├── uv.lock
├── data/                           # Processed data and vocabulary
│   ├── DATASET_PROCESSED.csv       # Combined and processed dataset
│   ├── word2index.pkl              # Word to index mapping
│   ├── index2word.pkl              # Index to word mapping
│   └── Datasets/
│       ├── S08/                    # Dataset from S08
│       ├── S09/                    # Dataset from S09
│       └── S10/                    # Dataset from S10
├── models/                         # Trained model files
│   ├── chatbot_model_no_attention.pth     # Seq2Seq without attention
│   └── chatbot_model_with_attention.pth   # Seq2Seq with Luong attention
├── notebooks/
│   └── preprocessing_and_split.ipynb      # Data preprocessing notebook
├── results/                        # Evaluation results
│   └── evaluation_results_*.txt    # Timestamped evaluation outputs
└── src/                           # Source code
    ├── evals.py                   # Model evaluation script
    ├── chatbot/
    │   ├── main.py                # FastAPI application
    │   ├── LICENSE
    │   └── app/
    │       ├── __init__.py
    │       ├── templates/         # HTML templates
    │       └── static/            # CSS and JavaScript files
    ├── models/
    │   ├── Chat.py                # Legacy chat interface
    │   ├── Seq2Seq_without_attention.py    # Basic Seq2Seq training
    │   └── Seq2Seq_with_attention.py       # Luong attention training
    └── utils/
        ├── device_config.py       # Device configuration utilities
        └── model_utils.py         # Consolidated model utilities
```

## Features

- **Data Preprocessing**: Text cleaning, tokenization, vocabulary building
- **Sequence Processing**: Padding, special token handling (`<sos>`, `<eos>`, `<pad>`, `<unk>`)
- **Data Splitting**: 80% train, 10% validation, 10% test
- **Device Optimization**: Automatic GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
- **Model Evaluation**: BLEU scores, accuracy metrics, and sample prediction analysis
- **Web Interface**: Real-time chat using WebSockets
- **Responsive Design**: Mobile-friendly chat interface

## Model Architecture

The project includes multiple model implementations:

1. **Seq2Seq Model Without Attention**: Located in `src/models/Seq2Seq_without_attention.py`
   - Basic sequence-to-sequence implementation with GRU encoder/decoder
   - Batch-first processing for efficient training
   - Saves model to `models/chatbot_model_no_attention.pth`

2. **Seq2Seq Model With Luong Attention**: Located in `src/models/Seq2Seq_with_attention.py`
   - Advanced sequence-to-sequence with bidirectional encoder
   - Luong attention mechanism for better context understanding
   - Top-k sampling for more diverse text generation
   - Interactive chat interface after training
   - Saves model to `models/chatbot_model_with_attention.pth`

3. **Consolidated Architecture**: All models use shared components from `src/utils/model_utils.py`
   - Consistent data loading and preprocessing
   - Unified device configuration (CPU/CUDA/MPS)
   - Optimized tensor operations for better performance

## Development Status

Current Status:
- Data preprocessing pipeline (complete)
- Seq2Seq model implementation without attention (complete)
- Seq2Seq model implementation with Luong attention (complete)
- Pre-trained chatbot model with utilities (complete)
- Processed dataset and vocabulary mappings (complete)
- Model evaluation with BLEU scores (complete)
- Multi-device support (CPU/CUDA/MPS) (complete)
- Web interface integration with the model (in progress)

## Performance Optimization

### Device Recommendations by Hardware:
- **Mac Mini M4**: Use MPS acceleration (auto-detected)
- **MacBook Pro/Air (Apple Silicon)**: Use MPS acceleration (auto-detected)
- **Windows/Linux with NVIDIA GPU**: Use CUDA acceleration (auto-detected)
- **Any system for debugging**: Use CPU with `export TORCH_DEVICE=cpu`

### Evaluation Metrics:
The evaluation script (`src/evals.py`) provides:
- **BLEU scores**: Industry-standard metric for sequence generation quality
- **Exact match accuracy**: Percentage of perfectly matched responses
- **Sample predictions**: Manual evaluation of model outputs
- **Comparative analysis**: Side-by-side performance of both models
- **Detailed logging**: Progress bars and timestamped results
- **Automatic result saving**: Results saved to `results/` directory with timestamps

### Code Quality:
- **Consolidated utilities**: Single source for model components
- **Consistent imports**: Proper path management across all scripts
- **Performance optimized**: Efficient tensor creation and data loading
- **Error handling**: Error messages and file existence checks
- **Unified device management**: Automatic detection of best available compute device