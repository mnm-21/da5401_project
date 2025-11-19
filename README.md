# DA5401 Data Challenge: Metric Score Prediction

**Author:** Mayank Chandak  
**Roll Number:** ME22B224

## Project Overview

This project implements a contrastive learning-based approach for predicting evaluation metric scores for AI-generated responses. The solution uses a multi-stage pipeline combining:
- **Sentence embeddings** (Google Gemma-300M) for text representation
- **Contrastive learning** for metric-text alignment
- **Stratified cross-validation** for robust model selection
- **Post-processing heuristics** for score calibration

The best submission (`submission_custom_round.csv`) achieved the lowest error on the public leaderboard using a frozen contrastive routing model with specialized post-processing.

## Project Structure

```
da5401_project/
├── data/                          # Input data files
│   ├── train_data.json           # Training set (5000 samples)
│   ├── test_data.json             # Test set (3638 samples)
│   ├── metric_names.json          # List of 145 evaluation metrics
│   ├── metric_name_embeddings.npy # Pre-computed metric embeddings (145, 768)
│   └── sample_submission.csv      # Submission template
│
├── saved_embeddings/              # Cached text embeddings (for faster iteration)
│   ├── train_text_embeddings.npy  # Train set embeddings (5000, 768)
│   └── test_text_embeddings.npy   # Test set embeddings (3638, 768)
│
├── phase0.ipynb                   # Data exploration & baseline models
├── phase1_best_round.ipynb        # **FROZEN** best model (reproduces submission_custom_round.csv)
├── phase1_exps.ipynb              # Experimental joint training variants
│
├── submission_custom_round.csv    # **BEST SUBMISSION** (leaderboard winner)
├── submission_round.csv           # Alternative submission with different rounding
├── submission.csv                 # Raw predictions (unrounded)
├── submission_rounded.csv         # Simple rounded predictions
│
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## File Descriptions

### Notebooks

- **`phase0.ipynb`**: Initial data exploration, baseline models (LightGBM), and feature engineering. Includes:
  - Data schema normalization
  - Metric name to embedding index mapping
  - Text length and script heuristics
  - Sentence embedding generation
  - Baseline LightGBM with isotonic calibration

- **`phase1_best_round.ipynb`**: **FROZEN** notebook that reproduces the best submission. This is the exact code used to generate `submission_custom_round.csv`. Key features:
  - Contrastive learning for metric-text matching
  - 10-fold stratified cross-validation
  - Router + expert architecture (low/high score specialists)
  - Custom rounding heuristic: `ceil(x)` if `x <= 7`, else `round(x)`
  - All hyperparameters and random seeds are fixed for reproducibility

- **`phase1_exps.ipynb`**: Experimental variants exploring joint training of contrastive + regression + ranking losses. Used for ablation studies but not the final submission.

### Data Files

- **`data/train_data.json`**: Training data with columns:
  - `system_prompt`: System instruction text
  - `user_prompt`: User query text
  - `response`: Model-generated response
  - `metric_name`: Evaluation metric name (one of 145 metrics)
  - `score`: Ground truth score (0-10, continuous)

- **`data/test_data.json`**: Test data (same columns except `score`)

- **`data/metric_names.json`**: List of 145 evaluation metric names

- **`data/metric_name_embeddings.npy`**: Pre-computed embeddings for all metrics (shape: 145 × 768)

- **`data/sample_submission.csv`**: Submission template with `ID` and `score` columns

### Submission Files

- **`submission_custom_round.csv`**: **BEST SUBMISSION** - Uses custom rounding: `ceil(x)` for scores ≤ 7, `round(x)` otherwise. This was the winning entry on the leaderboard.

- **`submission_round.csv`**: Alternative submission with the same rounding heuristic (from `phase1_best_round.ipynb`)

- **`submission.csv`**: Raw floating-point predictions (no rounding)

- **`submission_rounded.csv`**: Simple rounded predictions (`round(x)` for all scores)

### Cached Embeddings

The `saved_embeddings/` directory contains pre-computed text embeddings to avoid recomputation. These are generated automatically on first run and cached for subsequent executions.

## Environment Setup

### Option 1: Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n da5401 python=3.10

# Activate environment
conda activate da5401

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import sentence_transformers; print('Installation successful!')"
```

## Reproduction Steps for Best Submission

To reproduce `submission_custom_round.csv` exactly:

### Step 1: Environment Setup
Follow the environment setup instructions above.

### Step 2: Data Preparation
Ensure all data files are in the `data/` directory:
- `train_data.json`
- `test_data.json`
- `metric_names.json`
- `metric_name_embeddings.npy`
- `sample_submission.csv`

### Step 3: Run the Frozen Notebook

1. Open `phase1_best_round.ipynb` in Jupyter:
   ```bash
   jupyter notebook phase1_best_round.ipynb
   ```
   Or use JupyterLab:
   ```bash
   jupyter lab phase1_best_round.ipynb
   ```

2. **Important**: Execute cells **sequentially** and **do not modify any hyperparameters or random seeds**. The notebook is intentionally frozen to ensure reproducibility.

3. The notebook will:
   - Load and format training/test data
   - Generate or load cached text embeddings (first run takes ~10-15 minutes for encoding)
   - Train 10 contrastive models using stratified 10-fold CV
   - Select the best fold based on validation MSE
   - Generate test predictions using the best model
   - Apply custom rounding: `ceil(x)` if `x <= 7`, else `round(x)`
   - Save `submission_round.csv` (which matches `submission_custom_round.csv`)

### Step 4: Verify Output

After execution, check that `submission_round.csv` is generated with:
- 3638 rows (matching test set size)
- Scores in range [0, 10]
- Same file as `submission_custom_round.csv` (you can verify with `diff` or checksum)

### Expected Runtime

- **First run** (with embedding generation): ~20-30 minutes on CPU
- **Subsequent runs** (using cached embeddings): ~10-15 minutes on CPU
- Training time per fold: ~1-2 minutes
- Total 10-fold CV: ~10-20 minutes

### Key Hyperparameters (Frozen)

- `SEED = 42` (all random seeds)
- `BATCH_SIZE_ENCODE = 2`
- `TEXT_BATCH_SIZE_TRAIN = 128`
- `NEG_K = 2` (negative samples per anchor)
- `CONTRASTIVE_EPOCHS = 20`
- `N_FOLDS = 10`
- `EMBEDDING_MODEL = "google/embeddinggemma-300m"`
- `LOW_SCORE_THRESH = 7.0` (routing threshold)
- `FORCE_CPU = True` (for reproducibility)

## Methodology

### Model Architecture

1. **Text Embedding**: Google Gemma-300M (`google/embeddinggemma-300m`) generates 768-dimensional embeddings for each (system, user, response) triple.

2. **Contrastive Router**: A neural network learns to match text embeddings with their corresponding metric embeddings. Trained using contrastive loss with negative sampling.

3. **Expert Models**: Two specialized regression heads:
   - **Low Expert**: Handles samples with router scores ≤ 7.0
   - **High Expert**: Handles samples with router scores > 7.0

4. **Training**: 10-fold stratified cross-validation ensures each fold has a balanced distribution of target scores.

5. **Inference**: 
   - Router scores each test sample
   - Routes to appropriate expert based on threshold
   - Experts predict final scores
   - Post-processing: `ceil(x)` if `x <= 7`, else `round(x)`

### Why This Approach?

- **Contrastive learning** captures semantic relationships between text and metrics
- **Stratified CV** prevents validation bias from score distribution skew
- **Expert routing** allows specialized models for different score ranges
- **Custom rounding** addresses systematic under-prediction for low scores (≤7)

## Troubleshooting

### Common Issues

1. **Out of Memory**: If you encounter memory errors:
   - Reduce `TEXT_BATCH_SIZE_TRAIN` (default: 128)
   - Reduce `BATCH_SIZE_ENCODE` (default: 2)
   - Ensure sufficient RAM (recommended: 8GB+)

2. **Embedding Model Download**: First run will download `google/embeddinggemma-300m` (~600MB). Ensure stable internet connection.

3. **File Not Found Errors**: Verify all files in `data/` directory exist and paths are correct.

4. **Different Results**: Ensure:
   - All random seeds are set to 42
   - `FORCE_CPU = True` (GPU non-determinism can cause variance)
   - No modifications to hyperparameters
   - Same Python/PyTorch versions as in `requirements.txt`

### Performance Notes

- Training on CPU is intentionally slow but ensures reproducibility
- For faster iteration, you can reduce `N_FOLDS` or `CONTRASTIVE_EPOCHS` (but this won't reproduce the exact submission)
- Cached embeddings significantly speed up subsequent runs

## Results Summary

- **Best Model**: Contrastive router + expert architecture
- **Best Submission**: `submission_custom_round.csv`
- **CV Strategy**: 10-fold stratified cross-validation
- **Best Fold**: Selected based on lowest validation MSE
- **Post-processing**: Custom rounding heuristic for scores ≤ 7

## Citation

If you use this code, please cite:

```
Mayank Chandak (ME22B224)
DA5401 Data Challenge Submission
2025
```

## License

This project is for academic purposes as part of the DA5401 course.

## Contact

For questions or issues, please contact: Mayank Chandak (ME22B224)
