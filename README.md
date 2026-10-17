# Velocity Transcript Project

This repository implements an end-to-end deep learning pipeline designed for recognizing and transcribing human hand gestures, such as sign language signs (the "velocity transcript"). The system processes video inputs to extract detailed 3D landmark coordinates and corresponding velocity features, which are then used to train a classification model.

## 🚀 Architecture Overview

The project is structured into several distinct phases that handle the data lifecycle:

1.  **Data Acquisition:** Recording raw hand movement videos from the webcam or pre-recorded sources.
2.  **Feature Extraction (Preprocessing):** Processing video frames to extract standardized sequences of hand landmarks and calculating velocity features.
3.  **Data Structuring & Splitting:** Organizing extracted clips into clean training and testing sets, suitable for sequence modeling.
4.  **Model Training:** Training a recurrent neural network (RNN/LSTM) classifier on the structured feature data.
5.  **Inference:** Running the trained model against new video streams for real-time or batch transcription.

## 🛠️ Setup and Dependencies

### Prerequisites
*   Python 3.12+
*   OpenCV (`opencv-python`) with proper dependencies for video I/O.
*   PyTorch (`torch`) (GPU support highly recommended).
*   MediaPipe library for hand landmark detection.

### Installation
First, set up a virtual environment and install all required libraries using the provided `requirements.txt`:

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt
```

> **⚠️ Critical Note on PyTorch:** If you intend to run this project with a GPU, please ensure your `torch` installation matches your CUDA version for optimal performance (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).

## ⚙️ Workflow Guide

The pipeline should be executed sequentially to generate a functional model:

### Step 1: Record Own Clips
Capture raw video data using the webcam for individual signs.

```bash
python record_own_clips.py
# Follow prompts and capture clips for training.
```

### Step 2: Preprocess Data & Extract Features
Process all recorded or source videos to extract standardized landmark sequences, including velocity information. This populates the `data/processed` directory.

*   **Single File:** Process one clip at a time.
    ```bash
    python preprocess_single.py <path/to/video>
    ```
*   **Batch Processing:** Process multiple source clips or directories.
    ```bash
    # Example of batch preprocessing:
    python batch_preprocess.py 
    ```

### Step 3: Split Data for Training
Group the extracted features into clean training and testing sets, managing overlaps and ensuring representative data splits.

```bash
python session_split.py
# This creates necessary index files in data/processed
```

### Step 4: Train the Model (Baseline or Combined)
Train the LSTM classifier using the structured data. Start with the baseline model before attempting more complex combinations.

```bash
# Train Simple LSTM Classifier
python train_baseline.py 
```

### Step 5: Evaluate Performance & Test on Own Data
Test the trained model's performance and robustness against held-out validation/test sets.

*   **General Evaluation:**
    ```bash
    python evaluate_on_own.py
    ```
*   **Video Testing (Single File):** Run landmark extraction on a video file for quality checks, independent of the ML pipeline.
    ```bash
    python video_landmark_test.py "/path/to/video.MOV"
    ```

### Step 6: Live Inference / Prediction
Use the finalized, trained model to transcribe gestures from real-time webcam feeds or new videos.

*   **Real-Time Webcam Inference:**
    ```bash
    python live_inference.py
    # The script will connect to the camera and display predictions.
    ```

## 🧩 Key Components & Details

| Component | Description | Usage Files |
| :--- | :--- | :--- |
| **MediaPipe** | Handles real-time detection of hand landmarks (21 points per hand). | `batch_scan.py`, `hand_test.py` |
| **Feature Engineering** | Calculates velocity features and normalizes landmark coordinates for robust input data. | `preprocess_single.py` |
| **Model Training** | Implements the core LSTM classifier to map sequences of features to specific signs. | `train_baseline.py`, `train_combined.py` |
| **Data Management** | Manages the lifecycle from raw video -> extracted landmarks -> split indices. | `batch_preprocess.py`, `session_split.py` |

## 📚 Directory Structure

*   `config.py`: Global configuration variables (e.g., folder structure, sequence lengths).
*   `data/processed`: Directory where all feature-engineered NumPy data (`X.npy`, `y.npy`) and model checkpoints are saved.
*   `models/`: Location where trained PyTorch model weights (`*.pt`) are stored.