# 🫁 Chest Disease Identification System (CDIS)

> **Classifying Adenocarcinoma Cancer vs Normal Chest CT Scans using VGG16 Transfer Learning**
>
> An end-to-end MLOps system that accepts a chest CT scan image and predicts whether it shows **Adenocarcinoma Cancer** or a **Normal** scan — built on a frozen VGG16 backbone, a 4-stage DVC-orchestrated pipeline, MLflow experiment tracking via DagsHub, and automated CI/CD deployment to AWS.

---

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.12-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/API-Flask-black?logo=flask)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/Pipeline-DVC-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow%20%2B%20DagsHub-0194E2)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20EC2-orange?logo=amazonaws)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want the visual overview first?** The deck covers the clinical problem, VGG16 architecture, DVC pipeline, MLflow tracking, and deployment in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1S4CmZegZsCwnXBm-8WrL6csShNLLyhXn/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Data & Classes](#6-data--classes) |
| 7 | [ML Pipeline — Step by Step](#7-ml-pipeline--step-by-step) |
| 8 | [VGG16 Transfer Learning Architecture](#8-vgg16-transfer-learning-architecture) |
| 9 | [Model Performance](#9-model-performance) |
| 10 | [Flask Web Application](#10-flask-web-application) |
| 11 | [How to Replicate — Full Setup Guide](#11-how-to-replicate--full-setup-guide) |
| 12 | [Running the Application](#12-running-the-application) |
| 13 | [CI/CD & Cloud Deployment](#13-cicd--cloud-deployment) |
| 14 | [Business Applications & Other Domains](#14-business-applications--other-domains) |
| 15 | [How to Improve This Project](#15-how-to-improve-this-project) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Business Problem

### What problem are we solving?

Lung cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate diagnosis is critical — patients diagnosed at an early stage have significantly better survival outcomes. Chest CT scans are the gold-standard imaging modality for detecting lung abnormalities, but radiological interpretation requires specialist expertise that is scarce, expensive, and subject to human fatigue and variability.

Adenocarcinoma is the most common type of lung cancer and is often detected on chest CT scans. Distinguishing adenocarcinoma from normal tissue requires pattern recognition across complex 3D image data, making it an ideal candidate for deep learning automation.

Core pain points:

- 🏥 **Radiologist shortage** — the global deficit of radiologists means CT scans queue for days before review, delaying diagnosis and treatment
- 🔬 **Inter-reader variability** — different radiologists can reach different conclusions on the same CT scan, creating clinical uncertainty
- 📊 **Screening volume** — mass lung cancer screening programmes generate thousands of scans per centre per year — impossible to review manually at speed
- 💊 **Late-stage diagnosis** — without rapid automated triage, high-risk cases may not be prioritised, and early-stage cancers are missed

### What does CDIS answer?

> *"Given a chest CT scan image, does it show Adenocarcinoma Cancer or Normal tissue?"*

This is a **binary image classification** problem using deep learning. The system provides a class label and can be integrated into radiology workflows as a decision-support tool.

> ⚠️ **Important note:** This system is designed as a research and educational demonstration. It is **not a medical device** and must not be used for clinical diagnosis without appropriate validation, regulatory approval, and clinical oversight.

### Objectives

1. Build a binary CNN classifier (Adenocarcinoma vs Normal) from chest CT images using transfer learning
2. Use VGG16 pretrained on ImageNet as the frozen backbone — adding a custom classification head
3. Orchestrate the full training pipeline using **DVC** — enabling `dvc repro` reproducibility
4. Track all experiments, parameters, and metrics with **MLflow** via **DagsHub**
5. Serve predictions via a **Flask** web interface accepting base64-encoded CT scan images
6. Deploy automatically to **AWS EC2** via Docker and GitHub Actions CI/CD

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | Binary classification: Adenocarcinoma Cancer vs Normal chest CT scan |
| **Dataset** | Chest CT Scan dataset (~49 MB, 343 image files) from Google Drive |
| **Classes** | `0` = Adenocarcinoma Cancer · `1` = Normal |
| **Model** | VGG16 (ImageNet pretrained, all layers frozen) + custom Dense head |
| **Input size** | 224 × 224 × 3 (RGB) |
| **Output** | Softmax over 2 classes |
| **Optimizer** | SGD (`lr=0.01`) |
| **Loss** | CategoricalCrossentropy |
| **Augmentation** | Rotation, flip, shift, shear, zoom (when `AUGMENTATION=True`) |
| **Train/Val split** | 80 / 20 (training), 70 / 30 (evaluation) |
| **Batch size** | 16 |
| **Epochs** | 1 (demo; increase for production) |
| **Pipeline** | DVC (4 stages defined in `dvc.yaml`) |
| **Experiment tracking** | MLflow 2.2.2 → DagsHub remote |
| **Model format** | Keras `.h5` (~59 MB) |
| **Web framework** | Flask on port 8080 |
| **CI/CD** | GitHub Actions → Docker → AWS ECR → AWS EC2 |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.8+ | Core language across all pipeline stages |
| **Deep Learning** | TensorFlow 2.12.0 / Keras | VGG16 model, ImageDataGenerator, training loop |
| **Transfer Learning** | VGG16 (ImageNet) | Pretrained backbone for feature extraction |
| **Pipeline Orchestration** | DVC (Data Version Control) | Defines and reproduces 4-stage ML pipeline via `dvc.yaml` |
| **Experiment Tracking** | MLflow 2.2.2 | Logs params, metrics, and model artifacts per run |
| **Remote Tracking** | DagsHub | Hosts MLflow tracking server for team collaboration |
| **Data Download** | gdown | Downloads CT scan dataset from Google Drive |
| **Config Management** | PyYAML + python-box | Reads `config.yaml` and `params.yaml` as dot-accessible objects |
| **Schema enforcement** | `ensure` library | Validates function argument types at runtime |
| **Web Framework** | Flask + Flask-CORS | Serves `/predict` (base64 image), `/train`, and home routes |
| **Image I/O** | base64 (stdlib) | Encodes/decodes CT scan images for JSON API transport |
| **Serialisation** | joblib | Binary object save/load |
| **Containerisation** | Docker (`python:3.9-slim-buster`) | Packages app + AWS CLI |
| **Cloud Compute** | AWS EC2 (Ubuntu) | Hosts Flask app |
| **Container Registry** | AWS ECR | Stores Docker images |
| **CI/CD** | GitHub Actions (3-job pipeline) | Build → push → deploy on every `main` push |
| **Logging** | Python `logging` | Dual-output: file (`logs/running_logs.log`) + stdout |

---

## 4. High-Level Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│  Google Drive  ──(gdown)──►  artifacts/data_ingestion/data.zip     │
│                                        │                            │
│                              unzip ────┤                            │
│                                        ▼                            │
│                    artifacts/data_ingestion/Chest-CT-Scan-data/     │
│                        ├── train/  (class subdirectories)           │
│                        └── valid/  (class subdirectories)           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DVC-ORCHESTRATED PIPELINE  (dvc.yaml)             │
│                                                                     │
│  Stage 1: data_ingestion                                            │
│    gdown → data.zip → extract → Chest-CT-Scan-data/                │
│                                                                     │
│  Stage 2: prepare_base_model                                        │
│    VGG16(imagenet, include_top=False) → freeze → custom head       │
│    → base_model.h5 + base_model_updated.h5                         │
│                                                                     │
│  Stage 3: training                                                  │
│    ImageDataGenerator (augment) → model.fit() → model.h5           │
│                                                                     │
│  Stage 4: evaluation                                                │
│    model.evaluate() → scores.json                                   │
│    mlflow.log_params/metrics → DagsHub                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER  (app.py)                        │
│                                                                     │
│  Flask  port 8080                                                   │
│    GET  /        → index.html (Bootstrap upload UI)                 │
│    GET  /train   → python main.py (all 4 stages)                   │
│    POST /predict → base64 decode → VGG16 predict → class label     │
│                   Returns: "Adenocarcinoma Cancer" or "Normal"      │
│                                                                     │
│  [Docker Container]  ←──  [AWS EC2 :8080]                          │
│         ▲                                                           │
│  [GitHub push] → [Actions] → [ECR] → [EC2 deploy]                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data & Artifact Flow

| Stage | Input | Output | Size |
|-------|-------|--------|------|
| Data Ingestion | Google Drive URL | `Chest-CT-Scan-data/` (343 files) | ~49 MB |
| Prepare Base Model | `config.yaml` + `params.yaml` | `base_model.h5` + `base_model_updated.h5` | ~118 MB |
| Model Training | `base_model_updated.h5` + CT images | `artifacts/training/model.h5` | ~59 MB |
| Evaluation | `model.h5` + CT images | `scores.json` + MLflow logs | 73 bytes |
| Prediction | Base64 CT image | `"Adenocarcinoma Cancer"` or `"Normal"` | — |

---

## 5. Repository Structure

```
Chest-Disease-Identification/
│
├── src/cnnClassifier/                     # Core Python package (installed as -e .)
│   ├── __init__.py                        # Logger setup (file + stdout)
│   ├── components/                        # Stage business logic
│   │   ├── data_ingestion.py             # gdown download + ZIP extract
│   │   ├── prepare_base_model.py         # VGG16 load + freeze + custom head
│   │   ├── model_trainer.py              # ImageDataGenerator + model.fit()
│   │   └── model_evaluation_mlflow.py    # model.evaluate() + MLflow logging
│   ├── config/
│   │   └── configuration.py              # ConfigurationManager — reads both YAMLs
│   ├── constants/__init__.py             # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│   ├── entity/
│   │   └── config_entity.py             # Frozen dataclasses per stage
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_trainer.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py                 # PredictionPipeline — loads model.h5, predicts
│   └── utils/
│       └── common.py                     # read_yaml, create_directories, save_json,
│                                         # decodeImage, encodeImageIntoBase64
│
├── config/
│   └── config.yaml                       # Artifact paths + Google Drive source URL
├── params.yaml                           # All hyperparameters (epochs, batch, lr, image_size)
├── dvc.yaml                              # 4-stage DVC pipeline definition
├── dvc.lock                              # Locked dependency hashes for reproducibility
├── scores.json                           # Latest evaluation: {loss, accuracy}
│
├── research/                             # Jupyter notebooks (exploratory)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
│
├── artifacts/                            # Pipeline outputs (auto-created, DVC-tracked)
│   ├── data_ingestion/Chest-CT-Scan-data/ # CT scan images
│   ├── prepare_base_model/               # base_model.h5 + base_model_updated.h5
│   └── training/model.h5                 # Final trained model
│
├── model/model.h5                        # Production copy loaded by PredictionPipeline
│
├── templates/index.html                  # Bootstrap 4 upload UI
├── app.py                                # Flask app (3 routes)
├── main.py                               # Runs all 4 pipeline stages
├── Dockerfile                            # python:3.9-slim-buster + awscli
├── .github/workflows/main.yaml          # GitHub Actions CI/CD
├── requirements.txt                      # TensorFlow, MLflow, DVC, Flask, gdown…
└── setup.py                              # Package: cnnClassifier
```

---

## 6. Data & Classes

### Dataset

| Property | Detail |
|----------|--------|
| **Name** | Chest CT Scan Dataset |
| **Source** | Google Drive (`gdown` download at pipeline runtime) |
| **Total files** | 343 images |
| **Total size** | ~49 MB |
| **Format** | JPEG/PNG CT scan images in class subdirectories |
| **Structure** | YOLO-style: `Chest-CT-Scan-data/train/<class>/` + `valid/<class>/` |

### Classes

| Label | Class | Description |
|-------|-------|-------------|
| `0` | **Adenocarcinoma Cancer** | CT scan showing adenocarcinoma — the most common form of lung cancer, originating in glandular cells of the lung periphery |
| `1` | **Normal** | CT scan showing healthy lung tissue with no detectable malignancy |

### Why Two Classes?

The `params.yaml` sets `CLASSES: 2`, and the prediction pipeline maps:

```python
result[0] == 1  →  "Normal"
result[0] == 0  →  "Adenocarcinoma Cancer"
```

The model uses **categorical cross-entropy** with **softmax output** — treating this as a 2-class classification rather than a single sigmoid binary output. This design makes it straightforward to extend to additional disease classes (Pneumonia, Pleural Effusion, etc.) by increasing `CLASSES` and retraining.

### Data Augmentation (when `AUGMENTATION=True`)

Applied to the training generator only:

```python
rotation_range    = 40
horizontal_flip   = True
width_shift_range  = 0.2
height_shift_range = 0.2
shear_range        = 0.2
zoom_range         = 0.2
rescale            = 1./255
```

Augmentation increases the effective dataset size and teaches the model to be invariant to reasonable transformations that would still represent valid CT scan presentations.

---

## 7. ML Pipeline — Step by Step

The full pipeline is defined in `dvc.yaml` and executed by either `dvc repro` (DVC-tracked) or `python main.py` (direct). DVC tracks file hashes in `dvc.lock` — a stage only re-runs if its dependencies have changed since the last run.

---

### Stage 1 — Data Ingestion

**Component:** `data_ingestion.py` | **Config entity:** `DataIngestionConfig`
**DVC deps:** `stage_01_data_ingestion.py`, `config/config.yaml`
**DVC out:** `artifacts/data_ingestion/Chest-CT-Scan-data`

1. Reads `source_URL` from `config.yaml` (Google Drive share link)
2. Extracts the file ID from the URL
3. Downloads `data.zip` (~49 MB) via `gdown` to `artifacts/data_ingestion/`
4. Extracts ZIP to `artifacts/data_ingestion/` → creates `Chest-CT-Scan-data/`

---

### Stage 2 — Prepare Base Model

**Component:** `prepare_base_model.py` | **Config entity:** `PrepareBaseModelConfig`
**DVC params:** `IMAGE_SIZE`, `INCLUDE_TOP`, `CLASSES`, `WEIGHTS`, `LEARNING_RATE`
**DVC out:** `artifacts/prepare_base_model/` (2 `.h5` files, ~118 MB total)

1. Loads `tf.keras.applications.vgg16.VGG16`:
   - `input_shape=[224, 224, 3]`
   - `weights='imagenet'`
   - `include_top=False` — removes the original 1000-class ImageNet head
2. Saves raw backbone as `base_model.h5`
3. Adds custom classification head:
   - `Flatten()` layer on VGG16 output
   - `Dense(2, activation='softmax')` — 2-class output
4. **Freezes all VGG16 layers** (`model.trainable = False`) — only the new Dense head is trained
5. Compiles: `SGD(lr=0.01)`, `CategoricalCrossentropy`, metrics=`["accuracy"]`
6. Saves as `base_model_updated.h5`

---

### Stage 3 — Model Training

**Component:** `model_trainer.py` | **Config entity:** `TrainingConfig`
**DVC params:** `IMAGE_SIZE`, `EPOCHS`, `BATCH_SIZE`, `AUGMENTATION`
**DVC out:** `artifacts/training/model.h5` (~59 MB)

1. Loads `base_model_updated.h5`
2. Creates **validation generator** (20% split, no augmentation, rescale 1/255)
3. Creates **training generator** with augmentation (if `AUGMENTATION=True`)
4. Both generators use `flow_from_directory` on `Chest-CT-Scan-data/`
5. Calculates `steps_per_epoch = samples // batch_size`
6. Calls `model.fit()` for `EPOCHS` epochs
7. Saves trained model to `artifacts/training/model.h5`

---

### Stage 4 — Model Evaluation

**Component:** `model_evaluation_mlflow.py` | **Config entity:** `EvaluationConfig`
**DVC params:** `IMAGE_SIZE`, `BATCH_SIZE`
**DVC metric:** `scores.json`

1. Loads `artifacts/training/model.h5`
2. Creates **validation generator** (30% split, rescale 1/255)
3. Runs `model.evaluate()` → returns `[loss, accuracy]`
4. Saves results to `scores.json`: `{"loss": ..., "accuracy": ...}`
5. Logs to **MLflow on DagsHub**:
   - `mlflow.log_params(all_params)` — all `params.yaml` values
   - `mlflow.log_metrics({"loss": ..., "accuracy": ...})`
   - Registers model as `VGG16Model` in MLflow Model Registry

> **Note:** In the current code, `evaluation.log_into_mlflow()` is commented out in `stage_04_model_evaluation.py`. Uncomment and set the DagsHub environment variables to enable remote tracking.

---

## 8. VGG16 Transfer Learning Architecture

### Why VGG16?

VGG16 was trained on 14 million ImageNet images across 1,000 classes. Its convolutional layers have learned to detect universal visual features — edges, textures, shapes, colour gradients — that are also present in medical imaging. By reusing these frozen features and training only a small classification head, the model can learn to classify CT scans with a relatively small dataset and far fewer training epochs than training from scratch.

### Architecture Diagram

```
Input: 224×224×3 CT scan image
         │
         ▼
┌─────────────────────────────┐
│  VGG16 BACKBONE             │  ← FROZEN (all layers trainable=False)
│  (ImageNet pretrained)      │
│                             │
│  Block 1: 2× Conv2D(64)     │
│  Block 2: 2× Conv2D(128)    │
│  Block 3: 3× Conv2D(256)    │
│  Block 4: 3× Conv2D(512)    │
│  Block 5: 3× Conv2D(512)    │
│                             │
│  Output: 7×7×512 feature map│
└──────────────┬──────────────┘
               │
               ▼
         Flatten()
         25,088 features
               │
               ▼
      ┌─────────────────┐
      │  Dense(2)       │  ← TRAINABLE (only this layer)
      │  + Softmax      │
      └─────────────────┘
               │
               ▼
    [P(Adenocarcinoma), P(Normal)]
         argmax → class label
```

### Model Configuration

```yaml
# params.yaml
IMAGE_SIZE:     [224, 224, 3]   # VGG16 standard input
INCLUDE_TOP:    False           # Remove ImageNet 1000-class head
WEIGHTS:        imagenet        # Use ImageNet pretrained weights
CLASSES:        2               # Adenocarcinoma + Normal
LEARNING_RATE:  0.01
BATCH_SIZE:     16
EPOCHS:         1               # Increase to 20–50 for production
AUGMENTATION:   True
```

### Why Freeze All Layers?

With `freeze_all=True`, the VGG16 backbone acts as a **fixed feature extractor** — only the final Dense(2) layer is updated during training. This is appropriate when:
- The target dataset is small relative to ImageNet
- Training compute is limited
- ImageNet features are sufficiently general to capture CT scan patterns

For higher accuracy, consider **fine-tuning**: unfreeze the last 2–4 VGG16 convolutional blocks and train end-to-end with a lower learning rate.

---

## 9. Model Performance

### Current Evaluation Results (1 epoch, demo run)

```json
{
    "loss":     20.008527755737305,
    "accuracy": 0.5686274766921997
}
```

### Interpretation

| Metric | Value | Assessment |
|--------|-------|------------|
| **Accuracy** | 56.9% | Slightly above random (50%) — expected for 1 epoch |
| **Loss** | 20.01 | Very high — model has not yet converged |

A single training epoch with a frozen backbone on a small dataset is insufficient for clinical-grade performance. These numbers are consistent with a randomly initialised Dense head that has barely begun to learn from the training data.

### Expected Performance with Proper Training

With 20–50 epochs, fine-tuning of the last VGG16 blocks, and appropriate learning rate scheduling, this architecture can realistically achieve:

| Target | Expected Range |
|--------|---------------|
| Accuracy | 85–95% |
| AUC-ROC | 0.90–0.97 |
| Sensitivity (Recall) | > 90% (critical for cancer detection) |

See Section 15 for specific improvement recommendations.

---

## 10. Flask Web Application

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Renders `index.html` — Bootstrap 4 CT scan upload UI |
| `GET/POST` | `/train` | Triggers `python main.py` — runs all 4 pipeline stages |
| `POST` | `/predict` | Accepts `{"image": "<base64>"}` → returns `[{"image": "Adenocarcinoma Cancer"}]` or `[{"image": "Normal"}]` |

### Prediction Flow

```
POST /predict
  ├── image = request.json['image']          # base64 string from frontend
  ├── decodeImage(image, "inputImage.jpg")   # saves to disk
  ├── PredictionPipeline("inputImage.jpg").predict()
  │     ├── load_model("model/model.h5")
  │     ├── image.load_img(filename, target_size=(224,224))
  │     ├── img_to_array → expand_dims → predict()
  │     ├── np.argmax(result, axis=1)
  │     └── 0 → "Adenocarcinoma Cancer" | 1 → "Normal"
  └── return jsonify([{"image": prediction}])
```

> **Note:** `PredictionPipeline` loads from `model/model.h5` (production copy), not `artifacts/training/model.h5`. Ensure a trained model is placed at `model/model.h5` before the app can serve predictions.

### Frontend UI (`templates/index.html`)

The Bootstrap 4 interface provides:
- **Upload button** — file picker for CT scan JPEG/PNG
- **Predict button** — encodes image to base64, POSTs to `/predict`
- **Results panel** — displays the prediction label (`"Normal"` or `"Adenocarcinoma Cancer"`)
- **Loading spinner** — shown during inference

---

## 11. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.8+
- Conda or `venv`
- Git
- Docker Desktop (optional)
- AWS account (optional, for deployment)
- DagsHub account (optional, for MLflow remote tracking)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Chest-Disease-Identification.git
cd Chest-Disease-Identification
```

---

### Step 2 — Create Python Environment

```bash
conda create -n cdis python=3.8 -y
conda activate cdis
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# Installs TensorFlow 2.12, MLflow, DVC, Flask, gdown, and more
# Installs cnnClassifier package in editable mode (-e .)
```

---

### Step 4 — Configure MLflow Tracking (Optional)

To enable experiment logging to DagsHub, set these environment variables before running:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/Chest-Disease-Identification.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Then uncomment `evaluation.log_into_mlflow()` in `stage_04_model_evaluation.py`.

---

### Step 5 — Run the Pipeline

**Option A — via DVC (recommended, reproducible):**
```bash
dvc init        # if first time
dvc repro       # runs only changed stages
dvc dag         # visualise the pipeline graph
```

**Option B — direct Python:**
```bash
python main.py  # runs all 4 stages sequentially
```

After completion, verify:

```bash
ls artifacts/data_ingestion/Chest-CT-Scan-data/  # CT scan images
ls artifacts/prepare_base_model/                  # base_model.h5 + base_model_updated.h5
ls artifacts/training/model.h5                    # trained model (~59 MB)
cat scores.json                                    # {"loss": ..., "accuracy": ...}
```

---

### Step 6 — Copy Model for Prediction

```bash
mkdir -p model
cp artifacts/training/model.h5 model/model.h5
```

---

### Step 7 — Launch the Web App

```bash
python app.py
# Opens at http://0.0.0.0:8080
```

---

## 12. Running the Application

### Local Run

```bash
python app.py
# http://localhost:8080
```

### Docker Run

```bash
docker build -t cdis:latest .
docker run -d -p 8080:8080 cdis:latest
# http://localhost:8080
```

### Retrain via Browser

Visit `http://localhost:8080/train` — triggers `os.system("python main.py")` which runs all 4 stages.

### Local MLflow UI

```bash
mlflow ui
# http://localhost:5000
# Browse experiment runs, params, metrics, and registered models
```

### DVC Pipeline Commands

```bash
dvc repro           # Run only changed stages
dvc dag             # Visualise pipeline as ASCII DAG
dvc status          # Check which stages need re-running
dvc params diff     # Compare params.yaml changes
```

---

## 13. CI/CD & Cloud Deployment

Every push to `main` (excluding `README.md` changes) triggers the 3-job GitHub Actions pipeline:

```
Developer ──► git push origin main
                      │
             GitHub Actions triggered
                      │
       ┌──────────────┼──────────────────────────────────────┐
       │              │                                       │
  Job 1: CI      Job 2: CD (Build)              Job 3: Deploy
  (ubuntu)       (ubuntu)                       (self-hosted EC2)
       │              │                                       │
  Lint echo      AWS credentials                Checkout code
  Test echo      Login to ECR                   AWS credentials
                 docker build                   Login to ECR
                 docker push                    docker pull
                 → ECR :latest                  docker run -d -p 8080:8080
                                                docker system prune -f
```

### GitHub Secrets Required

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ECR_LOGIN_URI` | e.g. `566373416292.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | ECR repository name |

### IAM Policies Required

```
AmazonEC2ContainerRegistryFullAccess
AmazonEC2FullAccess
```

### Dockerfile

```dockerfile
FROM python:3.9-slim-buster
RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
```

---

## 14. Business Applications & Other Domains

### Primary Use Case — Chest CT Scan Triage

| User | Value Delivered |
|------|----------------|
| **Radiologists** | Automated pre-screening flags high-priority scans for immediate review |
| **Hospitals & clinics** | Reduce reporting backlogs in high-volume screening programmes |
| **Teleradiology services** | Triage incoming scan queues automatically before human review |
| **Oncology departments** | Earlier identification of suspected adenocarcinoma cases for biopsy |
| **Research institutions** | Automated labelling of large unlabelled CT scan archives |

> ⚠️ Any clinical deployment requires regulatory approval (e.g. FDA 510(k), CE marking), prospective clinical validation, and integration into a supervised clinical workflow.

### Adjacent Domains (Same CNN Transfer Learning Pattern)

| Domain | Analogous Application | Adaptation |
|--------|----------------------|-----------|
| **Radiology** | Pneumonia vs Normal X-ray classification | Replace CT dataset with chest X-ray dataset |
| **Dermatology** | Melanoma detection from dermoscopy images | Replace with ISIC dermoscopy dataset |
| **Pathology** | Histopathology slide classification | Replace with WSI patches dataset |
| **Ophthalmology** | Diabetic retinopathy grading from fundus images | Multi-class (5-grade) output |
| **Cardiology** | Echocardiogram view classification | Video frame extraction + classification |
| **Manufacturing** | Product defect detection from camera images | Replace CT scans with factory inspection images |
| **Agriculture** | Plant disease identification from leaf photos | Replace with PlantVillage dataset |

---

## 15. How to Improve This Project

### 🧠 Model & Training Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Increase epochs** | 🔴 High | `EPOCHS=1` is a demonstration — train for 20–50 epochs minimum; monitor val_loss for convergence |
| **Unfreeze top VGG16 layers** | 🔴 High | Fine-tune the last 2–3 convolutional blocks with a small learning rate (1e-4) for substantially better accuracy |
| **Add more disease classes** | 🔴 High | Extend to Pneumonia, Pleural Effusion, etc. by increasing `CLASSES` and updating the dataset |
| **Learning rate scheduling** | 🟡 Medium | Add `ReduceLROnPlateau` or `CosineDecay` — SGD with fixed lr=0.01 may overshoot optima |
| **Try EfficientNetB0/B3** | 🟡 Medium | EfficientNet is more parameter-efficient than VGG16 and achieves higher accuracy on medical imaging benchmarks |
| **Add sensitivity/specificity** | 🟡 Medium | For clinical use, accuracy alone is insufficient — add AUC-ROC, sensitivity, and specificity metrics |
| **Cross-validation** | 🟡 Medium | k-fold CV gives more reliable performance estimates than a single train/val split |
| **Class imbalance** | 🟢 Low | Check class distribution; add `class_weight` to `model.fit()` if imbalanced |

### 🏗️ MLOps & Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Enable MLflow logging** | Uncomment `evaluation.log_into_mlflow()` and set DagsHub env vars — the infrastructure is already built |
| **Add DVC remote storage** | Configure S3 or GCS as DVC remote for `artifacts/` — enables `dvc push/pull` across machines |
| **Replace `os.system("python main.py")`** | Use subprocess with error handling, or make `/train` asynchronous with Celery |
| **Add `/health` endpoint** | Check Flask is running and `model/model.h5` exists |
| **Unit tests** | Test `decodeImage`, `PredictionPipeline`, config loading, and pipeline stage logic |
| **Model card** | Document training data characteristics, known limitations, and appropriate use contexts |
| **Async training** | Flask `/train` blocks the server — run training in a background thread |

### 📦 Product Improvements

- Return a **confidence score** alongside the class prediction — e.g. `{"class": "Adenocarcinoma", "confidence": 0.847}`
- Add **Grad-CAM visualisation** — overlay a heatmap on the CT scan showing which regions influenced the prediction
- Support **DICOM format** input in addition to JPEG/PNG — DICOM is the clinical standard for medical images
- Add **audit logging** — record every prediction with timestamp and image hash for clinical accountability

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **VGG16** | Visual Geometry Group 16-layer CNN — a deep convolutional neural network pretrained on 14M ImageNet images |
| **Transfer Learning** | Reusing a model trained on one task (ImageNet classification) as the starting point for a different but related task (CT scan classification) |
| **Feature Extraction** | Using the frozen VGG16 backbone as a fixed feature extractor — only the new classification head is trained |
| **Fine-tuning** | Unfreezing some of the pretrained backbone layers and training them end-to-end on the target dataset with a low learning rate |
| **Adenocarcinoma** | The most common type of lung cancer, arising from glandular epithelial cells in the lung periphery |
| **ImageDataGenerator** | Keras utility that generates batches of augmented images from directories at training time |
| **Augmentation** | Applying random transformations (rotation, flip, zoom) to training images to increase effective dataset size and improve generalisation |
| **DVC** | Data Version Control — tool for versioning data and ML models, and orchestrating reproducible pipelines via `dvc.yaml` |
| **dvc repro** | DVC command that runs only the pipeline stages whose dependencies have changed since the last run |
| **dvc.lock** | File recording the exact file hashes of all pipeline inputs and outputs — enables reproducibility |
| **MLflow** | Open-source ML lifecycle platform for experiment tracking, model versioning, and deployment |
| **DagsHub** | Git + DVC + MLflow hosting platform — provides remote experiment tracking server for MLflow |
| **ConfigurationManager** | Central class reading both `config.yaml` and `params.yaml`, assembling frozen dataclass configs per stage |
| **ConfigBox** | `python-box` object allowing dot-notation access to YAML dictionaries |
| **frozen dataclass** | Python dataclass with `frozen=True` — all fields are immutable after creation |
| **Softmax** | Activation function that converts raw network outputs into probabilities summing to 1.0 across all classes |
| **CategoricalCrossentropy** | Loss function for multi-class classification with one-hot encoded labels |
| **SGD** | Stochastic Gradient Descent — optimiser used here with `learning_rate=0.01` |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚕️ Medical Disclaimer

This project is for **educational and research purposes only**. It is not validated for clinical use and must not be used for medical diagnosis. Any clinical application requires regulatory approval, prospective clinical validation, and integration into a supervised clinical workflow with appropriate governance.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---
