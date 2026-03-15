# End-to-End Chest Disease Identification

## 1. Project Overview
This project is an end-to-end Machine Learning application designed to classify chest diseases from CT scan images. It demonstrates a complete MLOps lifecycle, incorporating experiment tracking, pipeline orchestration, model versioning, and automated deployment using CI/CD pipelines.

The solution leverages Deep Learning (CNNs) for classification and provides a web interface for user interaction.

## 2. Key Features
- **End-to-End Pipeline**: From data ingestion to model deployment.
- **MLflow Integration**: Tracks experiments, runs, and logs models and metrics.
- **DVC (Data Version Control)**: Orchestrates the ML pipeline and manages data/model versions.
- **Modular Codebase**: Clean, structured code with separate components for each pipeline stage.
- **CI/CD Deployment**: Automated deployment pipeline using GitHub Actions to AWS EC2.
- **Containerization**: Application is containerized using Docker.

## 3. Technology Stack
- **Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow 2.12.0
- **Experiment Tracking**: MLflow, DaggerHub
- **Pipeline Orchestration**: DVC
- **Web Backend**: Flask
- **Containerization**: Docker
- **Cloud Provider**: AWS (EC2, ECR)
- **CI/CD**: GitHub Actions
- **Libraries**: Pandas, Numpy, Matplotlib, Seaborn, PyYAML, Joblib

## 4. Project Structure
The project follows a modular structure to ensure scalability and maintainability.

```
├── .github/workflows/   # CI/CD workflows for AWS deployment
├── config/              # Configuration files
│   └── config.yaml      # Main configuration for all pipeline stages
├── src/cnnClassifier/   # Main Source Code
│   ├── components/      # Business logic for each stage (Ingestion, Training, etc.)
│   ├── config/          # Configuration Manager
│   ├── entity/          # Data Classes for type-safe configuration
│   ├── pipeline/        # Scripts to execute individual pipeline stages
│   ├── utils/           # Utility functions (logging, file ops)
│   └── constants/       # Project constants
├── dvc.yaml             # DVC Pipeline definition
├── params.yaml          # Hyperparameters (epochs, batch size, learning rate)
├── requirements.txt     # Project dependencies
├── setup.py             # Package setup
├── main.py              # Entry point for the pipeline
└── app.py               # Flask application entry point
```

## 5. ML Pipeline Stages
The machine learning pipeline is orchestrated by DVC and defined in `dvc.yaml`.

1.  **Data Ingestion (`stage_01_data_ingestion`)**:
    *   Downloads the dataset from the source.
    *   Extracts the zip file.
2.  **Prepare Base Model (`stage_02_prepare_base_model`)**:
    *   Loads the pre-trained base model (e.g., VGG16) with ImageNet weights.
    *   Updates the model head for the custom classes.
3.  **Model Training (`stage_03_model_trainer`)**:
    *   Augments data.
    *   Fine-tunes the model on the training dataset.
    *   Saves the trained model (`model.h5`).
4.  **Evaluation (`stage_04_model_evaluation`)**:
    *   Evaluates the model on validation data.
    *   Logs metrics (accuracy, loss) to MLflow/Dagshub.

## 6. Workflow & Implementation Steps
To understand how updates are propagated through the system:
1.  **Update `config/config.yaml`**: Define paths and configuration constants.
2.  **Update `params.yaml`**: Adjust hyperparameters.
3.  **Update Entity**: Modify dataclasses in `src/entity` if config structure changes.
4.  **Update Configuration Manager**: Update `src/config/configuration.py` to read new configs.
5.  **Update Components**: Implement logic in `src/components`.
6.  **Update Pipeline**: Create/Update script in `src/pipeline`.
7.  **Run Pipeline**: Execute via `dvc repro` or `main.py`.

