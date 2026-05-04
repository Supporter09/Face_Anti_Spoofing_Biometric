# Research Plan

## Priority Order

1. Real detector + liveness model backend integration
2. Data preprocessing pipeline quality
3. Training and evaluation reproducibility
4. Paper drafting (delayed until metrics are solid)

## Dataset

Use `CelebA-Spoof` for the first liveness training cycle. Keep raw and processed data outside git and document paths in notebooks.

## Reusable Code Modules

- `src/fas/training_data.py`: manifest loading, splitting, face crop preparation
- `src/fas/training_loop.py`: training orchestration interface used by notebooks
- `src/fas/evaluation.py`: ACER/APCER/BPCER style metric helpers

## Notebook Workflow

- `01_data_prep_celeba_spoof.ipynb`: dataset prep and face crop generation
- `02_train_minifasnet.ipynb`: training and checkpointing
- `03_eval_minifasnet.ipynb`: evaluation and threshold selection
- `04_demo_inference.ipynb`: sanity checks for the backend inference path

## Kaggle Self-Contained Notebook Workflow

- `kaggle_full_01_data_prep.ipynb`: full inline prep code and manifest generation
- `kaggle_full_02_train.ipynb`: full inline model/dataset/training loop
- `kaggle_full_03_eval.ipynb`: full inline inference + threshold sweep metrics
- `kaggle_full_04_backend_sanity.ipynb`: full inline detector + liveness backend-style check

## Compute Strategy

- Local GPU for quick smoke runs and debugging
- Kaggle GPU for full training and evaluation

## Deliverables Before Paper Work

- A checkpoint that runs through backend inference
- Validation metrics with threshold analysis
- Basic spoof/live error analysis from held-out data
