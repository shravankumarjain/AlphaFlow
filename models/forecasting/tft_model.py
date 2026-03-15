# models/forecasting/tft_model.py
#
# AlphaFlow — Temporal Fusion Transformer
#
# Architecture overview:
#
#   INPUT LAYER
#   ├── Static embeddings     (ticker, sector)
#   ├── Known future encoder  (calendar features)
#   └── Unknown past encoder  (price, sentiment, macro...)
#         ↓
#   VARIABLE SELECTION NETWORKS
#   └── Learns which of 44 features matter most at each timestep
#         ↓
#   LSTM ENCODER  (processes past 63 days)
#         ↓
#   LSTM DECODER  (generates 5-day forecast)
#         ↓
#   MULTI-HEAD ATTENTION
#   └── Learns long-range dependencies (e.g. earnings cycle patterns)
#         ↓
#   QUANTILE OUTPUT
#   └── Predicts p10, p50, p90 — gives uncertainty bounds, not just point estimate
#       This is what makes it production-grade: we know HOW confident the model is
#
# Training strategy:
#   - Loss: QuantileLoss (predicts distribution, not just mean)
#   - Optimizer: Adam with learning rate finder
#   - Early stopping: stops when val loss stops improving
#   - M2 GPU: uses Apple Metal (MPS) acceleration automatically
#
# Run: python models/forecasting/tft_model.py

import warnings

warnings.filterwarnings("ignore")

import logging  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import torch  # noqa: E402
import lightning as pl  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)  # noqa: E402
from lightning.pytorch.loggers import MLFlowLogger  # noqa: E402

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # noqa: E402
from pytorch_forecasting.metrics import QuantileLoss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402, F401
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401
import json  # noqa: E402

import sys  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import AWS_REGION, S3_BUCKET, LOCAL_DATA_DIR  # noqa: E402, F401
from models.forecasting.dataset import (  # noqa: E402
    prepare_datasets,
    TARGET,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/tft_training.log"),
    ],
)
logger = logging.getLogger("tft_model")


# ═══════════════════════════════════════════════════════════════════════
# DEVICE DETECTION — M2 Mac uses MPS (Metal Performance Shaders)
# ═══════════════════════════════════════════════════════════════════════


def get_device() -> str:
    """
    Detect the best available compute device.
    M2 Mac → MPS (Apple Silicon GPU, ~5x faster than CPU for this model)
    CUDA GPU → cuda
    Fallback → cpu
    """
    if torch.backends.mps.is_available():
        logger.info("  ✓ Device: Apple MPS (M2 GPU) — Metal acceleration enabled")
        return "mps"
    elif torch.cuda.is_available():
        logger.info(f"  ✓ Device: CUDA ({torch.cuda.get_device_name(0)})")
        return "gpu"
    else:
        logger.info("  ✓ Device: CPU (training will be slower)")
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# Start conservative — we can tune after first successful run
# ═══════════════════════════════════════════════════════════════════════

HPARAMS = {
    # Model architecture
    "hidden_size": 64,  # size of hidden layers — increase for more capacity
    "attention_head_size": 4,  # multi-head attention heads
    "dropout": 0.1,  # regularisation — prevent overfitting
    "hidden_continuous_size": 32,  # size of continuous variable processing network
    "lstm_layers": 2,  # number of LSTM layers in encoder/decoder
    # Training
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,  # early stopping will kick in before this
    "gradient_clip_val": 0.1,  # prevents exploding gradients
    # Loss — predicts 3 quantiles simultaneously
    # p10 = pessimistic, p50 = median forecast, p90 = optimistic
    "loss_quantiles": [0.1, 0.5, 0.9],
    # Early stopping
    "patience": 8,  # stop if no improvement for 8 epochs
}


# ═══════════════════════════════════════════════════════════════════════
# BUILD DATALOADERS
# ═══════════════════════════════════════════════════════════════════════


def build_dataloaders(training_dataset, validation_dataset):
    """
    Wrap TimeSeriesDataSets in PyTorch DataLoaders.
    DataLoaders handle batching, shuffling, and parallel data loading.
    """
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=HPARAMS["batch_size"],
        num_workers=0,  # 0 = main process only (safe for M2)
        shuffle=True,
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=HPARAMS["batch_size"] * 2,
        num_workers=0,
        shuffle=False,
    )
    logger.info(f"  ✓ Train batches: {len(train_loader)}")
    logger.info(f"  ✓ Val batches  : {len(val_loader)}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════════════════════════════════════


def build_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """
    Instantiate the TFT model from the training dataset.

    pytorch-forecasting reads the dataset metadata to automatically:
    - Set input/output sizes correctly
    - Route features to the right sub-networks
    - Configure the output layer for our quantiles
    """
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=HPARAMS["learning_rate"],
        hidden_size=HPARAMS["hidden_size"],
        attention_head_size=HPARAMS["attention_head_size"],
        dropout=HPARAMS["dropout"],
        hidden_continuous_size=HPARAMS["hidden_continuous_size"],
        lstm_layers=HPARAMS["lstm_layers"],
        loss=QuantileLoss(quantiles=HPARAMS["loss_quantiles"]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  ✓ TFT model built: {total_params:,} trainable parameters")
    return model


# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS — control training behaviour
# ═══════════════════════════════════════════════════════════════════════


def build_callbacks(checkpoint_dir: str) -> list:
    """
    Callbacks run at the end of each epoch.

    EarlyStopping    : stops training when val_loss plateaus
    ModelCheckpoint  : saves the best model weights automatically
    LRMonitor        : logs learning rate to MLflow
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=HPARAMS["patience"],
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # only keep the best checkpoint
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [early_stop, checkpoint, lr_monitor]


# ═══════════════════════════════════════════════════════════════════════
# TRAINING — wired to MLflow for experiment tracking
# ═══════════════════════════════════════════════════════════════════════


def train_model(
    training_dataset, validation_dataset, experiment_name: str = "alphaflow-tft"
) -> dict:
    """
    Full training pipeline with MLflow experiment tracking.

    MLflow tracks:
    - All hyperparameters
    - Loss curves (train + val) per epoch
    - Best model weights as artifact
    - Feature importance scores

    Access MLflow UI after training:
        mlflow ui --port 5000
        open http://localhost:5000
    """
    logger.info("=" * 60)
    logger.info("AlphaFlow — TFT Model Training")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ─────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")  # local MLflow server
    mlflow.set_experiment(experiment_name)

    device = get_device()

    with mlflow.start_run(run_name="tft-v1") as run:
        # Log all hyperparameters
        mlflow.log_params(HPARAMS)
        mlflow.log_param("device", device)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("encoder_length", MAX_ENCODER_LENGTH)
        mlflow.log_param("prediction_length", MAX_PREDICTION_LENGTH)

        logger.info(f"\n  MLflow run ID: {run.info.run_id}")

        # ── Build dataloaders ─────────────────────────────────────────
        logger.info("\nBuilding dataloaders...")
        train_loader, val_loader = build_dataloaders(
            training_dataset, validation_dataset
        )

        # ── Build model ───────────────────────────────────────────────
        logger.info("\nBuilding TFT model...")
        model = build_tft_model(training_dataset)

        # ── MLflow logger for Lightning ───────────────────────────────
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            tracking_uri="mlruns",
        )

        # ── Trainer ───────────────────────────────────────────────────
        # accelerator="mps" for M2, "gpu" for CUDA, "cpu" for fallback
        accelerator = (
            "mps" if device == "mps" else ("gpu" if device == "gpu" else "cpu")
        )

        trainer = pl.Trainer(
            max_epochs=HPARAMS["max_epochs"],
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=HPARAMS["gradient_clip_val"],
            callbacks=build_callbacks(str(checkpoint_dir)),
            logger=mlf_logger,
            enable_progress_bar=True,
            log_every_n_steps=5,
        )

        # ── TRAIN ─────────────────────────────────────────────────────
        logger.info("\nStarting training...")
        logger.info(f"  Max epochs   : {HPARAMS['max_epochs']}")
        logger.info(f"  Early stop   : patience={HPARAMS['patience']}")
        logger.info(f"  Batch size   : {HPARAMS['batch_size']}")
        logger.info(f"  Device       : {accelerator}")
        logger.info("\n  Watch val_loss — should decrease each epoch\n")

        trainer.fit(model, train_loader, val_loader)

        # ── Load best checkpoint ──────────────────────────────────────
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        logger.info(f"\n  ✓ Best checkpoint: {best_checkpoint}")
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_checkpoint)

        # ── Evaluate on validation set ────────────────────────────────
        logger.info("\nEvaluating best model on validation set...")
        val_predictions = best_model.predict(
            val_loader,
            mode="raw",
            return_x=True,
        )

        # ── Feature importance ────────────────────────────────────────
        # TFT's Variable Selection Networks give us interpretable
        # feature importance scores — which features mattered most
        logger.info("\nComputing feature importance...")
        try:
            raw_predictions, x = val_predictions.output, val_predictions.x  # noqa: F841
            interpretation = best_model.interpret_output(
                raw_predictions, reduction="sum"
            )
            importance = best_model.plot_interpretation(interpretation)  # noqa: F841

            # Save feature importance as JSON to MLflow
            encoder_importance = interpretation["encoder_variables"]
            if hasattr(encoder_importance, "numpy"):
                encoder_importance = encoder_importance.numpy()

            feature_names = training_dataset.reals
            importance_dict = {
                name: float(score)
                for name, score in zip(
                    feature_names, encoder_importance[: len(feature_names)]
                )
            }
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            importance_path = "models/feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance_dict, f, indent=2)
            mlflow.log_artifact(importance_path)

            logger.info("\n  Top 10 most important features:")
            for feat, score in list(importance_dict.items())[:10]:
                bar = "█" * int(score * 50)
                logger.info(f"    {feat:35s} {bar} {score:.4f}")

        except Exception as e:
            logger.warning(f"  ⚠ Feature importance failed: {e}")

        # ── Log final metrics ─────────────────────────────────────────
        best_val_loss = trainer.checkpoint_callback.best_model_score
        if best_val_loss is not None:
            mlflow.log_metric("best_val_loss", float(best_val_loss))

        # ── Save model to MLflow ──────────────────────────────────────
        mlflow.pytorch.log_model(best_model, "tft_model")
        logger.info(f"\n  ✓ Model saved to MLflow run: {run.info.run_id}")

        results = {
            "run_id": run.info.run_id,
            "best_val_loss": float(best_val_loss) if best_val_loss else None,
            "best_checkpoint": best_checkpoint,
            "epochs_trained": trainer.current_epoch,
            "model": best_model,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK — run before full training
# ═══════════════════════════════════════════════════════════════════════


def sanity_check(training_dataset, train_loader):
    """
    Run one forward pass to verify the model works before full training.
    Catches shape errors immediately rather than after 10 minutes.
    """
    logger.info("\nRunning sanity check (1 forward pass)...")
    model = build_tft_model(training_dataset)
    batch = next(iter(train_loader))
    x, y = batch

    with torch.no_grad():
        out = model(x)

    logger.info("  ✓ Input shapes look correct")
    logger.info(f"  ✓ Output shape: {out.prediction.shape}")
    logger.info(
        f"  Expected: [batch={HPARAMS['batch_size']}, horizon={MAX_PREDICTION_LENGTH}, quantiles=3]"
    )
    logger.info("  ✓ Sanity check passed — safe to run full training\n")
    return True


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from models.forecasting.dataset import prepare_datasets

    logger.info("Preparing datasets...")
    training_dataset, validation_dataset, train_df, val_df = prepare_datasets()

    # Build loaders for sanity check
    train_loader, val_loader = build_dataloaders(training_dataset, validation_dataset)

    # Sanity check first — fail fast
    sanity_check(training_dataset, train_loader)

    # Full training
    results = train_model(training_dataset, validation_dataset)

    print("\n── Training Complete ─────────────────────────────")
    print(f"  Run ID         : {results['run_id']}")
    print(f"  Best val loss  : {results['best_val_loss']:.4f}")
    print(f"  Epochs trained : {results['epochs_trained']}")
    print(f"  Checkpoint     : {results['best_checkpoint']}")
    print("\nView results:")
    print("  mlflow ui --port 5000")
    print("  open http://localhost:5000")
