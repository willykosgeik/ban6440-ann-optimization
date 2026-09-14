"""BAN6440 Module 6 — Reproducible ANN optimization pipeline.

Uses teleconnect.csv, creates train/validation/test partitions, selects
thresholds on validation data only, and evaluates locked thresholds once
on the untouched test set.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, initializers, layers, optimizers

SEED = 42
DATA_FILE = Path("teleconnect.csv")
OUTPUT_DIR = Path("output_final")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_prepare(path: Path):
    df = pd.read_csv(path)
    if "customerID" in df:
        df = df.drop(columns="customerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    y = (df.pop("Churn") == "Yes").astype(int).to_numpy()
    X = pd.get_dummies(df, drop_first=True, dtype=int).astype(float)
    constant = [c for c in X if X[c].nunique() <= 1]
    if constant:
        X = X.drop(columns=constant)
    return X.to_numpy(), y, X.columns.tolist()


def split_and_scale(X, y):
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def baseline_model(input_dim: int):
    model = keras.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def optimized_model(input_dim: int, name: str):
    he = initializers.HeNormal(seed=SEED)
    model = keras.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(32, kernel_initializer=he),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(16, kernel_initializer=he),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name=f"ANN_{name}")
    if name == "SGD":
        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    elif name == "RMSprop":
        opt = optimizers.RMSprop(learning_rate=0.001)
    elif name == "Adam":
        opt = optimizers.Adam(learning_rate=0.001)
    else:
        raise ValueError(name)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def train(model, X_train, y_train, X_val, y_val, class_weight, epochs, name):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=15,
                restore_best_weights=True, verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7,
                min_lr=1e-6, verbose=0
            ),
        ],
        verbose=0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Training")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set(title=f"{name}: Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history.history["auc"], label="Training AUC")
    axes[1].plot(history.history["val_auc"], label="Validation AUC")
    axes[1].set(title=f"{name}: Training and Validation AUC", xlabel="Epoch", ylabel="AUC-ROC")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"training_{name.lower()}.png", dpi=150); plt.close(fig)
    return history


def probabilities(model, X):
    return model.predict(X, verbose=0).ravel()


def metric_row(name, y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "name": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
    }, pred


def choose_threshold(y_val, prob_val):
    rows = []
    for t in THRESHOLDS:
        row, _ = metric_row("validation", y_val, prob_val, t)
        rows.append({"threshold": t, "precision": row["precision"], "recall": row["recall"], "f1": row["f1"]})
    table = pd.DataFrame(rows)
    selected = float(table.loc[table["f1"].idxmax(), "threshold"])
    return selected, table


def save_confusion(y, pred, name):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay (0)", "Churn (1)"], yticklabels=["Stay (0)", "Churn (1)"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"cm_{name.lower().replace(' ', '_')}.png", dpi=150); plt.close()
    return cm.tolist()


def save_threshold_plot(table, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(table.threshold, table.precision, "bo-", label="Precision")
    plt.plot(table.threshold, table.recall, "rs-", label="Recall")
    plt.plot(table.threshold, table.f1, "g^-", label="F1-score")
    plt.xlabel("Classification threshold"); plt.ylabel("Validation score")
    plt.title(f"Threshold Tuning: {model_name} — validation data only")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_analysis.png", dpi=150); plt.close()


def permutation_importance(model, X, feature_names, n=1000):
    X_sample = X[:min(n, len(X))].copy()
    base = probabilities(model, X_sample)
    rng = np.random.default_rng(SEED)
    values = []
    for i, feature in enumerate(feature_names):
        shuffled = X_sample.copy(); rng.shuffle(shuffled[:, i])
        values.append((feature, float(np.mean(np.abs(probabilities(model, shuffled) - base)))))
    values.sort(key=lambda x: x[1], reverse=True)
    top = values[:15]
    names, scores = zip(*top)
    plt.figure(figsize=(10, 8)); plt.barh(np.arange(len(names)), scores, color="steelblue")
    plt.yticks(np.arange(len(names)), names); plt.gca().invert_yaxis()
    plt.xlabel("Mean absolute change in prediction")
    plt.title("Top 15 Features Driving Customer Churn")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150); plt.close()
    pd.DataFrame(top, columns=["feature", "importance"]).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    return top


def main():
    set_seed()
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y, feature_names = load_and_prepare(DATA_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    models = {"Baseline": baseline_model(X_train.shape[1])}
    histories = {"Baseline": train(models["Baseline"], X_train, y_train, X_val, y_val, None, 50, "Baseline")}
    for name in ("SGD", "RMSprop", "Adam"):
        models[name] = optimized_model(X_train.shape[1], name)
        histories[name] = train(models[name], X_train, y_train, X_val, y_val, class_weight, 100, name)

    val_probs = {name: probabilities(model, X_val) for name, model in models.items()}
    test_probs = {name: probabilities(model, X_test) for name, model in models.items()}

    thresholds = {"Baseline": 0.5}
    for name in ("SGD", "RMSprop", "Adam"):
        thresholds[name], table = choose_threshold(y_val, val_probs[name])
        table.to_csv(OUTPUT_DIR / f"threshold_{name.lower()}.csv", index=False)

    # Primary optimizer-selection metric: validation AUC, then locked test evaluation.
    validation_auc = {name: roc_auc_score(y_val, val_probs[name]) for name in ("SGD", "RMSprop", "Adam")}
    best_name = max(validation_auc, key=validation_auc.get)
    save_threshold_plot(pd.read_csv(OUTPUT_DIR / f"threshold_{best_name.lower()}.csv"), f"{best_name}-optimized — validation")

    rows = []; matrices = {}
    for name in ("Baseline", "SGD", "RMSprop", "Adam"):
        label = "Baseline (SGD, no tuning)" if name == "Baseline" else f"Optimized ({name})"
        row, pred = metric_row(label, y_test, test_probs[name], thresholds[name])
        rows.append(row); matrices[name] = save_confusion(y_test, pred, name)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    permutation_importance(models[best_name], X_train, feature_names)

    manifest = {
        "seed": SEED,
        "dataset": str(DATA_FILE),
        "dataset_sha256": sha256_file(DATA_FILE),
        "n_rows": int(len(y)),
        "n_features_after_encoding": int(X.shape[1]),
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "validation_auc": validation_auc,
        "selected_optimizer_by_validation_auc": best_name,
        "thresholds_selected_on_validation": thresholds,
        "confusion_matrices_test": matrices,
        "tensorflow_version": tf.__version__,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(comparison.to_string(index=False))
    print(f"Selected optimizer by validation AUC: {best_name}")
    print(f"Selected thresholds from validation data: {thresholds}")


if __name__ == "__main__":
    main()
