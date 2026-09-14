"""Pytest tests for ann_optimization_corrected.py."""
import numpy as np
import pandas as pd
import pytest

import ann_optimization_corrected as app


def test_dataset_loads_and_has_expected_shape():
    X, y, feature_names = app.load_and_prepare(app.DATA_FILE)
    assert len(y) == 7043
    assert X.shape[0] == 7043
    assert len(feature_names) == X.shape[1]
    assert set(np.unique(y)) == {0, 1}


def test_preprocessing_has_no_nan_or_infinite_values():
    X, y, _ = app.load_and_prepare(app.DATA_FILE)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_split_sizes_and_stratification():
    X, y, _ = app.load_and_prepare(app.DATA_FILE)
    X_train, X_val, X_test, y_train, y_val, y_test = app.split_and_scale(X, y)
    assert len(y_train) == 4507
    assert len(y_val) == 1127
    assert len(y_test) == 1409
    assert abs(y_train.mean() - y.mean()) < 0.02
    assert abs(y_val.mean() - y.mean()) < 0.02
    assert abs(y_test.mean() - y.mean()) < 0.02
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()


@pytest.mark.parametrize("optimizer_name", ["SGD", "RMSprop", "Adam"])
def test_optimized_model_builds(optimizer_name):
    model = app.optimized_model(30, optimizer_name)
    assert model.input_shape == (None, 30)
    assert model.output_shape == (None, 1)
    assert model.optimizer is not None


def test_baseline_model_builds():
    model = app.baseline_model(30)
    assert model.input_shape == (None, 30)
    assert model.output_shape == (None, 1)


def test_metric_row_is_correct_for_known_values():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.8, 0.7, 0.2])
    row, pred = app.metric_row("test", y, prob, 0.5)
    assert np.array_equal(pred, np.array([0, 1, 1, 0]))
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["f1"] == pytest.approx(0.5)


def test_threshold_selection_uses_validation_labels():
    y_val = np.array([0, 0, 1, 1, 1, 1])
    prob_val = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.9])
    threshold, table = app.choose_threshold(y_val, prob_val)
    assert threshold in app.THRESHOLDS
    assert list(table.columns) == ["threshold", "precision", "recall", "f1"]


def test_thresholds_are_valid():
    assert all(0 < threshold < 1 for threshold in app.THRESHOLDS)


def test_class_labels_are_binary():
    X, y, _ = app.load_and_prepare(app.DATA_FILE)
    assert set(np.unique(y)).issubset({0, 1})


def test_dataset_hash_is_reproducible():
    first = app.sha256_file(app.DATA_FILE)
    second = app.sha256_file(app.DATA_FILE)
    assert first == second
    assert len(first) == 64
