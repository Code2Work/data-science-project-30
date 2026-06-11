import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    generate_sales_data,
    split_data,
    train_polynomial,
    compare_degrees,
    detect_overfit,
    train_ridge_pipeline,
    train_lasso_pipeline,
    count_nonzero_coefs,
    compute_all_metrics,
    find_best_alpha,
    analyze_residuals,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline


# ─── 1. generate_sales_data ─────────────────────────────────

def test_generate_sales_data_shape():
    X, y = generate_sales_data(n=100)
    assert X.shape == (100, 1)
    assert y.shape == (100,)


def test_generate_sales_data_deterministic():
    """Aynı seed → aynı veri"""
    X1, y1 = generate_sales_data(n=50, seed=42)
    X2, y2 = generate_sales_data(n=50, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ─── 2. split_data ──────────────────────────────────────────

def test_split_data_proportions():
    X, y = generate_sales_data(n=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.3)
    assert len(X_tr) == 70
    assert len(X_te) == 30


# ─── 3. train_polynomial ────────────────────────────────────

def test_train_polynomial_returns_pipeline():
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_polynomial(X_tr, y_tr, degree=3)
    assert isinstance(model, Pipeline)
    # PolynomialFeatures + LinearRegression iki adım
    assert len(model.steps) == 2


def test_train_polynomial_fits_well_for_curve():
    """Polinom derece 3+ ile bu eğri için R² > 0.85 olmalı"""
    X, y = generate_sales_data(n=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = train_polynomial(X_tr, y_tr, degree=3)
    assert model.score(X_te, y_te) > 0.85


# ─── 4. compare_degrees ─────────────────────────────────────

def test_compare_degrees_returns_dataframe():
    X, y = generate_sales_data(n=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    df = compare_degrees(X_tr, X_te, y_tr, y_te, [1, 3, 5])
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'degree', 'train_r2', 'test_r2', 'gap'}
    assert len(df) == 3


def test_compare_degrees_higher_degree_better_train():
    """Yüksek dereceli model train'de daha iyi olmalı"""
    X, y = generate_sales_data(n=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    df = compare_degrees(X_tr, X_te, y_tr, y_te, [1, 5])
    deg1_train = df[df['degree'] == 1]['train_r2'].iloc[0]
    deg5_train = df[df['degree'] == 5]['train_r2'].iloc[0]
    assert deg5_train > deg1_train


# ─── 5. detect_overfit ──────────────────────────────────────

def test_detect_overfit_true_when_big_gap():
    assert detect_overfit(0.95, 0.40) is True


def test_detect_overfit_false_when_small_gap():
    assert detect_overfit(0.85, 0.82) is False


def test_detect_overfit_custom_threshold():
    assert detect_overfit(0.90, 0.85, threshold=0.10) is False
    assert detect_overfit(0.90, 0.85, threshold=0.03) is True


# ─── 6. train_ridge_pipeline ────────────────────────────────

def test_train_ridge_returns_pipeline_with_scaling():
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_ridge_pipeline(X_tr, y_tr, degree=5, alpha=0.1)
    assert isinstance(model, Pipeline)
    # PolynomialFeatures + StandardScaler + Ridge → 3 adım
    assert len(model.steps) == 3
    # Son adım Ridge olmalı
    assert isinstance(model.steps[-1][1], Ridge)


def test_train_ridge_alpha_set():
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_ridge_pipeline(X_tr, y_tr, degree=5, alpha=0.5)
    assert model.steps[-1][1].alpha == 0.5


# ─── 7. train_lasso_pipeline ────────────────────────────────

def test_train_lasso_returns_pipeline():
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_lasso_pipeline(X_tr, y_tr, degree=5, alpha=0.01)
    assert isinstance(model, Pipeline)
    assert isinstance(model.steps[-1][1], Lasso)


# ─── 8. count_nonzero_coefs ─────────────────────────────────

def test_count_nonzero_ridge_keeps_most():
    """Ridge çoğu katsayıyı tutar"""
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_ridge_pipeline(X_tr, y_tr, degree=10, alpha=0.1)
    n_total = 11  # degree=10 → 11 katsayı (1, x, x², ..., x^10)
    nonzero = count_nonzero_coefs(model)
    assert nonzero >= n_total - 1  # Ridge tam sıfır yapmaz (en fazla 1 yuvarlama)


def test_count_nonzero_lasso_zeros_some():
    """Lasso bazı katsayıları sıfır yapar"""
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    model = train_lasso_pipeline(X_tr, y_tr, degree=10, alpha=0.05)
    n_total = 11
    nonzero = count_nonzero_coefs(model)
    assert nonzero < n_total  # Lasso en az bir tanesini sıfırlamış olmalı


# ─── 9. compute_all_metrics ─────────────────────────────────

def test_compute_all_metrics_returns_dict():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = compute_all_metrics(y_true, y_pred)
    assert set(m.keys()) == {'mae', 'mse', 'rmse', 'r2'}


def test_compute_all_metrics_rmse_geq_mae():
    """RMSE her zaman MAE ≥ olur"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 1.8, 3.5, 3.7, 4.5])
    m = compute_all_metrics(y_true, y_pred)
    assert m['rmse'] >= m['mae']


def test_compute_all_metrics_perfect_prediction():
    """Mükemmel tahminde MAE=MSE=RMSE=0, R²=1"""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_all_metrics(y_true, y_pred)
    assert abs(m['mae']) < 1e-10
    assert abs(m['rmse']) < 1e-10
    assert abs(m['r2'] - 1.0) < 1e-10


# ─── 10. find_best_alpha ────────────────────────────────────

def test_find_best_alpha_returns_dict():
    X, y = generate_sales_data(n=100)
    X_tr, _, y_tr, _ = split_data(X, y)
    result = find_best_alpha(X_tr, y_tr, degree=5, alphas=[0.001, 0.01, 0.1, 1, 10])
    assert set(result.keys()) == {'best_alpha', 'best_cv_r2', 'best_model'}
    assert result['best_alpha'] in [0.001, 0.01, 0.1, 1, 10]


# ─── 11. analyze_residuals ──────────────────────────────────

def test_analyze_residuals_format():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    r = analyze_residuals(y_true, y_pred)
    assert set(r.keys()) == {'mean', 'std', 'max_abs', 'residuals'}


def test_analyze_residuals_mean_near_zero_good_model():
    """İyi modelde residual ortalaması ~0 olur"""
    X, y = generate_sales_data(n=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = train_polynomial(X_tr, y_tr, degree=3)
    y_pred = model.predict(X_te)
    r = analyze_residuals(y_te, y_pred)
    assert abs(r['mean']) < 0.3


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
