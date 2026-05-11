# Data Science Project 30 — Reklam → Satış: Polinom + Regularization

**Modül**: ML-02 (Regresyon) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir e-ticaret şirketinde **data scientist** olarak çalışıyorsun. Pazarlama ekibi şunu soruyor:

> *"Reklam harcamasından satışı tahmin eden bir model kursak ne kadar isabetli olur? Özellikle harcama yüksekken doyum noktasını görmemiz lazım."*

Reklam → satış ilişkisi **doğrusal değil**: ilk binlerde etki büyük, milyonlara çıkınca **azalan marjinal fayda** var. Bu projede:

1. **Polinom regresyonla** eğri ilişkiyi modellersin
2. Derece arttıkça **overfit'i sayıyla görürsün**
3. **Ridge** ve **Lasso regularization** ile modeli kontrol altına alırsın
4. **GridSearchCV** ile en iyi alpha'yı otomatik bulursun
5. **4 metrik** (MAE, MSE, RMSE, R²) ile değerlendirirsin
6. **Residual analizi** ile model sağlığını kontrol edersin

Bu projede ML-02 dersinde öğrendiklerini birleştirip uygulayacaksın:

- ✅ **PolynomialFeatures** ile feature genişletme
- ✅ Train/test gap ile **overfit teşhisi**
- ✅ **Ridge (L2)** — ağırlıkları küçültme, default tercih
- ✅ **Lasso (L1)** — otomatik feature seçimi (sparse model)
- ✅ **Pipeline** kullanımı: PolynomialFeatures → StandardScaler → Ridge/Lasso
- ✅ **GridSearchCV** + 5-fold cross-validation
- ✅ **MAE, MSE, RMSE, R²** — metrik karşılaştırma
- ✅ **Residual analizi** ile model sağlık kontrolü

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-30

# Virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# veya: venv\Scripts\activate     # Windows

# Bağımlılıklar
pip install -r requirements.txt
```

## 📝 Yapacakların

`tasks/task_manager.py` içinde **11 fonksiyon**:

| # | Fonksiyon | Ne Yapar |
|---|-----------|----------|
| 1 | `generate_sales_data()` | Sentetik reklam→satış verisi üret |
| 2 | `split_data()` | Train/test böl |
| 3 | `train_polynomial()` | PolynomialFeatures + LinearRegression |
| 4 | `compare_degrees()` | Farklı derecelerde train/test R² tablosu |
| 5 | `detect_overfit()` | Gap threshold kontrolü |
| 6 | `train_ridge_pipeline()` | Poly + Scaler + Ridge |
| 7 | `train_lasso_pipeline()` | Poly + Scaler + Lasso |
| 8 | `count_nonzero_coefs()` | Sparse model tespiti |
| 9 | `compute_all_metrics()` | MAE, MSE, RMSE, R² |
| 10 | `find_best_alpha()` | GridSearchCV ile otomatik seçim |
| 11 | `analyze_residuals()` | Residual istatistikleri |

Her fonksiyonun `pass` kısmını docstring'e göre doldur.

## ✅ Testleri Çalıştır

**Otomatik (önerilen)**: dosya değiştikçe testleri çalıştırır
```bash
python watch.py
```

**Manuel**:
```bash
pytest tests/test_question.py -v
```

Toplam **21 test**. Hepsi PASS olunca projen tamam.

## 💡 İpuçları ve Beklenen Sonuçlar

### Polinom Derece Seçimi
```
Derece 1  → underfit (eğri lineerle modellenemez)
Derece 3-5 → ideal (eğri pürüzsüz, gap küçük)
Derece 10+ → overfit (train iyi, test düşük)
```

### Pipeline Yapıları
```python
# Polinom (regularizationsız)
make_pipeline(PolynomialFeatures(d), LinearRegression())
   → 2 adım

# Ridge / Lasso (regularizationlı)
make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge(alpha))
   → 3 adım, scaling ŞART
```

### Ridge vs Lasso
- **Ridge**: tüm katsayıları küçültür ama sıfırlamaz → `count_nonzero_coefs() ≈ N`
- **Lasso**: bazı katsayıları tam sıfır yapar → `count_nonzero_coefs() < N`

### Lasso `max_iter` Uyarısı
Lasso iteratif çözüm gerekir (L1 cezası sıfırda türevsiz). `max_iter=1000` default değeri **çoğu zaman yetmez**. `100_000` koy, `ConvergenceWarning` görmezsen tamam.

### Metrikler İlişkisi
- `RMSE ≥ MAE` her zaman (kare etkisi büyük hataları abartır)
- Perfect tahminde her ikisi 0, R² = 1
- Outlier varsa: RMSE >> MAE

## 🎓 Pedagojik Hedefler

Bu projeyi bitirdiğinde:
- Lineer modelin sınırını ve polinomun avantajını biliyorsun
- Overfit'i sayıyla (gap) ve teşhis edebiliyorsun
- Ridge ve Lasso'nun arasındaki **matematiksel farkı** (kare vs mutlak değer) uygulamada görüyorsun
- GridSearchCV ile hiperparametre seçimini **doğru** yapıyorsun
- Residual analizi ile model sağlığını **metrik dışı** kontrol ediyorsun

## 🚀 Sonraki Adımlar

Bu projeyi bitirdiğinde **ML-03 (Sınıflandırma I)** modülüne geç. Orada artık sayı yerine **kategori** tahmin etmeye başlayacaksın — Logistic Regression, KNN, Confusion Matrix.

İyi çalışmalar 📊💰
