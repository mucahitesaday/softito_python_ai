# Klasik Makine Öğrenmesi Algoritmaları

Bu bölüm, ders arşivindeki beş klasik makine öğrenmesi konusunu farklı problem
ve veri setleriyle tek bir uygulama paketinde toplar.

| No | Konu | Uygulama | Temel kavram |
|---|---|---|---|
| 01 | Polynomial Regression | Sıcaklığa göre enerji talebi | Derece seçimi, bias–variance |
| 02 | Decision Tree | Çalışan terfi tahmini | Ağaç derinliği, feature importance |
| 03 | KNN vs Naive Bayes | El yazısı rakam tanıma | Uzaklık ve olasılık yaklaşımı |
| 04 | SVM | İç içe geçmiş sınıflar | Linear, polynomial, RBF kernel |
| 05 | Boosting | Makine arıza riski | AdaBoost, Gradient Boosting, XGBoost |

Her proje bağımsız çalışır ve kendi `figures/` ile `results/` klasörünü üretir.
Paket testleri:

```bash
python -m unittest -v MachineLearning/Supervised/03_classic_ml/test_classic_ml.py
```
