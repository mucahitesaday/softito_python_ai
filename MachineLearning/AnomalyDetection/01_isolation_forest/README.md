# Üretim Sensörlerinde Isolation Forest

Yalnızca normal üretim kayıtlarıyla eğitilen Isolation Forest modeli; aşırı
ısınma, rulman arızası, basınç kaçağı ve birleşik arıza türlerini tespit eder.

## Kapsam

- Eğitim ve test verisinin ayrı üretilmesi
- Sensör dağılımları ve korelasyonlu çalışma yükü
- StandardScaler ve veri sızıntısını önleyen dönüşüm
- Accuracy, precision, recall, F1 ve ROC-AUC
- Sürekli anomali skorlarının karşılaştırılması
- `contamination` değerinin 0.01–0.12 arasında etkisi
- Arıza türü bazında yakalama oranı
- PCA, confusion matrix ve skor dağılımı grafikleri

```bash
python MachineLearning/AnomalyDetection/01_isolation_forest/production_sensor_isolation.py
```
