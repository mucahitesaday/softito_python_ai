# Makine Arıza Riski — Boosting Karşılaştırması

AdaBoost ve Gradient Boosting, tekrarlanabilir sensör verisinde karşılaştırılır.
Ortamda `xgboost` kuruluysa XGBoost da otomatik olarak karşılaştırmaya katılır;
kurulu değilse proje scikit-learn modelleriyle çalışmaya devam eder.

## Veri seti

1800 makine kaydında sıcaklık, titreşim, basınç, RPM, yağ kalitesi, yük, makine
yaşı, hata sayısı ve dört ek sensör bulunur. Hedef sütun, yaklaşan arıza riskidir.
Veri `data/machine_sensor_data.csv` dosyasına kaydedilir.

## Yapılan işlemler

- Eksik/tekrar kontrolü ve sensörlerin tanımlayıcı istatistikleri
- Arızayla en güçlü korelasyona sahip sensörlerin incelenmesi
- AdaBoost, Gradient Boosting ve varsa XGBoost eğitimi
- Accuracy, precision, recall, F1 ve ROC-AUC
- Beş katlı cross-validation ve eğitim süresi
- ROC eğrileri, confusion matrix ve feature importance

XGBoost bağımlılığı zorunlu değildir. Böylece temel proje her ortamda çalışır;
paket kurulduğunda üçüncü model otomatik eklenir.

```bash
python MachineLearning/Supervised/03_classic_ml/05_boosting/maintenance_boosting.py
```

İsteğe bağlı gerçek XGBoost modeli:

```bash
python -m pip install xgboost
```
