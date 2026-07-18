# Finansal İşlemlerde COPOD

COPOD'un ampirik dağılım (ECDF), sol/sağ kuyruk olasılığı ve çarpıklık
düzeltmesi adımları sınıf olarak uygulanır. Hazır bir anomali paketi çağırmak
yerine algoritmanın karar skoru doğrudan kod içinde hesaplanır.

## Kapsam

- Korelasyonlu normal finansal işlem verisi
- Dört farklı finansal anomali türü
- Özellik bazında sol ve sağ kuyruk katkıları
- COPOD karar eşiği ve anomali skorları
- Isolation Forest ve One-Class SVM karşılaştırması
- Precision, recall, F1, ROC-AUC ve eğitim süresi
- ROC, özellik katkısı ve bağımlılık grafikleri

```bash
python MachineLearning/AnomalyDetection/03_copod/transaction_copod.py
```
