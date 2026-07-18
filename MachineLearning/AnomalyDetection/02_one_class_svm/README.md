# Ağ Trafiğinde One-Class SVM

Normal ağ akışlarıyla öğrenilen doğrusal olmayan sınır; DDoS, port tarama, veri
sızdırma ve kimlik deneme saldırılarını hedef etiketi kullanmadan tespit eder.

## Kapsam

- Ayrı normal eğitim, etiketli doğrulama ve bağımsız test kümeleri
- StandardScaler'ın yalnız eğitim verisinde öğrenilmesi
- RBF One-Class SVM
- `nu` ve `gamma` parametre taraması
- Precision, recall, F1 ve ROC-AUC
- Saldırı türü bazında yakalama oranı
- PCA görünümü, skor dağılımı ve confusion matrix

```bash
python MachineLearning/AnomalyDetection/02_one_class_svm/network_traffic_ocsvm.py
```
