# İç İçe Geçmiş Sınıflar — SVM Kernel Analizi

`make_moons` verisinde linear, polynomial ve RBF kernel karşılaştırılır. RBF
modeli için GridSearchCV ile `C` ve `gamma` seçilir; karar sınırları çizilir.

## Amaç

İki sınıf doğrusal bir çizgiyle tam ayrılamadığı için kernel seçiminin karar
sınırını nasıl değiştirdiği görsel olarak gösterilir. Veri seti 700 satır ve iki
sayısal sensör özelliğinden oluşur.

## Yapılan işlemler

- Veri boyutu, eksik/tekrar kontrolü, sayısal özet ve sınıf dağılımı
- StandardScaler ve SVC Pipeline'ı
- Linear, üçüncü derece polynomial ve RBF kernel
- RBF için `C × gamma` GridSearchCV
- Train/test accuracy, F1, ROC-AUC ve support vector sayısı
- Karar sınırı, ROC karşılaştırması ve confusion matrix

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Linear | 0.8800 | 0.8814 | 0.9530 |
| Polynomial | 0.8914 | 0.8939 | 0.9643 |
| RBF | 0.9200 | 0.9205 | 0.9832 |

```bash
python MachineLearning/Supervised/03_classic_ml/04_svm/moons_svm.py
```
