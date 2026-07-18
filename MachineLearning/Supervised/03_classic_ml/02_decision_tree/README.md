# Çalışan Terfi Tahmini — Decision Tree

1200 satırlık tekrarlanabilir çalışan verisinde eğitim, performans, deneyim ve
departman bilgileri kullanılır. GridSearchCV ile derinlik ve yaprak büyüklüğü
seçilir; karar ağacı ile özellik önemleri görselleştirilir.

## Değişkenler

| Sütun | Açıklama |
|---|---|
| `experience_years` | Toplam iş deneyimi |
| `performance_score` | 40–100 performans puanı |
| `training_hours` | Yıllık eğitim süresi |
| `projects_completed` | Tamamlanan proje sayısı |
| `department` | Çalışılan departman |
| `education` | Eğitim seviyesi |
| `promoted` | Hedef: terfi durumu |

## Yapılan işlemler

- Eksik değer, tekrar, sayısal özet ve kategori bazlı terfi oranları
- OneHotEncoder ve ColumnTransformer
- Gini/entropy, max_depth ve min_samples_leaf için GridSearchCV
- Train/test accuracy ile overfitting kontrolü
- Accuracy, F1, ROC-AUC ve classification report
- Karar kuralları, ağaç şeması ve feature importance

En iyi modelde performans puanı en güçlü değişken olmuş; ardından eğitim saati,
deneyim ve tamamlanan proje sayısı gelmiştir. Üretilen karar kuralları
`results/decision_rules.txt` dosyasındadır.

```bash
python MachineLearning/Supervised/03_classic_ml/02_decision_tree/promotion_tree.py
```
