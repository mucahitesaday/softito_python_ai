# El Yazısı Rakam Tanıma — KNN vs Naive Bayes

Scikit-learn Digits veri setindeki 1797 adet 8×8 görüntü kullanılır. KNN için
farklı `k` değerleri aranır; en iyi KNN, Gaussian Naive Bayes ile accuracy,
Macro-F1 ve tahmin süresi açısından karşılaştırılır.

## Veri seti

Her görüntü 64 piksel özelliğine dönüştürülmüştür ve hedef değer 0–9 arasındaki
rakamdır. Proje sınıf dağılımını, piksel aralığını, sıfır piksel oranını ve her
rakamdan örnek bir görüntüyü raporlar.

## Karşılaştırma

| Model | Accuracy | Macro-F1 | Temel yaklaşım |
|---|---:|---:|---|
| KNN (k=1) | 0.9711 | 0.9706 | Yakın örneklerin uzaklığı |
| GaussianNB | 0.8289 | 0.8305 | Koşullu olasılıklar |

KNN yüksek doğruluk sağlamış ancak tahmin süresi Naive Bayes'ten daha uzun
olmuştur. Beş katlı cross-validation sonuçları ve sınıf bazlı raporlar da
`results/` klasörüne kaydedilir.

```bash
python MachineLearning/Supervised/03_classic_ml/03_knn_naive_bayes/digits_comparison.py
```
