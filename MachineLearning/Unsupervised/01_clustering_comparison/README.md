# Dijital Müşteri Segmentasyonu

## Amaç

Bir e-ticaret sitesindeki 900 müşterinin ziyaret, oturum, sepet, satın alma,
indirim ve iade davranışlarını hedef değişken kullanmadan gruplandırmaktır.
Veride dört temel davranış profiline ek olarak sıra dışı müşteriler ve kontrollü
eksik değerler bulunur.

## Uygulanan analizler

- Veri boyutu, eksik değer, tekrar ve tanımlayıcı istatistik kontrolü
- Median ile eksik değer doldurma ve StandardScaler
- Elbow, Silhouette ve Davies–Bouldin ile küme sayısı seçimi
- K-Means, Agglomerative, DBSCAN ve Gaussian Mixture Model
- Calinski–Harabasz ve Adjusted Rand karşılaştırması
- DBSCAN için `eps` ve `min_samples` taraması
- PCA açıklanan varyans ve yük analizi
- Dendrogram, PCA ve t-SNE görselleştirmeleri
- GMM yumuşak küme üyelik olasılıkları

## Üretilen dosyalar

- `data/digital_customer_behavior.csv`: Ham müşteri davranışları
- `results/model_comparison.csv`: Dört algoritmanın metrikleri
- `results/cluster_profiles.csv`: Segment ortalamaları ve adları
- `results/customer_segments.csv`: Müşteri bazında segment sonucu
- `figures/`: Altı farklı analiz grafiği

```bash
python MachineLearning/Unsupervised/01_clustering_comparison/digital_customer_clustering.py
```
