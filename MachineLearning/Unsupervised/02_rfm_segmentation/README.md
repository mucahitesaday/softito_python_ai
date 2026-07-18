# E-Ticaret RFM Müşteri Segmentasyonu

## Amaç

Fatura satırlarından müşteri bazında **Recency**, **Frequency** ve **Monetary**
özelliklerini üretmek; müşterileri hem beşli RFM skorlarıyla hem de K-Means ile
segmentlere ayırmaktır. Veri seti 600 müşterinin bir yıllık alışveriş geçmişini,
ürün satırlarını, iptal ve iade kayıtlarını içerir.

## İşlem adımları

- İptal, iade, negatif adet ve geçersiz fiyatların temizlenmesi
- Müşteri bazında RFM ve ortalama fatura değeri hesaplanması
- RFM değişkenlerine `log1p` dönüşümü ve StandardScaler
- K=2–8 için Elbow, Silhouette ve Davies–Bouldin analizi
- K-Means müşteri segmentasyonu
- Beşli R, F ve M skorları ile üç haneli RFM skoru
- Segment profili, büyüklüğü ve üç boyutlu RFM görselleştirmesi

Ham işlemler `data/`; müşteri segmentleri, profil tablosu ve metrikler
`results/`; grafikler `figures/` klasörüne kaydedilir.

```bash
python MachineLearning/Unsupervised/02_rfm_segmentation/ecommerce_rfm.py
```
