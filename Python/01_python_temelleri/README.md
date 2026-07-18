# 01 — Python Temelleri: Iris Veri Analizi

Bu çalışmada Python'ın temel özellikleri, yapay olarak yazılmış birkaç sayı
yerine gerçek **Iris çiçeği veri seti** üzerinde uygulanmıştır. Veri setinde
150 çiçeğin çanak ve taç yaprak ölçümleri ile tür bilgileri bulunur.

## Uygulanan konular

- Değişkenler ve veri tipleri
- Liste ve sözlük yapıları
- `for` döngüsü ve koşullar
- Fonksiyon yazımı
- List comprehension
- Dosya okuma ve yazma (`CSV`, `JSON`)
- Hata yönetimi (`try/except`)
- Sınıf oluşturma ve `@property`
- Modülün `main` bloğu ile çalıştırılması

## Çalıştırma

Repo ana klasöründeyken:

```bash
python Python/01_python_temelleri/python_temelleri_iris.py
```

Kod ilk çalışmada Iris veri setini scikit-learn üzerinden alıp `data/iris.csv`
dosyasını oluşturur. Daha sonra analiz sonuçlarını ekrana yazdırır ve aşağıdaki
dosyalara kaydeder:

- `outputs/tur_ozeti.csv`
- `outputs/analiz_raporu.json`

Veri dosyasını yeniden oluşturmak için:

```bash
python Python/01_python_temelleri/python_temelleri_iris.py --veriyi-yenile
```

## Beklenen kısa sonuç

Program 150 satır okur. Üç Iris türünün her birinde 50 örnek olduğunu gösterir
ve her türe ait ölçüm ortalamalarını hesaplar.
