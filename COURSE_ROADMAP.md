# Softito Yapay Zekâ Yazılımcılığı — Ders Kapsamı

Bu yol haritası, eğitim boyunca kullanılan `python_ai_examples` ders arşivindeki
konular incelenerek hazırlanmıştır. Tekrar eden notebooklar aynı modülde
birleştirilir; her konu farklı örnekler, yeniden çalıştırılabilir kodlar,
sonuçlar, grafikler ve testlerle bu repoya aktarılır.

## 1. Python ve veri bilimi temelleri

- [x] Temel Python, veri tipleri, koşullar, döngüler ve fonksiyonlar
- [x] Koleksiyonlar, dosya işlemleri ve hata yönetimi
- [x] OOP, kalıtım, mixin, iterator ve dunder metotlar
- [x] Pandas ile veri yükleme, temizleme, EDA ve feature engineering

## 2. Denetimli makine öğrenmesi

- [x] Basit ve çoklu Linear Regression
- [x] Logistic Regression: ikili ve çok sınıflı sınıflandırma
- [x] Polynomial Regression ve bias–variance dengesi
- [x] Decision Tree ve hiperparametre optimizasyonu
- [x] KNN ve Naive Bayes karşılaştırması
- [x] Linear, polynomial ve RBF kernel SVM
- [x] AdaBoost, Gradient Boosting ve opsiyonel XGBoost karşılaştırması

## 3. Denetimsiz öğrenme ve müşteri analitiği

- [ ] K-Means, Agglomerative Clustering ve DBSCAN
- [ ] PCA ve küme kalitesi metrikleri
- [ ] RFM müşteri segmentasyonu

## 4. Anomali tespiti ve model izleme

- [ ] Isolation Forest
- [ ] One-Class SVM
- [ ] COPOD
- [ ] Data drift: KS testi, PSI ve dağılım karşılaştırması

## 5. Optimizasyon algoritmaları

- [ ] Genetic Algorithms
- [ ] CMA-ES
- [ ] Particle Swarm Optimization

## 6. Derin öğrenme ve sıralı modeller

- [ ] Nöron, aktivasyon, loss, gradient descent ve backpropagation
- [ ] ANN ve PyTorch eğitim döngüsü
- [ ] Simple RNN ve zaman serisi tahmini
- [ ] LSTM hücresi, kapılar ve sınıflandırma
- [ ] Self-attention, Q/K/V ve multi-head attention
- [ ] Positional encoding ve temel Transformer
- [ ] Softmax temperature, top-k/top-p ve autoregression
- [ ] Karakter seviyesinde Small Language Model

## 7. Computer Vision

- [ ] Piksel, kanal, renk uzayı ve görüntü dönüşümleri
- [ ] Filtreleme, kenar bulma, threshold ve morfolojik işlemler
- [ ] Augmentation, PCA ve t-SNE
- [ ] NumPy ile convolution, pooling ve feature map
- [ ] CNN ile görüntü sınıflandırma
- [ ] Haar Cascade ile yüz tespiti

## 8. NLP ve metin temsili

- [ ] Metin temizleme ve tokenization
- [ ] TF-IDF: sıfırdan ve scikit-learn
- [ ] Word2Vec, GloVe ve FastText
- [ ] Haber sentiment sınıflandırması

## 9. LLM, fine-tuning ve RAG

- [ ] BPE, WordPiece, Unigram ve Hugging Face Dataset hazırlama
- [ ] Fine-tuning veri formatı ve eğitim yaşam döngüsü
- [ ] PEFT, LoRA ve QLoRA
- [ ] Quantization ve LLM optimizasyonu
- [ ] Embedding, semantic search ve vector database
- [ ] RAG mimarisi ve uçtan uca RAG pipeline

## 10. Veri mühendisliği, MLOps ve DevOps

- [ ] ETL ve Airflow DAG mantığı
- [ ] Dockerfile ve Docker Compose
- [ ] Mini DFS, YARN ve Spark kavramları
- [ ] Model/veri izleme ve üretim kontrolleri

## Uygulama standardı

Her tamamlanan modül mümkün olduğunca şunları içerir:

1. Konuyu ve veri setini açıklayan README
2. İnternetsiz yeniden çalışabilen Python dosyası
3. Sabit random seed ve tekrarlanabilir sonuçlar
4. Uygun performans metrikleri ve görseller
5. CSV/JSON sonuç dosyaları
6. Otomatik testler
7. Ayrı branch, yerel doğrulama ve pull request
