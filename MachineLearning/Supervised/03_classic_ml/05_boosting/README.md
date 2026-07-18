# Makine Arıza Riski — Boosting Karşılaştırması

AdaBoost ve Gradient Boosting, tekrarlanabilir sensör verisinde karşılaştırılır.
Ortamda `xgboost` kuruluysa XGBoost da otomatik olarak karşılaştırmaya katılır;
kurulu değilse proje scikit-learn modelleriyle çalışmaya devam eder.

```bash
python MachineLearning/Supervised/03_classic_ml/05_boosting/maintenance_boosting.py
```

İsteğe bağlı gerçek XGBoost modeli:

```bash
python -m pip install xgboost
```
