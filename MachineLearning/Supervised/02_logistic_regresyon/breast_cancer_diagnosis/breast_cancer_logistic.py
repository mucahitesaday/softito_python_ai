"""Meme tümörü verisiyle ikili Logistic Regression uygulaması."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KLASOR = Path(__file__).resolve().parent
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42


def veriyi_yukle() -> tuple[pd.DataFrame, pd.Series]:
    """569 gözlem ve 30 ölçümden oluşan veri setini döndürür."""
    veri = load_breast_cancer(as_frame=True)
    X = veri.data
    # Orijinal veri 0=malignant, 1=benign biçimindedir. Riskli durumu pozitif yapıyoruz.
    y = (veri.target == 0).astype(int).rename("malignant")
    return X, y


def model_olustur() -> Pipeline:
    return Pipeline(
        [
            ("olcekleme", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metrikleri_hesapla(y_gercek: pd.Series, tahmin: np.ndarray, olasilik: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_gercek, tahmin)), 4),
        "precision": round(float(precision_score(y_gercek, tahmin)), 4),
        "recall": round(float(recall_score(y_gercek, tahmin)), 4),
        "f1": round(float(f1_score(y_gercek, tahmin)), 4),
        "roc_auc": round(float(roc_auc_score(y_gercek, olasilik)), 4),
    }


def esik_analizi(y_gercek: pd.Series, olasilik: np.ndarray) -> pd.DataFrame:
    satirlar = []
    for esik in (0.30, 0.50, 0.70):
        tahmin = (olasilik >= esik).astype(int)
        satirlar.append(
            {
                "threshold": esik,
                "precision": precision_score(y_gercek, tahmin, zero_division=0),
                "recall": recall_score(y_gercek, tahmin, zero_division=0),
                "f1": f1_score(y_gercek, tahmin, zero_division=0),
            }
        )
    return pd.DataFrame(satirlar).round(4)


def modeli_egit(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = model_olustur()
    model.fit(X_train, y_train)
    tahmin = model.predict(X_test)
    olasilik = model.predict_proba(X_test)[:, 1]

    katsayilar = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.named_steps["model"].coef_[0],
        }
    ).sort_values("coefficient", key=abs, ascending=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_skorlari = cross_val_score(model_olustur(), X, y, cv=cv, scoring="roc_auc")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "tahmin": tahmin,
        "olasilik": olasilik,
        "metrikler": metrikleri_hesapla(y_test, tahmin, olasilik),
        "esik_sonuclari": esik_analizi(y_test, olasilik),
        "katsayilar": katsayilar,
        "cv_roc_auc": cv_skorlari,
    }


def gorselleri_uret(y: pd.Series, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y.map({0: "İyi huylu", 1: "Kötü huylu"}), color="#4C78A8")
    plt.xlabel("Sınıf")
    plt.ylabel("Örnek sayısı")
    plt.title("Tümör Sınıf Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_distribution.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        sonuc["y_test"],
        sonuc["tahmin"],
        display_labels=["İyi huylu", "Kötü huylu"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()

    fpr, tpr, _ = roc_curve(sonuc["y_test"], sonuc["olasilik"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {sonuc['metrikler']['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Rastgele tahmin")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curve.png", dpi=140)
    plt.close()

    precision, recall, _ = precision_recall_curve(sonuc["y_test"], sonuc["olasilik"])
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#E45756")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Eğrisi")
    plt.tight_layout()
    plt.savefig(FIGURES / "precision_recall_curve.png", dpi=140)
    plt.close()

    en_guclu = sonuc["katsayilar"].head(12).sort_values("coefficient")
    renkler = np.where(en_guclu["coefficient"] >= 0, "#D62728", "#2CA02C")
    plt.figure(figsize=(9, 6))
    plt.barh(en_guclu["feature"], en_guclu["coefficient"], color=renkler)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standartlaştırılmış katsayı")
    plt.title("Kötü Huylu Sınıfı Etkileyen En Güçlü Özellikler")
    plt.tight_layout()
    plt.savefig(FIGURES / "coefficients.png", dpi=140)
    plt.close()

    esikler = sonuc["esik_sonuclari"].set_index("threshold")
    esikler.plot(marker="o", figsize=(7, 5), color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylim(0, 1.05)
    plt.xlabel("Sınıflandırma eşiği")
    plt.ylabel("Skor")
    plt.title("Eşiğe Göre Precision, Recall ve F1")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "threshold_analysis.png", dpi=140)
    plt.close()


def main() -> None:
    X, y = veriyi_yukle()
    sonuc = modeli_egit(X, y)
    gorselleri_uret(y, sonuc)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rapor = {
        "veri_boyutu": list(X.shape),
        "sinif_sayilari": {"iyi_huylu": int((y == 0).sum()), "kotu_huylu": int((y == 1).sum())},
        "test_metrikleri": sonuc["metrikler"],
        "cv_roc_auc_ortalama": round(float(sonuc["cv_roc_auc"].mean()), 4),
        "cv_roc_auc_std": round(float(sonuc["cv_roc_auc"].std()), 4),
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as dosya:
        json.dump(rapor, dosya, ensure_ascii=False, indent=2)
    sonuc["katsayilar"].to_csv(RESULTS / "coefficients.csv", index=False)
    sonuc["esik_sonuclari"].to_csv(RESULTS / "threshold_metrics.csv", index=False)

    print("MEME TÜMÖRÜ SINIFLANDIRMASI — LOGISTIC REGRESSION")
    print("=" * 55)
    print(f"Veri boyutu: {X.shape}")
    print(f"Sınıf dağılımı: iyi huylu={(y == 0).sum()}, kötü huylu={(y == 1).sum()}")
    print(f"Test metrikleri: {sonuc['metrikler']}")
    print(f"5-fold CV ROC-AUC: {sonuc['cv_roc_auc'].mean():.4f} ± {sonuc['cv_roc_auc'].std():.4f}")
    print("\nEşik karşılaştırması:\n", sonuc["esik_sonuclari"].to_string(index=False))
    print("\nEn güçlü katsayılar:\n", sonuc["katsayilar"].head().to_string(index=False))
    print("\nNot: Bu çıktı tıbbi teşhis amacı taşımaz.")
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
