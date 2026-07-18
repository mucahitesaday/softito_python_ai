"""Digits veri setinde ayrıntılı KNN ve Gaussian Naive Bayes karşılaştırması."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KLASOR = Path(__file__).resolve().parent
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
K_DEGERLERI = (1, 3, 5, 7, 9, 11)


def veriyi_yukle():
    veri = load_digits(as_frame=True)
    return veri.data, veri.target.rename("digit"), veri.images


def veriyi_incele(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "shape": X.shape,
        "missing_values": int(X.isna().sum().sum()),
        "class_counts": y.value_counts().sort_index(),
        "pixel_min": float(X.min().min()),
        "pixel_max": float(X.max().max()),
        "zero_pixel_ratio": float((X == 0).to_numpy().mean()),
    }


def knn_olustur(k: int) -> Pipeline:
    return Pipeline(
        [
            ("olcekleme", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )


def modeli_egit(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    k_sonuclari = []
    modeller = {}
    for k in K_DEGERLERI:
        model = knn_olustur(k)
        model.fit(X_train, y_train)
        tahmin = model.predict(X_test)
        k_sonuclari.append(
            {
                "k": k,
                "accuracy": accuracy_score(y_test, tahmin),
                "f1_macro": f1_score(y_test, tahmin, average="macro"),
            }
        )
        modeller[k] = model

    k_df = pd.DataFrame(k_sonuclari)
    en_iyi_k = int(k_df.loc[k_df["accuracy"].idxmax(), "k"])
    adaylar = {f"KNN (k={en_iyi_k})": modeller[en_iyi_k], "GaussianNB": GaussianNB()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    satirlar = []
    tahminler = {}
    raporlar = {}

    for ad, model in adaylar.items():
        baslangic = time.perf_counter()
        model.fit(X_train, y_train)
        egitim_suresi = (time.perf_counter() - baslangic) * 1000
        baslangic = time.perf_counter()
        tahmin = model.predict(X_test)
        tahmin_suresi = (time.perf_counter() - baslangic) * 1000
        cv_skorlari = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        tahminler[ad] = tahmin
        raporlar[ad] = pd.DataFrame(classification_report(y_test, tahmin, output_dict=True)).T
        satirlar.append(
            {
                "model": ad,
                "accuracy": accuracy_score(y_test, tahmin),
                "f1_macro": f1_score(y_test, tahmin, average="macro"),
                "cv_f1_mean": cv_skorlari.mean(),
                "cv_f1_std": cv_skorlari.std(),
                "fit_ms": egitim_suresi,
                "prediction_ms": tahmin_suresi,
            }
        )

    return {
        "X_test": X_test,
        "y_test": y_test,
        "k_results": k_df,
        "best_k": en_iyi_k,
        "comparison": pd.DataFrame(satirlar),
        "predictions": tahminler,
        "reports": raporlar,
    }


def gorselleri_uret(images: np.ndarray, y: pd.Series, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for rakam, ax in enumerate(axes.ravel()):
        indeks = int(np.flatnonzero(y.to_numpy() == rakam)[0])
        ax.imshow(images[indeks], cmap="gray_r")
        ax.set_title(f"Rakam {rakam}")
        ax.axis("off")
    fig.suptitle("Digits Veri Setinden Örnekler")
    fig.tight_layout()
    fig.savefig(FIGURES / "digit_samples.png", dpi=140)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    sns.barplot(x=sonuc["y_test"].value_counts().sort_index().index,
                y=sonuc["y_test"].value_counts().sort_index().values,
                color="#4C78A8")
    plt.xlabel("Rakam")
    plt.ylabel("Test örneği")
    plt.title("Test Kümesi Sınıf Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_distribution.png", dpi=140)
    plt.close()

    sonuc["k_results"].plot(x="k", y=["accuracy", "f1_macro"], marker="o", figsize=(7, 4))
    plt.ylim(0.85, 1.0)
    plt.title("K Değerine Göre KNN Performansı")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "k_selection.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ad, tahmin) in zip(axes, sonuc["predictions"].items()):
        ConfusionMatrixDisplay.from_predictions(
            sonuc["y_test"], tahmin, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(ad)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=140)
    plt.close(fig)


def main() -> None:
    X, y, images = veriyi_yukle()
    inceleme = veriyi_incele(X, y)
    sonuc = modeli_egit(X, y)
    gorselleri_uret(images, y, sonuc)

    RESULTS.mkdir(parents=True, exist_ok=True)
    sonuc["k_results"].round(4).to_csv(RESULTS / "k_results.csv", index=False)
    sonuc["comparison"].round(4).to_csv(RESULTS / "model_comparison.csv", index=False)
    for ad, rapor in sonuc["reports"].items():
        dosya_adi = "knn_report.csv" if ad.startswith("KNN") else "naive_bayes_report.csv"
        rapor.round(4).to_csv(RESULTS / dosya_adi)
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(
            {"rows": len(X), "features": X.shape[1], "best_k": sonuc["best_k"]},
            dosya,
            indent=2,
        )

    print("KNN vs NAIVE BAYES — RAKAM TANIMA")
    print("=" * 38)
    print(f"Veri boyutu: {inceleme['shape']}")
    print(f"Eksik değer: {inceleme['missing_values']}")
    print(f"Piksel aralığı: {inceleme['pixel_min']:.0f}–{inceleme['pixel_max']:.0f}")
    print(f"Sıfır değerli piksel oranı: %{inceleme['zero_pixel_ratio']*100:.1f}")
    print("\nSınıf dağılımı:\n", inceleme["class_counts"].to_string())
    print("\nK seçimi:\n", sonuc["k_results"].round(4).to_string(index=False))
    print(f"\nEn iyi k: {sonuc['best_k']}")
    print("\nModel karşılaştırması:\n", sonuc["comparison"].round(4).to_string(index=False))
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
