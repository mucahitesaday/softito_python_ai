"""Diabetes veri setiyle basit ve çoklu doğrusal regresyon."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

KLASOR = Path(__file__).resolve().parent
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42


def veriyi_yukle() -> tuple[pd.DataFrame, pd.Series]:
    veri = load_diabetes(as_frame=True)
    return veri.data, veri.target


def metrikleri_hesapla(gercek: pd.Series, tahmin: np.ndarray) -> dict[str, float]:
    return {
        "r2": round(float(r2_score(gercek, tahmin)), 4),
        "mae": round(float(mean_absolute_error(gercek, tahmin)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(gercek, tahmin))), 4),
    }


def modelleri_egit(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    basit = LinearRegression().fit(X_train[["bmi"]], y_train)
    basit_tahmin = basit.predict(X_test[["bmi"]])

    coklu = LinearRegression().fit(X_train, y_train)
    coklu_tahmin = coklu.predict(X_test)
    cv_skorlari = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
    katsayilar = pd.DataFrame(
        {"feature": X.columns, "coefficient": coklu.coef_}
    ).sort_values("coefficient", key=abs, ascending=False)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "basit_model": basit,
        "coklu_model": coklu,
        "basit_tahmin": basit_tahmin,
        "coklu_tahmin": coklu_tahmin,
        "basit_metrikler": metrikleri_hesapla(y_test, basit_tahmin),
        "coklu_metrikler": metrikleri_hesapla(y_test, coklu_tahmin),
        "cv_r2_skorlari": cv_skorlari,
        "katsayilar": katsayilar,
    }


def gorselleri_uret(X: pd.DataFrame, y: pd.Series, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    korelasyon_verisi = X.copy()
    korelasyon_verisi["target"] = y
    plt.figure(figsize=(10, 8))
    sns.heatmap(korelasyon_verisi.corr(), cmap="coolwarm", center=0, square=True)
    plt.title("Diabetes Özellikleri Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_matrix.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(X["bmi"], y, alpha=0.5, color="#4C78A8")
    bmi_cizgi = np.linspace(X["bmi"].min(), X["bmi"].max(), 100)
    tahmin_cizgi = sonuc["basit_model"].predict(pd.DataFrame({"bmi": bmi_cizgi}))
    plt.plot(bmi_cizgi, tahmin_cizgi, color="crimson", linewidth=2)
    plt.xlabel("Standartlaştırılmış BMI")
    plt.ylabel("İlerleme skoru")
    plt.title("Basit Regresyon: BMI → Hastalık İlerleme Skoru")
    plt.tight_layout()
    plt.savefig(FIGURES / "simple_bmi_regression.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(data=sonuc["katsayilar"], x="coefficient", y="feature", color="#6C5CE7")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Çoklu Regresyon Katsayıları")
    plt.tight_layout()
    plt.savefig(FIGURES / "coefficients.png", dpi=140, bbox_inches="tight")
    plt.close()

    gercek = sonuc["y_test"]
    tahmin = sonuc["coklu_tahmin"]
    plt.figure(figsize=(6, 6))
    plt.scatter(gercek, tahmin, alpha=0.7, color="#00A896")
    sinirlar = [min(gercek.min(), tahmin.min()), max(gercek.max(), tahmin.max())]
    plt.plot(sinirlar, sinirlar, "r--")
    plt.xlabel("Gerçek skor")
    plt.ylabel("Tahmin edilen skor")
    plt.title("Gerçek ve Tahmin Edilen İlerleme Skoru")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=140, bbox_inches="tight")
    plt.close()

    artiklar = gercek.to_numpy() - tahmin
    plt.figure(figsize=(8, 5))
    sns.histplot(artiklar, kde=True, color="#F4A261")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Artık")
    plt.title("Tahmin Hatalarının Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_distribution.png", dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    X, y = veriyi_yukle()
    sonuc = modelleri_egit(X, y)
    gorselleri_uret(X, y, sonuc)

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrikler = {
        "veri_satiri": len(X),
        "ozellik_sayisi": X.shape[1],
        "basit_regresyon": sonuc["basit_metrikler"],
        "coklu_regresyon": sonuc["coklu_metrikler"],
        "cv_r2_ortalama": round(float(sonuc["cv_r2_skorlari"].mean()), 4),
        "cv_r2_std": round(float(sonuc["cv_r2_skorlari"].std()), 4),
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as dosya:
        json.dump(metrikler, dosya, ensure_ascii=False, indent=2)
    sonuc["katsayilar"].to_csv(RESULTS / "coefficients.csv", index=False)

    print("DİYABET İLERLEME SKORU — LINEAR REGRESSION")
    print("=" * 49)
    print(f"Veri boyutu: {X.shape}")
    print(f"Basit BMI modeli: {sonuc['basit_metrikler']}")
    print(f"Çoklu model: {sonuc['coklu_metrikler']}")
    print(
        f"5-fold CV R²: {sonuc['cv_r2_skorlari'].mean():.4f} "
        f"± {sonuc['cv_r2_skorlari'].std():.4f}"
    )
    print("\nEn güçlü katsayılar:\n", sonuc["katsayilar"].head().to_string(index=False))
    print("\nNot: Bu çıktı tıbbi teşhis amacı taşımaz.")
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
