"""Restoran bahşiş miktarı için basit ve çoklu doğrusal regresyon."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

KLASOR = Path(__file__).resolve().parent
VERI_DOSYASI = KLASOR / "data" / "tips.csv"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42


def veriyi_yukle() -> pd.DataFrame:
    df = pd.read_csv(VERI_DOSYASI)
    if df.isna().sum().sum() > 0:
        raise ValueError("Tips veri setinde beklenmeyen eksik değer bulundu.")
    return df


def metrikleri_hesapla(gercek: pd.Series, tahmin: np.ndarray) -> dict[str, float]:
    return {
        "r2": round(float(r2_score(gercek, tahmin)), 4),
        "mae": round(float(mean_absolute_error(gercek, tahmin)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(gercek, tahmin))), 4),
    }


def modelleri_egit(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=0.25, random_state=RANDOM_STATE)

    basit = LinearRegression()
    basit.fit(train[["total_bill"]], train["tip"])
    basit_tahmin = basit.predict(test[["total_bill"]])

    sayisal = ["total_bill", "size"]
    kategorik = ["sex", "smoker", "day", "time"]
    donusturucu = ColumnTransformer(
        [("kategori", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), kategorik)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    coklu = Pipeline(
        [("donusturucu", donusturucu), ("model", LinearRegression())]
    )
    coklu.fit(train[sayisal + kategorik], train["tip"])
    coklu_tahmin = coklu.predict(test[sayisal + kategorik])

    ozellik_adlari = coklu.named_steps["donusturucu"].get_feature_names_out()
    katsayilar = pd.DataFrame(
        {
            "feature": ozellik_adlari,
            "coefficient": coklu.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", key=abs, ascending=False)

    return {
        "train": train,
        "test": test,
        "basit_model": basit,
        "coklu_model": coklu,
        "basit_tahmin": basit_tahmin,
        "coklu_tahmin": coklu_tahmin,
        "basit_metrikler": metrikleri_hesapla(test["tip"], basit_tahmin),
        "coklu_metrikler": metrikleri_hesapla(test["tip"], coklu_tahmin),
        "katsayilar": katsayilar,
    }


def gorselleri_uret(df: pd.DataFrame, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(df[["total_bill", "tip", "size"]].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Sayısal Değişkenler Korelasyonu")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_matrix.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", alpha=0.7)
    x_cizgi = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100)
    y_cizgi = sonuc["basit_model"].predict(pd.DataFrame({"total_bill": x_cizgi}))
    plt.plot(x_cizgi, y_cizgi, color="crimson", linewidth=2, label="Regresyon doğrusu")
    plt.title("Basit Regresyon: Hesap Tutarı → Bahşiş")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "simple_regression.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sonuc["katsayilar"], x="coefficient", y="feature", color="#4C78A8")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Çoklu Regresyon Katsayıları")
    plt.tight_layout()
    plt.savefig(FIGURES / "coefficients.png", dpi=140, bbox_inches="tight")
    plt.close()

    gercek = sonuc["test"]["tip"]
    tahmin = sonuc["coklu_tahmin"]
    plt.figure(figsize=(6, 6))
    plt.scatter(gercek, tahmin, alpha=0.7, color="#2A9D8F")
    sinirlar = [min(gercek.min(), tahmin.min()), max(gercek.max(), tahmin.max())]
    plt.plot(sinirlar, sinirlar, "r--", label="Kusursuz tahmin")
    plt.xlabel("Gerçek bahşiş")
    plt.ylabel("Tahmin edilen bahşiş")
    plt.title("Gerçek ve Tahmin Edilen Bahşiş")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=140, bbox_inches="tight")
    plt.close()

    artiklar = gercek.to_numpy() - tahmin
    plt.figure(figsize=(8, 5))
    plt.scatter(tahmin, artiklar, alpha=0.7, color="#E76F51")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Tahmin")
    plt.ylabel("Artık (gerçek - tahmin)")
    plt.title("Artık Analizi")
    plt.tight_layout()
    plt.savefig(FIGURES / "residuals.png", dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = veriyi_yukle()
    sonuc = modelleri_egit(df)
    gorselleri_uret(df, sonuc)

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrikler = {
        "veri_satiri": len(df),
        "basit_regresyon": sonuc["basit_metrikler"],
        "coklu_regresyon": sonuc["coklu_metrikler"],
        "basit_egim": round(float(sonuc["basit_model"].coef_[0]), 4),
        "basit_kesisim": round(float(sonuc["basit_model"].intercept_), 4),
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as dosya:
        json.dump(metrikler, dosya, ensure_ascii=False, indent=2)
    sonuc["katsayilar"].to_csv(RESULTS / "coefficients.csv", index=False)

    print("RESTORAN BAHŞİŞ TAHMİNİ — LINEAR REGRESSION")
    print("=" * 50)
    print(f"Veri boyutu: {df.shape}")
    print(f"Basit model: {sonuc['basit_metrikler']}")
    print(f"Çoklu model: {sonuc['coklu_metrikler']}")
    print(f"Basit model denklemi: tip = {sonuc['basit_model'].intercept_:.3f} + {sonuc['basit_model'].coef_[0]:.3f} × total_bill")
    print("\nEn güçlü katsayılar:\n", sonuc["katsayilar"].head().to_string(index=False))
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
