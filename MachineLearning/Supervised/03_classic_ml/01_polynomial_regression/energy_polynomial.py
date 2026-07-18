"""Sıcaklık ve enerji talebi üzerinde ayrıntılı Polynomial Regression analizi."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
DERECELER = (1, 2, 3, 5, 8)


def veriyi_uret(n: int = 360) -> pd.DataFrame:
    """Isıtma ve soğutma ihtiyacını taklit eden tekrarlanabilir veri üretir."""
    rng = np.random.default_rng(RANDOM_STATE)
    sicaklik = rng.uniform(-5, 40, n)
    nem = np.clip(65 - 0.7 * sicaklik + rng.normal(0, 8, n), 20, 95)
    enerji = (
        92
        + 0.42 * (sicaklik - 19) ** 2
        + 0.003 * (sicaklik - 19) ** 3
        + 0.08 * nem
        + rng.normal(0, 8, n)
    )
    tarih = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": tarih,
            "temperature_c": sicaklik.round(2),
            "humidity_percent": nem.round(2),
            "energy_demand_mwh": enerji.round(2),
        }
    )


def veriyi_incele(df: pd.DataFrame) -> dict:
    """Boyut, eksik değer, tekrar ve temel istatistikleri hazırlar."""
    sayisal = df.select_dtypes(include="number")
    return {
        "shape": df.shape,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "temperature_min": float(df["temperature_c"].min()),
        "temperature_max": float(df["temperature_c"].max()),
        "correlation": sayisal.corr().round(3),
        "description": sayisal.describe().round(2),
    }


def model_olustur(derece: int) -> Pipeline:
    """Polinom dönüşümü, ölçekleme ve doğrusal modeli tek Pipeline'da toplar."""
    return Pipeline(
        [
            ("polinom", PolynomialFeatures(degree=derece, include_bias=False)),
            ("olcekleme", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def metrikleri_hesapla(gercek: pd.Series, tahmin: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(gercek, tahmin)),
        "rmse": float(np.sqrt(mean_squared_error(gercek, tahmin))),
        "r2": float(r2_score(gercek, tahmin)),
    }


def modelleri_karsilastir(df: pd.DataFrame) -> dict:
    X = df[["temperature_c"]]
    y = df["energy_demand_mwh"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    satirlar: list[dict] = []
    modeller: dict[int, Pipeline] = {}
    for derece in DERECELER:
        model = model_olustur(derece)
        model.fit(X_train, y_train)
        modeller[derece] = model

        for bolum, X_bolum, y_bolum in (
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ):
            tahmin = model.predict(X_bolum)
            satirlar.append(
                {"degree": derece, "split": bolum, **metrikleri_hesapla(y_bolum, tahmin)}
            )

    metrikler = pd.DataFrame(satirlar)
    test_metrikleri = metrikler.query("split == 'test'")
    en_iyi_derece = int(test_metrikleri.loc[test_metrikleri["rmse"].idxmin(), "degree"])
    en_iyi_tahmin = modeller[en_iyi_derece].predict(X_test)
    return {
        "models": modeller,
        "metrics": metrikler,
        "best_degree": en_iyi_derece,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_prediction": en_iyi_tahmin,
    }


def gorselleri_uret(df: pd.DataFrame, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["temperature_c"], kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Sıcaklık Dağılımı")
    sns.scatterplot(data=df, x="temperature_c", y="energy_demand_mwh", ax=axes[1], alpha=0.65)
    axes[1].set_title("Sıcaklık–Enerji İlişkisi")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_overview.png", dpi=140)
    plt.close(fig)

    x_cizgi = np.linspace(df["temperature_c"].min(), df["temperature_c"].max(), 400)
    model = sonuc["models"][sonuc["best_degree"]]
    cizgi_tahmini = model.predict(pd.DataFrame({"temperature_c": x_cizgi}))
    plt.figure(figsize=(8, 5))
    plt.scatter(df["temperature_c"], df["energy_demand_mwh"], s=18, alpha=0.35)
    plt.plot(x_cizgi, cizgi_tahmini, color="crimson", linewidth=2.5)
    plt.xlabel("Sıcaklık (°C)")
    plt.ylabel("Enerji talebi (MWh)")
    plt.title(f"En İyi Polinom Modeli — Derece {sonuc['best_degree']}")
    plt.tight_layout()
    plt.savefig(FIGURES / "best_polynomial_fit.png", dpi=140)
    plt.close()

    hata_tablosu = sonuc["metrics"].pivot(index="degree", columns="split", values="rmse")
    hata_tablosu.plot(marker="o", figsize=(7, 5))
    plt.ylabel("RMSE")
    plt.title("Model Karmaşıklığı ve Bias–Variance Dengesi")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "degree_comparison.png", dpi=140)
    plt.close()

    artiklar = sonuc["y_test"].to_numpy() - sonuc["best_prediction"]
    plt.figure(figsize=(7, 5))
    plt.scatter(sonuc["best_prediction"], artiklar, alpha=0.7, color="#F58518")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Tahmin edilen enerji")
    plt.ylabel("Artık (gerçek - tahmin)")
    plt.title("En İyi Modelin Artık Analizi")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_analysis.png", dpi=140)
    plt.close()


def main() -> None:
    df = veriyi_uret()
    inceleme = veriyi_incele(df)
    sonuc = modelleri_karsilastir(df)
    gorselleri_uret(df, sonuc)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "energy_demand.csv", index=False)
    sonuc["metrics"].round(4).to_csv(RESULTS / "degree_metrics.csv", index=False)
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(
            {
                "rows": len(df),
                "missing_values": inceleme["missing_values"],
                "duplicate_rows": inceleme["duplicate_rows"],
                "best_degree": sonuc["best_degree"],
            },
            dosya,
            ensure_ascii=False,
            indent=2,
        )

    print("POLYNOMIAL REGRESSION — ENERJİ TALEBİ")
    print("=" * 44)
    print(f"Veri boyutu: {inceleme['shape']}")
    print(f"Eksik değer: {inceleme['missing_values']} | Tekrarlı satır: {inceleme['duplicate_rows']}")
    print("\nSayısal değişken özeti:\n", inceleme["description"].to_string())
    print("\nKorelasyon matrisi:\n", inceleme["correlation"].to_string())
    print("\nDerece karşılaştırması:\n", sonuc["metrics"].round(4).to_string(index=False))
    print(f"\nEn iyi derece: {sonuc['best_degree']}")
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
