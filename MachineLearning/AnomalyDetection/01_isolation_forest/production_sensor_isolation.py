"""Üretim hattı sensörlerinde Isolation Forest ile anomali tespiti."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
OZELLIKLER = ["temperature_c", "vibration_mm_s", "pressure_bar", "rotation_rpm", "power_kw"]


def normal_sensor_verisi(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    yuk = rng.normal(0, 1, adet)
    sicaklik = 68 + 5.2 * yuk + rng.normal(0, 1.8, adet)
    titresim = 2.1 + 0.30 * yuk + rng.normal(0, 0.18, adet)
    basinc = 7.4 + 0.35 * yuk + rng.normal(0, 0.16, adet)
    devir = 1480 + 105 * yuk + rng.normal(0, 35, adet)
    guc = 42 + 6.5 * yuk + rng.normal(0, 2.2, adet)
    return pd.DataFrame(np.column_stack([sicaklik, titresim, basinc, devir, guc]), columns=OZELLIKLER)


def anomali_sensor_verisi(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    turler = np.resize(np.array(["Asiri_Isinma", "Rulman_Arizasi", "Basinc_Kacagi", "Kombine_Ariza"]), adet)
    rng.shuffle(turler)
    normal = normal_sensor_verisi(adet, rng)
    for i, tur in enumerate(turler):
        if tur == "Asiri_Isinma":
            normal.loc[i, "temperature_c"] += rng.uniform(18, 32)
            normal.loc[i, "power_kw"] += rng.uniform(5, 14)
        elif tur == "Rulman_Arizasi":
            normal.loc[i, "vibration_mm_s"] += rng.uniform(2.2, 4.5)
            normal.loc[i, "temperature_c"] += rng.uniform(5, 13)
        elif tur == "Basinc_Kacagi":
            normal.loc[i, "pressure_bar"] -= rng.uniform(2.2, 3.8)
            normal.loc[i, "power_kw"] += rng.uniform(3, 10)
        else:
            normal.loc[i, "temperature_c"] += rng.uniform(12, 24)
            normal.loc[i, "vibration_mm_s"] += rng.uniform(1.5, 3.2)
            normal.loc[i, "rotation_rpm"] -= rng.uniform(250, 480)
    normal["anomaly_type"] = turler
    return normal


def veriyi_uret(egitim_adet: int = 1200, test_normal: int = 500, test_anomali: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    egitim = normal_sensor_verisi(egitim_adet, rng)
    egitim["is_anomaly"] = 0
    egitim["anomaly_type"] = "Normal"

    normal_test = normal_sensor_verisi(test_normal, rng)
    normal_test["is_anomaly"] = 0
    normal_test["anomaly_type"] = "Normal"
    anomali_test = anomali_sensor_verisi(test_anomali, rng)
    anomali_test["is_anomaly"] = 1
    test = pd.concat([normal_test, anomali_test], ignore_index=True)
    test = test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    egitim.insert(0, "record_id", [f"TR{i:05d}" for i in range(1, len(egitim) + 1)])
    test.insert(0, "record_id", [f"TE{i:05d}" for i in range(1, len(test) + 1)])
    return egitim.round(4), test.round(4)


def veriyi_incele(egitim: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "train_shape": list(egitim.shape),
        "test_shape": list(test.shape),
        "train_missing": int(egitim.isna().sum().sum()),
        "test_missing": int(test.isna().sum().sum()),
        "train_duplicates": int(egitim.duplicated().sum()),
        "test_anomaly_rate": round(float(test["is_anomaly"].mean()), 4),
        "anomaly_type_counts": test["anomaly_type"].value_counts().to_dict(),
    }


def metrikleri_hesapla(y: pd.Series, tahmin: np.ndarray, skor: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, tahmin),
        "precision": precision_score(y, tahmin, zero_division=0),
        "recall": recall_score(y, tahmin, zero_division=0),
        "f1": f1_score(y, tahmin, zero_division=0),
        "roc_auc": roc_auc_score(y, skor),
        "detected_anomalies": int(tahmin.sum()),
    }


def modeli_egit(egitim: pd.DataFrame, test: pd.DataFrame, contamination: float = 0.08) -> dict:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(egitim[OZELLIKLER])
    X_test = scaler.transform(test[OZELLIKLER])
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    ham_tahmin = model.predict(X_test)
    tahmin = (ham_tahmin == -1).astype(int)
    skor = -model.decision_function(X_test)
    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "prediction": tahmin,
        "score": skor,
        "metrics": metrikleri_hesapla(test["is_anomaly"], tahmin, skor),
    }


def contamination_analizi(egitim: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for oran in (0.01, 0.03, 0.05, 0.08, 0.12):
        sonuc = modeli_egit(egitim, test, oran)
        satirlar.append({"contamination": oran, **sonuc["metrics"]})
    return pd.DataFrame(satirlar)


def ariza_turu_basarisi(test: pd.DataFrame, tahmin: np.ndarray) -> pd.DataFrame:
    gecici = test[["anomaly_type", "is_anomaly"]].copy()
    gecici["prediction"] = tahmin
    anomaliler = gecici[gecici["is_anomaly"] == 1]
    return anomaliler.groupby("anomaly_type").agg(sample_count=("prediction", "size"), detected=("prediction", "sum"), recall=("prediction", "mean")).reset_index()


def gorselleri_uret(egitim: pd.DataFrame, test: pd.DataFrame, sonuc: dict, contamination_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    egitim[OZELLIKLER].hist(bins=30, figsize=(13, 9), color="#4C78A8", edgecolor="white")
    plt.suptitle("Normal Eğitim Sensörlerinin Dağılımları")
    plt.tight_layout()
    plt.savefig(FIGURES / "normal_sensor_distributions.png", dpi=140)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    birlesik = np.vstack([sonuc["X_train"], sonuc["X_test"]])
    pca.fit(birlesik)
    test_pca = pca.transform(sonuc["X_test"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(test_pca[:, 0], test_pca[:, 1], c=test["is_anomaly"], cmap="coolwarm", s=24, alpha=0.75)
    axes[0].set_title("Gerçek Durum")
    axes[1].scatter(test_pca[:, 0], test_pca[:, 1], c=sonuc["prediction"], cmap="coolwarm", s=24, alpha=0.75)
    axes[1].set_title("Isolation Forest Tahmini")
    for ax in axes:
        ax.set(xlabel="PC1", ylabel="PC2")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_actual_vs_detected.png", dpi=140)
    plt.close()

    skor_df = pd.DataFrame({"anomaly_score": sonuc["score"], "class": test["is_anomaly"].map({0: "Normal", 1: "Anomali"})})
    plt.figure(figsize=(8, 5))
    sns.histplot(data=skor_df, x="anomaly_score", hue="class", bins=35, kde=True, element="step")
    plt.title("Isolation Forest Anomali Skoru Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "anomaly_score_distribution.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(contamination_df["contamination"], contamination_df["precision"], "o-", label="Precision")
    axes[0].plot(contamination_df["contamination"], contamination_df["recall"], "o-", label="Recall")
    axes[0].plot(contamination_df["contamination"], contamination_df["f1"], "o-", label="F1")
    axes[0].set(title="Contamination Etkisi", xlabel="Contamination", ylabel="Skor", ylim=(0, 1.05))
    axes[0].legend()
    axes[1].plot(contamination_df["contamination"], contamination_df["detected_anomalies"], "o-", color="#E45756")
    axes[1].axhline(test["is_anomaly"].sum(), color="black", linestyle="--", label="Gerçek anomali")
    axes[1].set(title="Tespit Edilen Anomali Sayısı", xlabel="Contamination", ylabel="Adet")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "contamination_effect.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(test["is_anomaly"], sonuc["prediction"], display_labels=["Normal", "Anomali"], cmap="Blues", colorbar=False)
    plt.title("Isolation Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()


def main() -> None:
    egitim, test = veriyi_uret()
    inceleme = veriyi_incele(egitim, test)
    sonuc = modeli_egit(egitim, test)
    contamination_df = contamination_analizi(egitim, test)
    tur_basarisi = ariza_turu_basarisi(test, sonuc["prediction"])
    gorselleri_uret(egitim, test, sonuc, contamination_df)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    egitim.to_csv(DATA / "normal_training_sensors.csv", index=False)
    test.to_csv(DATA / "mixed_test_sensors.csv", index=False)
    contamination_df.round(4).to_csv(RESULTS / "contamination_analysis.csv", index=False)
    tur_basarisi.round(4).to_csv(RESULTS / "anomaly_type_recall.csv", index=False)
    pd.DataFrame({"record_id": test["record_id"], "actual": test["is_anomaly"], "prediction": sonuc["prediction"], "anomaly_score": sonuc["score"]}).to_csv(RESULTS / "predictions.csv", index=False)
    ozet = {"data_review": inceleme, "test_metrics": {k: round(float(v), 4) for k, v in sonuc["metrics"].items()}}
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(ozet, dosya, ensure_ascii=False, indent=2)

    print("ÜRETİM SENSÖRLERİ — ISOLATION FOREST")
    print("=" * 45)
    print(f"Eğitim boyutu: {egitim.shape} | Test boyutu: {test.shape}")
    print(f"Test anomali oranı: %{test['is_anomaly'].mean() * 100:.2f}")
    print("\nNormal eğitim verisi özeti:\n", egitim[OZELLIKLER].describe().round(2).to_string())
    print("\nTest metrikleri:", {k: round(float(v), 4) for k, v in sonuc["metrics"].items()})
    print("\nContamination karşılaştırması:\n", contamination_df.round(4).to_string(index=False))
    print("\nArıza türü yakalama oranları:\n", tur_basarisi.round(4).to_string(index=False))
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
