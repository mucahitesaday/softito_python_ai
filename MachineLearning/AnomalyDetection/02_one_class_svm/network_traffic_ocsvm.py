"""Ağ trafiğinde One-Class SVM ile saldırı anomalisi tespiti."""

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
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
OZELLIKLER = ["duration_ms", "packets_per_second", "bytes_per_packet", "inbound_ratio", "unique_ports", "failed_connections"]


def normal_trafik(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    yogunluk = rng.lognormal(mean=0.0, sigma=0.42, size=adet)
    duration = rng.gamma(2.2, 55, adet) + 15
    packets = 32 * yogunluk + rng.normal(0, 5, adet)
    bytes_packet = 620 + 80 * yogunluk + rng.normal(0, 70, adet)
    inbound = rng.beta(5, 5, adet)
    ports = rng.poisson(2.2, adet) + 1
    failures = rng.poisson(0.35, adet)
    veri = pd.DataFrame(np.column_stack([duration, packets, bytes_packet, inbound, ports, failures]), columns=OZELLIKLER)
    veri[["duration_ms", "packets_per_second", "bytes_per_packet"]] = veri[["duration_ms", "packets_per_second", "bytes_per_packet"]].clip(lower=0.1)
    return veri


def saldiri_trafigi(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    turler = np.resize(np.array(["DDoS", "Port_Tarama", "Veri_Sizdirma", "Kimlik_Deneme"]), adet)
    rng.shuffle(turler)
    veri = normal_trafik(adet, rng)
    for i, tur in enumerate(turler):
        if tur == "DDoS":
            veri.loc[i, "packets_per_second"] += rng.uniform(130, 260)
            veri.loc[i, "bytes_per_packet"] = rng.uniform(80, 260)
            veri.loc[i, "duration_ms"] = rng.uniform(10, 70)
        elif tur == "Port_Tarama":
            veri.loc[i, "unique_ports"] += rng.integers(18, 55)
            veri.loc[i, "failed_connections"] += rng.integers(8, 25)
            veri.loc[i, "bytes_per_packet"] = rng.uniform(60, 250)
        elif tur == "Veri_Sizdirma":
            veri.loc[i, "bytes_per_packet"] += rng.uniform(900, 1800)
            veri.loc[i, "inbound_ratio"] = rng.uniform(0.01, 0.12)
            veri.loc[i, "duration_ms"] += rng.uniform(250, 700)
        else:
            veri.loc[i, "failed_connections"] += rng.integers(15, 40)
            veri.loc[i, "unique_ports"] += rng.integers(2, 10)
    veri["traffic_type"] = turler
    return veri


def veri_setlerini_uret(egitim_adet: int = 1000, dogrulama_normal: int = 240, dogrulama_anomali: int = 60, test_normal: int = 480, test_anomali: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    egitim = normal_trafik(egitim_adet, rng)
    egitim["is_anomaly"] = 0
    egitim["traffic_type"] = "Normal"

    def karisik(normal_adet: int, anomali_adet: int, seed: int) -> pd.DataFrame:
        yerel = np.random.default_rng(seed)
        normal = normal_trafik(normal_adet, yerel)
        normal["is_anomaly"] = 0
        normal["traffic_type"] = "Normal"
        anomali = saldiri_trafigi(anomali_adet, yerel)
        anomali["is_anomaly"] = 1
        return pd.concat([normal, anomali], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    dogrulama = karisik(dogrulama_normal, dogrulama_anomali, RANDOM_STATE + 1)
    test = karisik(test_normal, test_anomali, RANDOM_STATE + 2)
    for onek, veri in (("TR", egitim), ("VA", dogrulama), ("TE", test)):
        veri.insert(0, "flow_id", [f"{onek}{i:05d}" for i in range(1, len(veri) + 1)])
    return egitim.round(4), dogrulama.round(4), test.round(4)


def skorla(y: pd.Series, tahmin: np.ndarray, anomali_skoru: np.ndarray) -> dict:
    return {
        "precision": precision_score(y, tahmin, zero_division=0),
        "recall": recall_score(y, tahmin, zero_division=0),
        "f1": f1_score(y, tahmin, zero_division=0),
        "roc_auc": roc_auc_score(y, anomali_skoru),
        "detected": int(tahmin.sum()),
    }


def parametre_ara(egitim: pd.DataFrame, dogrulama: pd.DataFrame) -> tuple[pd.DataFrame, dict, StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(egitim[OZELLIKLER])
    X_val = scaler.transform(dogrulama[OZELLIKLER])
    satirlar = []
    for nu in (0.01, 0.03, 0.05, 0.08, 0.12):
        for gamma in ("scale", 0.03, 0.08, 0.20):
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(X_train)
            tahmin = (model.predict(X_val) == -1).astype(int)
            skor = -model.decision_function(X_val)
            satirlar.append({"nu": nu, "gamma": str(gamma), **skorla(dogrulama["is_anomaly"], tahmin, skor)})
    sonuclar = pd.DataFrame(satirlar)
    en_iyi = sonuclar.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]
    parametre = {"nu": float(en_iyi["nu"]), "gamma": "scale" if en_iyi["gamma"] == "scale" else float(en_iyi["gamma"])}
    return sonuclar, parametre, scaler


def modeli_egit(egitim: pd.DataFrame, dogrulama: pd.DataFrame, test: pd.DataFrame) -> dict:
    arama, parametre, scaler = parametre_ara(egitim, dogrulama)
    X_train = scaler.transform(egitim[OZELLIKLER])
    X_test = scaler.transform(test[OZELLIKLER])
    model = OneClassSVM(kernel="rbf", **parametre)
    model.fit(X_train)
    tahmin = (model.predict(X_test) == -1).astype(int)
    skor = -model.decision_function(X_test)
    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "prediction": tahmin,
        "score": skor,
        "metrics": skorla(test["is_anomaly"], tahmin, skor),
        "parameter_results": arama,
        "best_parameters": parametre,
    }


def saldiri_basarisi(test: pd.DataFrame, tahmin: np.ndarray) -> pd.DataFrame:
    veri = test[["traffic_type", "is_anomaly"]].copy()
    veri["prediction"] = tahmin
    return veri[veri["is_anomaly"] == 1].groupby("traffic_type").agg(sample_count=("prediction", "size"), detected=("prediction", "sum"), recall=("prediction", "mean")).reset_index()


def gorselleri_uret(egitim: pd.DataFrame, test: pd.DataFrame, sonuc: dict, tur_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(egitim[OZELLIKLER].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Normal Ağ Trafiği Korelasyonları")
    plt.tight_layout()
    plt.savefig(FIGURES / "normal_traffic_correlation.png", dpi=140)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(sonuc["X_train"])
    koordinat = pca.transform(sonuc["X_test"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(koordinat[:, 0], koordinat[:, 1], c=test["is_anomaly"], cmap="coolwarm", s=23, alpha=0.72)
    axes[0].set_title("Gerçek Ağ Trafiği Sınıfları")
    axes[1].scatter(koordinat[:, 0], koordinat[:, 1], c=sonuc["prediction"], cmap="coolwarm", s=23, alpha=0.72)
    axes[1].set_title("One-Class SVM Tespiti")
    for ax in axes:
        ax.set(xlabel="PC1", ylabel="PC2")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_actual_vs_detected.png", dpi=140)
    plt.close()

    en_iyi_gamma = str(sonuc["best_parameters"]["gamma"])
    nu_df = sonuc["parameter_results"].query("gamma == @en_iyi_gamma")
    plt.figure(figsize=(8, 5))
    for metrik in ("precision", "recall", "f1"):
        plt.plot(nu_df["nu"], nu_df[metrik], "o-", label=metrik)
    plt.ylim(0, 1.05)
    plt.xlabel("nu")
    plt.ylabel("Skor")
    plt.title(f"One-Class SVM nu Etkisi — gamma={en_iyi_gamma}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "nu_parameter_effect.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=tur_df, x="traffic_type", y="recall", color="#E45756")
    plt.ylim(0, 1.05)
    plt.xlabel("Saldırı türü")
    plt.ylabel("Yakalama oranı")
    plt.title("Saldırı Türlerine Göre One-Class SVM Başarısı")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "attack_type_recall.png", dpi=140)
    plt.close()

    skor_df = pd.DataFrame({"score": sonuc["score"], "class": test["is_anomaly"].map({0: "Normal", 1: "Saldırı"})})
    plt.figure(figsize=(8, 5))
    sns.histplot(data=skor_df, x="score", hue="class", bins=35, kde=True, element="step")
    plt.title("One-Class SVM Anomali Skorları")
    plt.tight_layout()
    plt.savefig(FIGURES / "anomaly_score_distribution.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(test["is_anomaly"], sonuc["prediction"], display_labels=["Normal", "Saldırı"], cmap="Purples", colorbar=False)
    plt.title("One-Class SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()


def main() -> None:
    egitim, dogrulama, test = veri_setlerini_uret()
    sonuc = modeli_egit(egitim, dogrulama, test)
    tur_df = saldiri_basarisi(test, sonuc["prediction"])
    gorselleri_uret(egitim, test, sonuc, tur_df)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    egitim.to_csv(DATA / "normal_network_training.csv", index=False)
    dogrulama.to_csv(DATA / "network_validation.csv", index=False)
    test.to_csv(DATA / "network_test.csv", index=False)
    sonuc["parameter_results"].round(4).to_csv(RESULTS / "nu_gamma_search.csv", index=False)
    tur_df.round(4).to_csv(RESULTS / "attack_type_recall.csv", index=False)
    pd.DataFrame({"flow_id": test["flow_id"], "actual": test["is_anomaly"], "prediction": sonuc["prediction"], "anomaly_score": sonuc["score"]}).to_csv(RESULTS / "predictions.csv", index=False)
    ozet = {
        "train_shape": list(egitim.shape),
        "validation_shape": list(dogrulama.shape),
        "test_shape": list(test.shape),
        "best_parameters": sonuc["best_parameters"],
        "test_metrics": {k: round(float(v), 4) for k, v in sonuc["metrics"].items()},
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(ozet, dosya, ensure_ascii=False, indent=2)

    print("AĞ TRAFİĞİ — ONE-CLASS SVM ANOMALİ TESPİTİ")
    print("=" * 52)
    print(f"Eğitim: {egitim.shape} | Doğrulama: {dogrulama.shape} | Test: {test.shape}")
    print(f"Test saldırı oranı: %{test['is_anomaly'].mean() * 100:.2f}")
    print("\nNormal trafik özeti:\n", egitim[OZELLIKLER].describe().round(2).to_string())
    print(f"\nEn iyi parametreler: {sonuc['best_parameters']}")
    print("Test metrikleri:", {k: round(float(v), 4) for k, v in sonuc["metrics"].items()})
    print("\nSaldırı türü başarıları:\n", tur_df.round(4).to_string(index=False))
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
