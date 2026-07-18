"""Finansal işlemlerde ampirik copula tabanlı COPOD uygulaması."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
OZELLIKLER = ["amount", "transactions_24h", "risk_score", "geo_distance_km", "device_age_days", "hour_deviation"]


class EmpiricalCOPOD:
    """ECDF sol/sağ kuyruklarını birleştiren açıklanabilir COPOD modeli."""

    def __init__(self, contamination: float = 0.05):
        if not 0 < contamination < 0.5:
            raise ValueError("contamination 0 ile 0.5 arasında olmalıdır")
        self.contamination = contamination
        self.sorted_train_: np.ndarray | None = None
        self.skewness_: np.ndarray | None = None
        self.threshold_: float | None = None

    def _katkilar(self, X: np.ndarray) -> np.ndarray:
        if self.sorted_train_ is None or self.skewness_ is None:
            raise RuntimeError("Model önce fit edilmelidir")
        X = np.asarray(X, dtype=float)
        n, d = self.sorted_train_.shape
        katkilar = np.zeros_like(X, dtype=float)
        for j in range(d):
            sirali = self.sorted_train_[:, j]
            sol_sira = np.searchsorted(sirali, X[:, j], side="right")
            sag_sira = n - np.searchsorted(sirali, X[:, j], side="left")
            sol_olasilik = (sol_sira + 1) / (n + 2)
            sag_olasilik = (sag_sira + 1) / (n + 2)
            sol_kuyruk = -np.log(sol_olasilik)
            sag_kuyruk = -np.log(sag_olasilik)
            iki_yonlu = np.maximum(sol_kuyruk, sag_kuyruk)
            baskin_kuyruk = sag_kuyruk if self.skewness_[j] >= 0 else sol_kuyruk
            katkilar[:, j] = 0.5 * (iki_yonlu + baskin_kuyruk)
        return katkilar

    def fit(self, X: np.ndarray) -> "EmpiricalCOPOD":
        X = np.asarray(X, dtype=float)
        self.sorted_train_ = np.sort(X, axis=0)
        self.skewness_ = skew(X, axis=0, bias=False, nan_policy="omit")
        egitim_skoru = self.decision_function(X)
        self.threshold_ = float(np.quantile(egitim_skoru, 1 - self.contamination))
        return self

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        return self._katkilar(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._katkilar(X).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Model önce fit edilmelidir")
        return (self.decision_function(X) > self.threshold_).astype(int)


def normal_islemler(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    musteri_aktivitesi = rng.lognormal(0.0, 0.55, adet)
    tutar = rng.lognormal(4.4 + 0.20 * np.log1p(musteri_aktivitesi), 0.55, adet)
    islem_sayisi = rng.poisson(2.5 * musteri_aktivitesi, adet) + 1
    risk = np.clip(12 + 0.035 * tutar + 1.5 * islem_sayisi + rng.normal(0, 5, adet), 0, 100)
    mesafe = rng.gamma(1.8, 18, adet)
    cihaz_yasi = rng.gamma(3.5, 95, adet) + 15
    saat_sapma = np.abs(rng.normal(0, 2.0, adet))
    return pd.DataFrame(np.column_stack([tutar, islem_sayisi, risk, mesafe, cihaz_yasi, saat_sapma]), columns=OZELLIKLER)


def anomali_islemler(adet: int, rng: np.random.Generator) -> pd.DataFrame:
    turler = np.resize(np.array(["Yuksek_Tutar", "Cografi_Uyumsuzluk", "Hizli_Mikro_Islem", "Hesap_Ele_Gecirme"]), adet)
    rng.shuffle(turler)
    veri = normal_islemler(adet, rng)
    for i, tur in enumerate(turler):
        if tur == "Yuksek_Tutar":
            veri.loc[i, "amount"] += rng.uniform(1800, 4500)
            veri.loc[i, "risk_score"] = rng.uniform(75, 99)
        elif tur == "Cografi_Uyumsuzluk":
            veri.loc[i, "geo_distance_km"] += rng.uniform(600, 2200)
            veri.loc[i, "device_age_days"] = rng.uniform(1, 20)
        elif tur == "Hizli_Mikro_Islem":
            veri.loc[i, "transactions_24h"] += rng.integers(35, 90)
            veri.loc[i, "amount"] = rng.uniform(2, 25)
            veri.loc[i, "risk_score"] = rng.uniform(65, 95)
        else:
            veri.loc[i, "device_age_days"] = rng.uniform(0.1, 4)
            veri.loc[i, "hour_deviation"] += rng.uniform(7, 12)
            veri.loc[i, "geo_distance_km"] += rng.uniform(250, 900)
            veri.loc[i, "risk_score"] = rng.uniform(80, 100)
    veri["anomaly_type"] = turler
    return veri


def veriyi_uret(egitim_adet: int = 2200, test_normal: int = 900, test_anomali: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    egitim = normal_islemler(egitim_adet, rng)
    egitim["is_anomaly"] = 0
    egitim["anomaly_type"] = "Normal"
    normal = normal_islemler(test_normal, rng)
    normal["is_anomaly"] = 0
    normal["anomaly_type"] = "Normal"
    anomali = anomali_islemler(test_anomali, rng)
    anomali["is_anomaly"] = 1
    test = pd.concat([normal, anomali], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    egitim.insert(0, "transaction_id", [f"TRN{i:06d}" for i in range(1, len(egitim) + 1)])
    test.insert(0, "transaction_id", [f"TST{i:06d}" for i in range(1, len(test) + 1)])
    return egitim.round(4), test.round(4)


def metrikler(y: pd.Series, tahmin: np.ndarray, skor: np.ndarray) -> dict:
    return {
        "precision": precision_score(y, tahmin, zero_division=0),
        "recall": recall_score(y, tahmin, zero_division=0),
        "f1": f1_score(y, tahmin, zero_division=0),
        "roc_auc": roc_auc_score(y, skor),
        "detected": int(tahmin.sum()),
    }


def modelleri_karsilastir(egitim: pd.DataFrame, test: pd.DataFrame, contamination: float = 0.05) -> dict:
    X_train = egitim[OZELLIKLER].to_numpy()
    X_test = test[OZELLIKLER].to_numpy()
    y = test["is_anomaly"]
    ciktilar = {}
    satirlar = []

    baslangic = perf_counter()
    copod = EmpiricalCOPOD(contamination=contamination).fit(X_train)
    fit_suresi = perf_counter() - baslangic
    copod_skor = copod.decision_function(X_test)
    copod_tahmin = copod.predict(X_test)
    ciktilar["COPOD"] = {"model": copod, "score": copod_skor, "prediction": copod_tahmin}
    satirlar.append({"model": "COPOD", **metrikler(y, copod_tahmin, copod_skor), "fit_seconds": fit_suresi})

    baslangic = perf_counter()
    isolation = IsolationForest(n_estimators=300, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train)
    fit_suresi = perf_counter() - baslangic
    isolation_skor = -isolation.decision_function(X_test)
    isolation_tahmin = (isolation.predict(X_test) == -1).astype(int)
    ciktilar["IsolationForest"] = {"model": isolation, "score": isolation_skor, "prediction": isolation_tahmin}
    satirlar.append({"model": "IsolationForest", **metrikler(y, isolation_tahmin, isolation_skor), "fit_seconds": fit_suresi})

    scaler = StandardScaler().fit(X_train)
    baslangic = perf_counter()
    ocsvm = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale").fit(scaler.transform(X_train))
    fit_suresi = perf_counter() - baslangic
    ocsvm_skor = -ocsvm.decision_function(scaler.transform(X_test))
    ocsvm_tahmin = (ocsvm.predict(scaler.transform(X_test)) == -1).astype(int)
    ciktilar["OneClassSVM"] = {"model": ocsvm, "score": ocsvm_skor, "prediction": ocsvm_tahmin}
    satirlar.append({"model": "OneClassSVM", **metrikler(y, ocsvm_tahmin, ocsvm_skor), "fit_seconds": fit_suresi})

    return {"models": ciktilar, "metrics": pd.DataFrame(satirlar), "scaler": scaler}


def katkilar_raporu(test: pd.DataFrame, copod: EmpiricalCOPOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    katkilar = pd.DataFrame(copod.feature_contributions(test[OZELLIKLER].to_numpy()), columns=OZELLIKLER)
    katkilar.insert(0, "transaction_id", test["transaction_id"])
    katkilar["anomaly_type"] = test["anomaly_type"].to_numpy()
    tur_ozeti = katkilar[katkilar["anomaly_type"] != "Normal"].groupby("anomaly_type")[OZELLIKLER].mean()
    return katkilar, tur_ozeti


def gorselleri_uret(egitim: pd.DataFrame, test: pd.DataFrame, sonuc: dict, tur_katkisi: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(egitim[OZELLIKLER].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Normal Finansal İşlemlerin Korelasyon Yapısı")
    plt.tight_layout()
    plt.savefig(FIGURES / "transaction_correlation.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    normal_ornek = egitim.sample(800, random_state=RANDOM_STATE)
    axes[0].scatter(normal_ornek["amount"], normal_ornek["risk_score"], alpha=0.45, s=18, label="Normal")
    anomali = test[test["is_anomaly"] == 1]
    axes[0].scatter(anomali["amount"], anomali["risk_score"], color="#E45756", alpha=0.75, s=25, label="Anomali")
    axes[0].set(xlabel="İşlem tutarı", ylabel="Risk skoru", title="Tutar–Risk Bağımlılığı")
    axes[0].legend()
    axes[1].scatter(normal_ornek["geo_distance_km"], normal_ornek["device_age_days"], alpha=0.45, s=18, label="Normal")
    axes[1].scatter(anomali["geo_distance_km"], anomali["device_age_days"], color="#E45756", alpha=0.75, s=25, label="Anomali")
    axes[1].set(xlabel="Coğrafi mesafe", ylabel="Cihaz yaşı", title="Mesafe–Cihaz Bağımlılığı")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "dependency_scatter.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    y = test["is_anomaly"]
    for ad, cikti in sonuc["models"].items():
        fpr, tpr, _ = roc_curve(y, cikti["score"])
        auc = roc_auc_score(y, cikti["score"])
        plt.plot(fpr, tpr, label=f"{ad} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Rastgele")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Anomali Modelleri ROC Karşılaştırması")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_comparison.png", dpi=140)
    plt.close()

    metrik_uzun = sonuc["metrics"].melt(id_vars="model", value_vars=["precision", "recall", "f1", "roc_auc"], var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrik_uzun, x="metric", y="score", hue="model")
    plt.ylim(0, 1.05)
    plt.title("COPOD, Isolation Forest ve One-Class SVM")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_metrics.png", dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(tur_katkisi, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("COPOD — Anomali Türüne Göre Özellik Katkıları")
    plt.xlabel("Özellik")
    plt.ylabel("Anomali türü")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_contributions.png", dpi=140)
    plt.close()

    copod_skor = pd.DataFrame({"score": sonuc["models"]["COPOD"]["score"], "class": y.map({0: "Normal", 1: "Anomali"})})
    plt.figure(figsize=(8, 5))
    sns.histplot(data=copod_skor, x="score", hue="class", bins=35, kde=True, element="step")
    plt.title("COPOD Skor Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "copod_score_distribution.png", dpi=140)
    plt.close()


def main() -> None:
    egitim, test = veriyi_uret()
    sonuc = modelleri_karsilastir(egitim, test)
    katkilar, tur_katkisi = katkilar_raporu(test, sonuc["models"]["COPOD"]["model"])
    gorselleri_uret(egitim, test, sonuc, tur_katkisi)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    egitim.to_csv(DATA / "normal_financial_transactions.csv", index=False)
    test.to_csv(DATA / "mixed_financial_transactions.csv", index=False)
    sonuc["metrics"].round(4).to_csv(RESULTS / "model_comparison.csv", index=False)
    katkilar.round(4).to_csv(RESULTS / "copod_feature_contributions.csv", index=False)
    tur_katkisi.round(4).to_csv(RESULTS / "anomaly_type_contributions.csv")
    tahminler = pd.DataFrame({"transaction_id": test["transaction_id"], "actual": test["is_anomaly"]})
    for ad, cikti in sonuc["models"].items():
        tahminler[f"{ad}_prediction"] = cikti["prediction"]
        tahminler[f"{ad}_score"] = cikti["score"]
    tahminler.to_csv(RESULTS / "predictions.csv", index=False)
    ozet = {
        "train_shape": list(egitim.shape),
        "test_shape": list(test.shape),
        "anomaly_rate": round(float(test["is_anomaly"].mean()), 4),
        "copod_threshold": round(float(sonuc["models"]["COPOD"]["model"].threshold_), 4),
        "best_model_by_auc": sonuc["metrics"].sort_values("roc_auc", ascending=False).iloc[0]["model"],
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(ozet, dosya, ensure_ascii=False, indent=2)

    print("FİNANSAL İŞLEMLER — COPOD ANOMALİ TESPİTİ")
    print("=" * 49)
    print(f"Eğitim: {egitim.shape} | Test: {test.shape} | Anomali oranı: %{test['is_anomaly'].mean() * 100:.2f}")
    print("\nNormal işlem özeti:\n", egitim[OZELLIKLER].describe().round(2).to_string())
    print("\nModel karşılaştırması:\n", sonuc["metrics"].round(4).to_string(index=False))
    print("\nAnomali türlerine göre ortalama COPOD katkıları:\n", tur_katkisi.round(3).to_string())
    print(f"\nCOPOD karar eşiği: {sonuc['models']['COPOD']['model'].threshold_:.4f}")
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
