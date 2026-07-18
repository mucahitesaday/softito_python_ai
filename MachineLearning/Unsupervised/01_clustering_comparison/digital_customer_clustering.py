"""Dijital müşteri davranışlarında denetimsiz öğrenme karşılaştırması."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42

OZELLIKLER = [
    "monthly_visits",
    "session_minutes",
    "pages_per_session",
    "cart_additions",
    "purchase_rate",
    "average_order_value",
    "discount_usage_rate",
    "return_rate",
]

PROFILLER = {
    "Dusuk_Etkilesim": [4, 3.5, 4, 0.8, 0.08, 180, 0.25, 0.08],
    "Kampanya_Odakli": [18, 6.5, 10, 3.2, 0.24, 260, 0.82, 0.12],
    "Kararsiz_Sepet": [13, 9.0, 15, 6.0, 0.16, 340, 0.48, 0.20],
    "Sadik_Yuksek_Deger": [11, 12.5, 20, 4.8, 0.68, 720, 0.18, 0.04],
}


def veriyi_uret(musteri_sayisi: int = 900) -> pd.DataFrame:
    """Dört davranış profili ve az sayıda sıra dışı gözlem üretir."""
    rng = np.random.default_rng(RANDOM_STATE)
    aykiri_sayisi = max(12, round(musteri_sayisi * 0.03))
    normal_sayi = musteri_sayisi - aykiri_sayisi
    dagilim = [normal_sayi // 4] * 4
    for i in range(normal_sayi % 4):
        dagilim[i] += 1

    satirlar: list[dict] = []
    sapmalar = np.array([2.0, 1.4, 2.2, 0.8, 0.06, 75, 0.10, 0.035])
    for (profil, ortalamalar), adet in zip(PROFILLER.items(), dagilim):
        degerler = rng.normal(ortalamalar, sapmalar, size=(adet, len(OZELLIKLER)))
        for satir in degerler:
            kayit = dict(zip(OZELLIKLER, satir))
            kayit["known_profile"] = profil
            satirlar.append(kayit)

    alt = np.array([1, 1, 1, 0, 0.01, 60, 0.01, 0.0])
    ust = np.array([35, 20, 30, 10, 0.95, 1200, 0.98, 0.45])
    for satir in rng.uniform(alt, ust, size=(aykiri_sayisi, len(OZELLIKLER))):
        kayit = dict(zip(OZELLIKLER, satir))
        kayit["known_profile"] = "Siradisi"
        satirlar.append(kayit)

    veri = pd.DataFrame(satirlar).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    veri.insert(0, "customer_id", [f"DC{i:04d}" for i in range(1, len(veri) + 1)])
    pozitifler = ["monthly_visits", "session_minutes", "pages_per_session", "cart_additions", "average_order_value"]
    veri[pozitifler] = veri[pozitifler].clip(lower=0.1)
    oranlar = ["purchase_rate", "discount_usage_rate", "return_rate"]
    veri[oranlar] = veri[oranlar].clip(0, 1)

    # Gerçek bir veri temizleme adımını göstermek için sayısal alanlara az miktarda eksik değer eklenir.
    for sutun in OZELLIKLER:
        indeksler = rng.choice(veri.index, size=max(1, len(veri) // 100), replace=False)
        veri.loc[indeksler, sutun] = np.nan
    return veri.round(4)


def veriyi_incele(veri: pd.DataFrame) -> dict:
    return {
        "shape": list(veri.shape),
        "missing_values": int(veri.isna().sum().sum()),
        "duplicate_rows": int(veri.duplicated().sum()),
        "known_profile_counts": veri["known_profile"].value_counts().to_dict(),
    }


def on_isleme(veri: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    doldurulmus = pd.DataFrame(imputer.fit_transform(veri[OZELLIKLER]), columns=OZELLIKLER)
    scaler = StandardScaler()
    olcekli = scaler.fit_transform(doldurulmus)
    return doldurulmus, olcekli, imputer, scaler


def k_degerlerini_incele(X: np.ndarray) -> tuple[pd.DataFrame, int]:
    satirlar = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        etiketler = model.fit_predict(X)
        satirlar.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, etiketler),
                "davies_bouldin": davies_bouldin_score(X, etiketler),
            }
        )
    sonuclar = pd.DataFrame(satirlar)
    en_iyi_k = int(sonuclar.loc[sonuclar["silhouette"].idxmax(), "k"])
    return sonuclar, en_iyi_k


def dbscan_ara(X: np.ndarray) -> tuple[DBSCAN, np.ndarray, dict]:
    en_iyi = None
    for eps in (0.7, 0.9, 1.1, 1.3, 1.5, 1.8):
        for min_samples in (5, 8, 12):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            etiketler = model.fit_predict(X)
            maske = etiketler != -1
            kume_sayisi = len(set(etiketler[maske]))
            if kume_sayisi < 2 or maske.sum() < len(X) * 0.65:
                continue
            skor = silhouette_score(X[maske], etiketler[maske])
            aday = (skor, model, etiketler, eps, min_samples, kume_sayisi)
            if en_iyi is None or aday[0] > en_iyi[0]:
                en_iyi = aday
    if en_iyi is None:
        model = DBSCAN(eps=1.8, min_samples=5).fit(X)
        return model, model.labels_, {"eps": 1.8, "min_samples": 5}
    skor, model, etiketler, eps, min_samples, kume_sayisi = en_iyi
    return model, etiketler, {
        "eps": eps,
        "min_samples": min_samples,
        "cluster_count": kume_sayisi,
        "silhouette_without_noise": skor,
    }


def kume_metrikleri(X: np.ndarray, etiketler: np.ndarray) -> dict:
    maske = etiketler != -1
    temiz_etiketler = etiketler[maske]
    temiz_X = X[maske]
    kume_sayisi = len(set(temiz_etiketler))
    if kume_sayisi < 2:
        return {"cluster_count": kume_sayisi, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "cluster_count": kume_sayisi,
        "silhouette": silhouette_score(temiz_X, temiz_etiketler),
        "davies_bouldin": davies_bouldin_score(temiz_X, temiz_etiketler),
        "calinski_harabasz": calinski_harabasz_score(temiz_X, temiz_etiketler),
    }


def modelleri_karsilastir(X: np.ndarray, en_iyi_k: int, gercek_profiller: pd.Series) -> dict:
    modeller = {
        "KMeans": KMeans(n_clusters=en_iyi_k, n_init=20, random_state=RANDOM_STATE),
        "Agglomerative": AgglomerativeClustering(n_clusters=en_iyi_k, linkage="ward"),
        "GMM": GaussianMixture(n_components=en_iyi_k, covariance_type="full", random_state=RANDOM_STATE),
    }
    etiketler = {ad: model.fit_predict(X) for ad, model in modeller.items()}
    dbscan_model, dbscan_etiket, dbscan_parametreleri = dbscan_ara(X)
    modeller["DBSCAN"] = dbscan_model
    etiketler["DBSCAN"] = dbscan_etiket

    gercek_kod = pd.Categorical(gercek_profiller).codes
    metrik_satirlari = []
    for ad, model_etiketleri in etiketler.items():
        metrik = kume_metrikleri(X, model_etiketleri)
        metrik.update(
            {
                "model": ad,
                "noise_count": int((model_etiketleri == -1).sum()),
                "adjusted_rand": adjusted_rand_score(gercek_kod, model_etiketleri),
            }
        )
        metrik_satirlari.append(metrik)
    metrikler = pd.DataFrame(metrik_satirlari)[
        ["model", "cluster_count", "noise_count", "silhouette", "davies_bouldin", "calinski_harabasz", "adjusted_rand"]
    ]
    return {
        "models": modeller,
        "labels": etiketler,
        "metrics": metrikler,
        "dbscan_parameters": dbscan_parametreleri,
    }


def segment_adi(profil: pd.Series) -> str:
    if profil["ClusterSize"] < 50:
        return "Siradisi_Davranis"
    if profil["purchase_rate"] > 0.45 and profil["average_order_value"] > 500:
        return "Sadik_Yuksek_Deger"
    if profil["discount_usage_rate"] > 0.60:
        return "Kampanya_Odakli"
    if profil["cart_additions"] > 4 and profil["purchase_rate"] < 0.35:
        return "Kararsiz_Sepet"
    return "Dusuk_Etkilesim"


def gorselleri_uret(veri: pd.DataFrame, doldurulmus: pd.DataFrame, X: np.ndarray, k_sonuclari: pd.DataFrame, sonuc: dict) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    doldurulmus[["monthly_visits", "purchase_rate", "average_order_value", "discount_usage_rate"]].hist(
        bins=25, figsize=(12, 8), color="#4C78A8", edgecolor="white"
    )
    plt.suptitle("Dijital Müşteri Davranış Dağılımları")
    plt.tight_layout()
    plt.savefig(FIGURES / "data_distributions.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_sonuclari["k"], k_sonuclari["inertia"], "o-", color="#4C78A8")
    axes[0].set(title="Elbow Yöntemi", xlabel="Küme sayısı", ylabel="Inertia")
    axes[1].plot(k_sonuclari["k"], k_sonuclari["silhouette"], "o-", color="#F58518")
    axes[1].set(title="Silhouette Karşılaştırması", xlabel="Küme sayısı", ylabel="Silhouette")
    plt.tight_layout()
    plt.savefig(FIGURES / "optimal_k.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 6))
    dendrogram(linkage(X[:250], method="ward"), truncate_mode="level", p=6, no_labels=True)
    plt.title("Hiyerarşik Kümeleme Dendrogramı — İlk 250 Müşteri")
    plt.xlabel("Birleşen müşteri grupları")
    plt.ylabel("Ward uzaklığı")
    plt.tight_layout()
    plt.savefig(FIGURES / "dendrogram.png", dpi=140)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_koordinat = pca.fit_transform(X)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (ad, etiket) in zip(axes.flat, sonuc["labels"].items()):
        ax.scatter(pca_koordinat[:, 0], pca_koordinat[:, 1], c=etiket, cmap="tab10", s=18, alpha=0.72)
        ax.set_title(ad)
        ax.set_xlabel(f"PC1 (%{pca.explained_variance_ratio_[0] * 100:.1f})")
        ax.set_ylabel(f"PC2 (%{pca.explained_variance_ratio_[1] * 100:.1f})")
    plt.suptitle("PCA Üzerinde Kümeleme Algoritmaları")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_model_comparison.png", dpi=140)
    plt.close()

    tsne = TSNE(n_components=2, perplexity=35, learning_rate="auto", init="pca", max_iter=750, random_state=RANDOM_STATE)
    tsne_koordinat = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_koordinat[:, 0], tsne_koordinat[:, 1], c=sonuc["labels"]["KMeans"], cmap="viridis", s=20, alpha=0.75)
    plt.title("t-SNE ile K-Means Segmentleri")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(FIGURES / "tsne_kmeans.png", dpi=140)
    plt.close()

    gmm_olasilik = sonuc["models"]["GMM"].predict_proba(X).max(axis=1)
    plt.figure(figsize=(8, 5))
    sns.histplot(gmm_olasilik, bins=25, kde=True, color="#54A24B")
    plt.title("GMM En Yüksek Küme Üyelik Olasılığı")
    plt.xlabel("Üyelik güveni")
    plt.tight_layout()
    plt.savefig(FIGURES / "gmm_membership_confidence.png", dpi=140)
    plt.close()

    pca_yukleri = pd.DataFrame(pca.components_.T, index=OZELLIKLER, columns=["PC1", "PC2"])
    return {"pca": pca, "pca_loadings": pca_yukleri, "gmm_confidence": gmm_olasilik}


def main() -> None:
    veri = veriyi_uret()
    inceleme = veriyi_incele(veri)
    doldurulmus, X, _, _ = on_isleme(veri)
    k_sonuclari, en_iyi_k = k_degerlerini_incele(X)
    sonuc = modelleri_karsilastir(X, en_iyi_k, veri["known_profile"])
    gorsel_sonuc = gorselleri_uret(veri, doldurulmus, X, k_sonuclari, sonuc)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    veri.to_csv(DATA / "digital_customer_behavior.csv", index=False)
    k_sonuclari.round(4).to_csv(RESULTS / "k_selection_metrics.csv", index=False)
    sonuc["metrics"].round(4).to_csv(RESULTS / "model_comparison.csv", index=False)
    gorsel_sonuc["pca_loadings"].round(4).to_csv(RESULTS / "pca_loadings.csv")

    segmentler = doldurulmus.copy()
    segmentler.insert(0, "customer_id", veri["customer_id"])
    segmentler["cluster"] = sonuc["labels"]["KMeans"]
    profiller = segmentler.groupby("cluster")[OZELLIKLER].mean().round(3)
    profiller["ClusterSize"] = segmentler.groupby("cluster").size()
    adlar = {kume: segment_adi(satir) for kume, satir in profiller.iterrows()}
    profiller["segment_name"] = pd.Series(adlar)
    segmentler["segment_name"] = segmentler["cluster"].map(adlar)
    profiller.to_csv(RESULTS / "cluster_profiles.csv")
    segmentler.to_csv(RESULTS / "customer_segments.csv", index=False)

    ozet = {
        "data_review": inceleme,
        "selected_k": en_iyi_k,
        "dbscan_parameters": sonuc["dbscan_parameters"],
        "pca_explained_variance": gorsel_sonuc["pca"].explained_variance_ratio_.round(4).tolist(),
        "segment_counts": segmentler["segment_name"].value_counts().to_dict(),
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(ozet, dosya, ensure_ascii=False, indent=2)

    print("DENETİMSİZ ÖĞRENME — DİJİTAL MÜŞTERİ SEGMENTASYONU")
    print("=" * 62)
    print(f"Veri boyutu: {veri.shape}")
    print(f"Eksik değer: {inceleme['missing_values']} | Tekrarlı satır: {inceleme['duplicate_rows']}")
    print("\nSayısal özet:\n", veri[OZELLIKLER].describe().round(2).to_string())
    print("\nK seçimi:\n", k_sonuclari.round(4).to_string(index=False))
    print(f"\nSeçilen küme sayısı: {en_iyi_k}")
    print("\nAlgoritma karşılaştırması:\n", sonuc["metrics"].round(4).to_string(index=False))
    print("\nK-Means küme profilleri:\n", profiller.to_string())
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
