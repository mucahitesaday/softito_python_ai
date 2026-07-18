"""Özgün e-ticaret işlem verisi üzerinde RFM müşteri segmentasyonu."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
REFERANS_TARIHI = pd.Timestamp("2026-01-01")

MUSTERI_PROFILLERI = {
    "Sampiyon": {"weight": 0.13, "recency": (2, 25), "frequency": (12, 22), "invoice_value": (450, 850)},
    "Sadik": {"weight": 0.24, "recency": (20, 90), "frequency": (7, 14), "invoice_value": (260, 520)},
    "Potansiyel": {"weight": 0.25, "recency": (5, 75), "frequency": (2, 6), "invoice_value": (160, 380)},
    "Riskli": {"weight": 0.20, "recency": (120, 260), "frequency": (5, 12), "invoice_value": (220, 480)},
    "Uykuda": {"weight": 0.18, "recency": (230, 360), "frequency": (1, 4), "invoice_value": (80, 240)},
}


def islemleri_uret(musteri_sayisi: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    profil_adlari = list(MUSTERI_PROFILLERI)
    agirliklar = [MUSTERI_PROFILLERI[ad]["weight"] for ad in profil_adlari]
    satirlar = []
    fatura_no = 1

    for musteri_no in range(1, musteri_sayisi + 1):
        profil = rng.choice(profil_adlari, p=agirliklar)
        ayar = MUSTERI_PROFILLERI[profil]
        recency = int(rng.integers(*ayar["recency"]))
        frequency = int(rng.integers(*ayar["frequency"]))
        son_tarih = REFERANS_TARIHI - pd.Timedelta(days=recency)
        onceki_araliklar = np.sort(rng.integers(0, max(2, 365 - recency), size=frequency))[::-1]
        tarihler = [son_tarih - pd.Timedelta(days=int(gun)) for gun in onceki_araliklar]

        for tarih in tarihler:
            fatura = f"INV{fatura_no:06d}"
            fatura_no += 1
            hedef_tutar = float(rng.uniform(*ayar["invoice_value"]))
            urun_sayisi = int(rng.integers(1, 5))
            paylar = rng.dirichlet(np.ones(urun_sayisi))
            for urun_no, pay in enumerate(paylar, start=1):
                adet = int(rng.integers(1, 6))
                birim_fiyat = max(5.0, hedef_tutar * pay / adet)
                satirlar.append(
                    {
                        "invoice_no": fatura,
                        "customer_id": f"CU{musteri_no:04d}",
                        "invoice_date": tarih,
                        "product_code": f"P{rng.integers(1, 180):03d}",
                        "quantity": adet,
                        "unit_price": round(birim_fiyat, 2),
                        "is_cancelled": False,
                        "known_profile": profil,
                    }
                )

    veri = pd.DataFrame(satirlar)
    # İptal ve iade kayıtları temizleme adımını göstermek amacıyla eklenir.
    ornek = veri.sample(frac=0.015, random_state=RANDOM_STATE).copy()
    ornek["invoice_no"] = "C" + ornek["invoice_no"].astype(str)
    ornek["quantity"] = -ornek["quantity"].abs()
    ornek["is_cancelled"] = True
    return pd.concat([veri, ornek], ignore_index=True).sort_values("invoice_date").reset_index(drop=True)


def islemleri_temizle(veri: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    temiz = veri.dropna(subset=["customer_id", "invoice_date"]).copy()
    temiz["invoice_date"] = pd.to_datetime(temiz["invoice_date"])
    temiz = temiz[(temiz["quantity"] > 0) & (temiz["unit_price"] > 0)]
    temiz = temiz[~temiz["is_cancelled"]]
    temiz = temiz[~temiz["invoice_no"].astype(str).str.startswith("C")]
    temiz["total_price"] = temiz["quantity"] * temiz["unit_price"]
    rapor = {
        "raw_rows": int(len(veri)),
        "clean_rows": int(len(temiz)),
        "removed_rows": int(len(veri) - len(temiz)),
        "customer_count": int(temiz["customer_id"].nunique()),
        "invoice_count": int(temiz["invoice_no"].nunique()),
    }
    return temiz, rapor


def rfm_olustur(temiz: pd.DataFrame) -> pd.DataFrame:
    rfm = temiz.groupby("customer_id").agg(
        Recency=("invoice_date", lambda x: (REFERANS_TARIHI - x.max()).days),
        Frequency=("invoice_no", "nunique"),
        Monetary=("total_price", "sum"),
        FirstPurchase=("invoice_date", "min"),
        LastPurchase=("invoice_date", "max"),
    )
    rfm["CustomerLifetime"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AverageInvoiceValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm.reset_index()


def rfm_skorlari_ekle(rfm: pd.DataFrame) -> pd.DataFrame:
    sonuc = rfm.copy()
    sonuc["R_Score"] = pd.qcut(sonuc["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    sonuc["F_Score"] = pd.qcut(sonuc["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    sonuc["M_Score"] = pd.qcut(sonuc["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    sonuc["RFM_Score"] = sonuc[["R_Score", "F_Score", "M_Score"]].astype(str).agg("".join, axis=1)
    sonuc["RFM_Total"] = sonuc[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return sonuc


def kumeleme_verisi(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    donusmus = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    return scaler.fit_transform(donusmus), scaler


def k_sec(X: np.ndarray) -> tuple[pd.DataFrame, int]:
    satirlar = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        etiket = model.fit_predict(X)
        satirlar.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, etiket),
                "davies_bouldin": davies_bouldin_score(X, etiket),
            }
        )
    sonuclar = pd.DataFrame(satirlar)
    en_iyi_k = int(sonuclar.loc[sonuclar["silhouette"].idxmax(), "k"])
    return sonuclar, en_iyi_k


def segment_etiketleri(profiller: pd.DataFrame) -> dict[int, str]:
    """Birbirinden farklı ve iş açısından yorumlanabilir segment adları üretir."""
    degerler = profiller[["Recency_mean", "Frequency_mean", "Monetary_mean"]]
    z = (degerler - degerler.mean()) / degerler.std(ddof=0).replace(0, 1)
    kalan = set(profiller.index)
    etiketler: dict[int, str] = {}

    sampiyon = (-z["Recency_mean"] + z["Frequency_mean"] + z["Monetary_mean"]).idxmax()
    etiketler[int(sampiyon)] = "Sampiyonlar"
    kalan.remove(sampiyon)

    if kalan:
        uykuda = (z.loc[list(kalan), "Recency_mean"] - z.loc[list(kalan), "Frequency_mean"] - z.loc[list(kalan), "Monetary_mean"]).idxmax()
        etiketler[int(uykuda)] = "Uykudaki_Musteriler"
        kalan.remove(uykuda)
    if kalan:
        riskli = (z.loc[list(kalan), "Recency_mean"] + z.loc[list(kalan), "Frequency_mean"] + 0.5 * z.loc[list(kalan), "Monetary_mean"]).idxmax()
        etiketler[int(riskli)] = "Risk_Altinda"
        kalan.remove(riskli)
    if kalan:
        sadik = (-0.5 * z.loc[list(kalan), "Recency_mean"] + z.loc[list(kalan), "Frequency_mean"] + z.loc[list(kalan), "Monetary_mean"]).idxmax()
        etiketler[int(sadik)] = "Sadik_Musteriler"
        kalan.remove(sadik)
    for kume in kalan:
        etiketler[int(kume)] = "Potansiyel_Musteriler"
    return etiketler


def segmentlere_ayir(rfm: pd.DataFrame) -> dict:
    X, scaler = kumeleme_verisi(rfm)
    k_sonuclari, en_iyi_k = k_sec(X)
    model = KMeans(n_clusters=en_iyi_k, n_init=30, random_state=RANDOM_STATE)
    etiketler = model.fit_predict(X)
    segmentli = rfm_skorlari_ekle(rfm)
    segmentli["Cluster"] = etiketler
    profiller = segmentli.groupby("Cluster").agg(
        CustomerCount=("customer_id", "count"),
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        AverageInvoice_mean=("AverageInvoiceValue", "mean"),
        RFM_Total_mean=("RFM_Total", "mean"),
    ).round(2)
    profiller["CustomerPct"] = (profiller["CustomerCount"] / len(segmentli) * 100).round(1)
    adlar = segment_etiketleri(profiller)
    profiller["Segment"] = pd.Series(adlar)
    segmentli["Segment"] = segmentli["Cluster"].map(adlar)
    return {
        "data": segmentli,
        "profiles": profiller,
        "k_results": k_sonuclari,
        "best_k": en_iyi_k,
        "model": model,
        "scaler": scaler,
        "X": X,
        "silhouette": silhouette_score(X, etiketler),
        "davies_bouldin": davies_bouldin_score(X, etiketler),
    }


def gorselleri_uret(sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    veri = sonuc["data"]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sutun, renk in zip(axes, ["Recency", "Frequency", "Monetary"], ["#4C78A8", "#F58518", "#54A24B"]):
        sns.histplot(veri[sutun], bins=30, kde=True, ax=ax, color=renk)
        ax.set_title(f"{sutun} Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "rfm_distributions.png", dpi=140)
    plt.close()

    k_sonuclari = sonuc["k_results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_sonuclari["k"], k_sonuclari["inertia"], "o-")
    axes[0].set(title="RFM Elbow Yöntemi", xlabel="K", ylabel="Inertia")
    axes[1].plot(k_sonuclari["k"], k_sonuclari["silhouette"], "o-", color="#E45756")
    axes[1].set(title="RFM Silhouette Skoru", xlabel="K", ylabel="Silhouette")
    plt.tight_layout()
    plt.savefig(FIGURES / "rfm_optimal_k.png", dpi=140)
    plt.close()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    noktalar = ax.scatter(veri["Recency"], veri["Frequency"], np.log1p(veri["Monetary"]), c=veri["Cluster"], cmap="tab10", s=25, alpha=0.75)
    ax.set(xlabel="Recency", ylabel="Frequency", zlabel="log(Monetary)", title="Üç Boyutlu RFM Segmentleri")
    fig.colorbar(noktalar, ax=ax, label="Küme", shrink=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES / "rfm_3d_segments.png", dpi=140)
    plt.close()

    standart_profil = sonuc["profiles"][["Recency_mean", "Frequency_mean", "Monetary_mean"]].copy()
    standart_profil = (standart_profil - standart_profil.mean()) / standart_profil.std(ddof=0)
    standart_profil.index = sonuc["profiles"]["Segment"]
    standart_profil.index.name = "Segment"
    plt.figure(figsize=(8, 5))
    sns.heatmap(standart_profil, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Standartlaştırılmış Küme Profilleri")
    plt.tight_layout()
    plt.savefig(FIGURES / "segment_profile_heatmap.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 5))
    sirali = veri["Segment"].value_counts().sort_values()
    sirali.plot.barh(color="#72B7B2")
    plt.xlabel("Müşteri sayısı")
    plt.title("RFM Segment Büyüklükleri")
    plt.tight_layout()
    plt.savefig(FIGURES / "segment_sizes.png", dpi=140)
    plt.close()


def main() -> None:
    ham_veri = islemleri_uret()
    temiz, temizlik_raporu = islemleri_temizle(ham_veri)
    rfm = rfm_olustur(temiz)
    sonuc = segmentlere_ayir(rfm)
    gorselleri_uret(sonuc)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ham_veri.to_csv(DATA / "ecommerce_transactions.csv", index=False)
    sonuc["data"].to_csv(RESULTS / "rfm_customer_segments.csv", index=False)
    sonuc["profiles"].to_csv(RESULTS / "rfm_segment_profiles.csv")
    sonuc["k_results"].round(4).to_csv(RESULTS / "rfm_k_selection.csv", index=False)

    ozet = {
        "cleaning": temizlik_raporu,
        "rfm_customer_count": int(len(rfm)),
        "selected_k": sonuc["best_k"],
        "silhouette": round(float(sonuc["silhouette"]), 4),
        "davies_bouldin": round(float(sonuc["davies_bouldin"]), 4),
        "segment_counts": sonuc["data"]["Segment"].value_counts().to_dict(),
    }
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(ozet, dosya, ensure_ascii=False, indent=2)

    print("E-TİCARET RFM MÜŞTERİ SEGMENTASYONU")
    print("=" * 45)
    print(f"Ham işlem sayısı: {temizlik_raporu['raw_rows']}")
    print(f"Temiz işlem sayısı: {temizlik_raporu['clean_rows']}")
    print(f"Çıkarılan iptal/iade satırı: {temizlik_raporu['removed_rows']}")
    print(f"Müşteri sayısı: {len(rfm)} | Fatura sayısı: {temizlik_raporu['invoice_count']}")
    print("\nRFM sayısal özeti:\n", rfm[["Recency", "Frequency", "Monetary", "AverageInvoiceValue"]].describe().round(2).to_string())
    print("\nK seçimi:\n", sonuc["k_results"].round(4).to_string(index=False))
    print(f"\nSeçilen K: {sonuc['best_k']} | Silhouette: {sonuc['silhouette']:.4f} | Davies-Bouldin: {sonuc['davies_bouldin']:.4f}")
    print("\nSegment profilleri:\n", sonuc["profiles"].to_string())
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
