"""04 — Palmer Penguins çift değişkenli analiz çalışması."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

KLASOR = Path(__file__).resolve().parent
VERI_DOSYASI = KLASOR.parent / "data" / "penguins.csv"
FIGURES = KLASOR / "figures"
SAYISAL = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def veriyi_hazirla() -> pd.DataFrame:
    df = pd.read_csv(VERI_DOSYASI, na_values=["NA", ""]).drop_duplicates()
    for sutun in SAYISAL:
        df[sutun] = df[sutun].fillna(df.groupby("species")[sutun].transform("median"))
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0])
    return df


def kaydet(dosya_adi: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / dosya_adi, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = veriyi_hazirla()
    print("04 — ÇİFT DEĞİŞKENLİ ANALİZ")
    print("=" * 32)

    korelasyon = df[SAYISAL].corr()
    print("\nKorelasyon matrisi:\n", korelasyon.round(2))
    plt.figure(figsize=(8, 6))
    sns.heatmap(korelasyon, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Penguen Ölçümleri Korelasyon Matrisi")
    kaydet("04_korelasyon_heatmap.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="bill_length_mm", y="bill_depth_mm",
        hue="species", style="sex", alpha=0.75
    )
    plt.title("Gaga Uzunluğu ve Derinliği — Tür ve Cinsiyet")
    kaydet("04_scatter_iliski.png")

    capraz = pd.crosstab(df["island"], df["species"], normalize="index") * 100
    print("\nAda içindeki tür yüzdeleri:\n", capraz.round(1))
    capraz.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="Set2")
    plt.ylabel("Yüzde")
    plt.title("Adalara Göre Penguen Türleri")
    plt.xticks(rotation=0)
    kaydet("04_kategorik_kategorik_iliski.png")

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="sex", palette="Set2")
    plt.title("Tür ve Cinsiyete Göre Vücut Kütlesi")
    kaydet("04_kategorik_sayisal_iliski.png")
    print(f"\nDört grafik kaydedildi: {FIGURES}")


if __name__ == "__main__":
    main()
