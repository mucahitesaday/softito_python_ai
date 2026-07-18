"""02 — Palmer Penguins veri temizleme çalışması."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

KLASOR = Path(__file__).resolve().parent
VERI_DOSYASI = KLASOR.parent / "data" / "penguins.csv"
FIGURES = KLASOR / "figures"
OUTPUTS = KLASOR / "outputs"
SAYISAL = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
KATEGORIK = ["species", "island", "sex"]


def veriyi_temizle(df: pd.DataFrame) -> pd.DataFrame:
    temiz = df.copy().drop_duplicates().reset_index(drop=True)

    for sutun in KATEGORIK:
        temiz[sutun] = temiz[sutun].astype("string").str.strip().str.title()

    # Her türün vücut ölçüleri farklı olduğundan eksikleri genel medyan yerine
    # aynı türün medyanıyla doldurmak daha anlamlıdır.
    for sutun in SAYISAL:
        temiz[sutun] = temiz[sutun].fillna(
            temiz.groupby("species")[sutun].transform("median")
        )
        temiz[sutun] = temiz[sutun].fillna(temiz[sutun].median())

    for sutun in KATEGORIK:
        if temiz[sutun].isna().any():
            temiz[sutun] = temiz[sutun].fillna(temiz[sutun].mode().iloc[0])

    temiz["year"] = temiz["year"].astype(int)
    return temiz


def aykiri_deger_sayisi(seri: pd.Series) -> int:
    q1, q3 = seri.quantile([0.25, 0.75])
    iqr = q3 - q1
    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((seri < alt) | (seri > ust)).sum())


def main() -> None:
    ham = pd.read_csv(VERI_DOSYASI, na_values=["NA", ""])
    print("02 — VERİ TEMİZLEME")
    print("=" * 23)
    print(f"Ham boyut: {ham.shape}")
    print(f"Ham eksik değer: {ham.isna().sum().sum()}")
    print(f"Tekrarlı satır: {ham.duplicated().sum()}")

    temiz = veriyi_temizle(ham)
    print(f"Temiz boyut: {temiz.shape}")
    print(f"Kalan eksik değer: {temiz.isna().sum().sum()}")
    print("\nIQR aykırı değer sayıları:")
    for sutun in SAYISAL:
        print(f"- {sutun}: {aykiri_deger_sayisi(temiz[sutun])}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for sutun, ax in zip(SAYISAL, np.array(axes).reshape(-1)):
        sns.boxplot(y=temiz[sutun], ax=ax, color="#4C78A8")
        ax.set_title(f"{sutun} — Boxplot")
    plt.tight_layout()
    plt.savefig(FIGURES / "02_aykiri_deger_boxplot.png", dpi=140, bbox_inches="tight")
    plt.close()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    temiz.to_csv(OUTPUTS / "cleaned_penguins.csv", index=False)
    print(f"\nTemiz veri kaydedildi: {OUTPUTS / 'cleaned_penguins.csv'}")
    print(f"Grafik kaydedildi: {FIGURES}")


if __name__ == "__main__":
    main()
