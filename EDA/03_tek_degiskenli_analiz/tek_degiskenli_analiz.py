"""03 — Palmer Penguins tek değişkenli analiz çalışması."""

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
SAYISAL = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
KATEGORIK = ["species", "island", "sex", "year"]


def veriyi_hazirla() -> pd.DataFrame:
    df = pd.read_csv(VERI_DOSYASI, na_values=["NA", ""]).drop_duplicates()
    for sutun in SAYISAL:
        df[sutun] = df[sutun].fillna(df.groupby("species")[sutun].transform("median"))
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0])
    return df


def main() -> None:
    df = veriyi_hazirla()
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("03 — TEK DEĞİŞKENLİ ANALİZ")
    print("=" * 30)
    print(f"Veri boyutu: {df.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for sutun, ax in zip(SAYISAL, np.array(axes).reshape(-1)):
        sns.histplot(df[sutun], kde=True, ax=ax, color="#2A9D8F")
        ax.set_title(f"{sutun} dağılımı")
        seri = df[sutun]
        print(
            f"{sutun}: ort={seri.mean():.2f}, medyan={seri.median():.2f}, "
            f"std={seri.std():.2f}, çarpıklık={seri.skew():.2f}"
        )
    plt.tight_layout()
    plt.savefig(FIGURES / "03_sayisal_degisken_dagilimlari.png", dpi=140, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for sutun, ax in zip(KATEGORIK, np.array(axes).reshape(-1)):
        siralama = df[sutun].value_counts().index
        sns.countplot(data=df, x=sutun, order=siralama, ax=ax, color="#E9C46A")
        ax.set_title(f"{sutun} frekansı")
        ax.tick_params(axis="x", rotation=20)
        print(f"{sutun} dağılımı: {df[sutun].value_counts().to_dict()}")
    plt.tight_layout()
    plt.savefig(FIGURES / "03_kategorik_degisken_dagilimlari.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nGrafikler kaydedildi: {FIGURES}")


if __name__ == "__main__":
    main()
