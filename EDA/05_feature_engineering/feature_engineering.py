"""05 — Palmer Penguins feature engineering çalışması."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

KLASOR = Path(__file__).resolve().parent
VERI_DOSYASI = KLASOR.parent / "data" / "penguins.csv"
FIGURES = KLASOR / "figures"
OUTPUTS = KLASOR / "outputs"
SAYISAL = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def ozellik_uret(df: pd.DataFrame) -> pd.DataFrame:
    sonuc = df.copy().drop_duplicates()
    for sutun in SAYISAL:
        sonuc[sutun] = sonuc[sutun].fillna(
            sonuc.groupby("species")[sutun].transform("median")
        )
    sonuc["sex"] = sonuc["sex"].fillna(sonuc["sex"].mode().iloc[0])

    sonuc["body_mass_kg"] = (sonuc["body_mass_g"] / 1000).round(3)
    sonuc["bill_area_index"] = (
        sonuc["bill_length_mm"] * sonuc["bill_depth_mm"]
    ).round(2)
    sonuc["flipper_mass_ratio"] = (
        sonuc["flipper_length_mm"] / sonuc["body_mass_kg"]
    ).round(2)
    sonuc["body_size_category"] = pd.qcut(
        sonuc["body_mass_g"], q=3, labels=["Küçük", "Orta", "Büyük"]
    )
    sonuc["sex_encoded"] = sonuc["sex"].map({"female": 0, "male": 1}).astype(int)

    one_hot = pd.get_dummies(sonuc["island"], prefix="island", dtype=int)
    return pd.concat([sonuc, one_hot], axis=1)


def main() -> None:
    ham = pd.read_csv(VERI_DOSYASI, na_values=["NA", ""])
    df = ozellik_uret(ham)
    yeni = [
        "body_mass_kg", "bill_area_index", "flipper_mass_ratio",
        "body_size_category", "sex_encoded",
        *[sutun for sutun in df.columns if sutun.startswith("island_")],
    ]

    print("05 — FEATURE ENGINEERING")
    print("=" * 25)
    print(f"Ham sütun sayısı: {ham.shape[1]}")
    print(f"İşlenmiş sütun sayısı: {df.shape[1]}")
    print(f"Yeni özellikler: {yeni}")
    print("\nYeni sayısal özellik özeti:\n", df[["body_mass_kg", "bill_area_index", "flipper_mass_ratio"]].describe().T.round(2))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for sutun, ax in zip(["body_mass_kg", "bill_area_index", "flipper_mass_ratio"], axes):
        sns.histplot(df[sutun], kde=True, ax=ax, color="#8E44AD")
        ax.set_title(sutun)
    plt.tight_layout()
    plt.savefig(FIGURES / "05_yeni_ozellik_dagilimlari.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="body_size_category", hue="species", palette="Set2")
    plt.title("Vücut Büyüklüğü Kategorileri")
    plt.tight_layout()
    plt.savefig(FIGURES / "05_vucut_buyuklugu_dagilimi.png", dpi=140, bbox_inches="tight")
    plt.close()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS / "processed_penguins.csv", index=False)
    print(f"\nİşlenmiş veri kaydedildi: {OUTPUTS / 'processed_penguins.csv'}")
    print(f"Grafikler kaydedildi: {FIGURES}")


if __name__ == "__main__":
    main()
