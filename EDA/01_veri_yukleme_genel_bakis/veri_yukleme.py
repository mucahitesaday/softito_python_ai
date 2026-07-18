"""01 — Palmer Penguins veri yükleme ve genel bakış çalışması."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

KLASOR = Path(__file__).resolve().parent
VERI_DOSYASI = KLASOR.parent / "data" / "penguins.csv"
FIGURES = KLASOR / "figures"


def veriyi_yukle(dosya: Path = VERI_DOSYASI) -> pd.DataFrame:
    if not dosya.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {dosya}")
    return pd.read_csv(dosya, na_values=["NA", ""])


def figur_kaydet(dosya_adi: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / dosya_adi, dpi=140, bbox_inches="tight")
    plt.close()


def sutun_ozeti(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "veri_tipi": df.dtypes.astype(str),
            "dolu": df.notna().sum(),
            "eksik": df.isna().sum(),
            "eksik_orani_yuzde": (df.isna().mean() * 100).round(2),
            "benzersiz": df.nunique(dropna=True),
        }
    )


def main() -> None:
    df = veriyi_yukle()
    print("01 — VERİ YÜKLEME VE GENEL BAKIŞ")
    print("=" * 38)
    print("\nİlk 5 satır:\n", df.head())
    print("\nSon 3 satır:\n", df.tail(3))
    print("\nRastgele 3 satır:\n", df.sample(3, random_state=42))
    print(f"\nBoyut: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Tekrarlı satır: {df.duplicated().sum()}")
    print("\nSütun özeti:\n", sutun_ozeti(df))
    print("\nSayısal özet:\n", df.describe().T.round(2))
    print("\nKategorik özet:\n", df.describe(include="object").T)

    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="YlOrRd")
    plt.title("Palmer Penguins — Eksik Değer Haritası")
    figur_kaydet("01_eksik_deger_haritasi.png")

    eksikler = df.isna().sum().sort_values()
    eksikler = eksikler[eksikler > 0]
    plt.figure(figsize=(8, 4))
    eksikler.plot(kind="barh", color="#E67E22")
    plt.xlabel("Eksik değer sayısı")
    plt.title("Sütun Bazında Eksik Değerler")
    figur_kaydet("01_sutun_eksik_deger_sayisi.png")
    print(f"\nGrafikler kaydedildi: {FIGURES}")


if __name__ == "__main__":
    main()
