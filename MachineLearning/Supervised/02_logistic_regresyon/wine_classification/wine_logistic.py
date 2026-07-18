"""Wine veri setiyle çok sınıflı Logistic Regression uygulaması."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

KLASOR = Path(__file__).resolve().parent
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42


def veriyi_yukle() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    veri = load_wine(as_frame=True)
    return veri.data, veri.target.rename("wine_class"), list(veri.target_names)


def model_olustur() -> Pipeline:
    return Pipeline(
        [
            ("olcekleme", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )


def modeli_egit(X: pd.DataFrame, y: pd.Series, sinif_adlari: list[str]) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = model_olustur()
    model.fit(X_train, y_train)
    tahmin = model.predict(X_test)
    olasilik = model.predict_proba(X_test)

    metrikler = {
        "accuracy": round(float(accuracy_score(y_test, tahmin)), 4),
        "precision_macro": round(float(precision_score(y_test, tahmin, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, tahmin, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_test, tahmin, average="macro")), 4),
        "roc_auc_ovr_macro": round(float(roc_auc_score(y_test, olasilik, multi_class="ovr", average="macro")), 4),
        "log_loss": round(float(log_loss(y_test, olasilik)), 4),
    }

    model_katsayilari = model.named_steps["model"].coef_
    katsayilar = pd.DataFrame(model_katsayilari.T, index=X.columns, columns=sinif_adlari)
    katsayi_uzun = katsayilar.reset_index(names="feature").melt(
        id_vars="feature", var_name="class_name", value_name="coefficient"
    )

    rapor = pd.DataFrame(
        classification_report(
            y_test,
            tahmin,
            target_names=sinif_adlari,
            output_dict=True,
            zero_division=0,
        )
    ).T
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_skorlari = cross_val_score(model_olustur(), X, y, cv=cv, scoring="f1_macro")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "tahmin": tahmin,
        "olasilik": olasilik,
        "metrikler": metrikler,
        "katsayilar": katsayilar,
        "katsayi_uzun": katsayi_uzun,
        "sinif_raporu": rapor,
        "cv_f1_macro": cv_skorlari,
    }


def gorselleri_uret(X: pd.DataFrame, y: pd.Series, sinif_adlari: list[str], sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y.map(dict(enumerate(sinif_adlari))), color="#7A5195")
    plt.xlabel("Şarap sınıfı")
    plt.ylabel("Örnek sayısı")
    plt.title("Şarap Sınıf Dağılımı")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_distribution.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        sonuc["y_test"],
        sonuc["tahmin"],
        display_labels=sinif_adlari,
        cmap="Purples",
        colorbar=False,
    )
    plt.title("Çok Sınıflı Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 7))
    sns.heatmap(sonuc["katsayilar"], cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.xlabel("Sınıf")
    plt.ylabel("Özellik")
    plt.title("Sınıflara Göre Logistic Regression Katsayıları")
    plt.tight_layout()
    plt.savefig(FIGURES / "coefficient_heatmap.png", dpi=140)
    plt.close()

    y_ikili = label_binarize(sonuc["y_test"], classes=np.arange(len(sinif_adlari)))
    plt.figure(figsize=(7, 5))
    for sinif_no, sinif_adi in enumerate(sinif_adlari):
        fpr, tpr, _ = roc_curve(y_ikili[:, sinif_no], sonuc["olasilik"][:, sinif_no])
        plt.plot(fpr, tpr, label=sinif_adi)
    plt.plot([0, 1], [0, 1], "k--", label="Rastgele tahmin")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Eğrileri")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "multiclass_roc.png", dpi=140)
    plt.close()

    olcekli = StandardScaler().fit_transform(X)
    pca_koordinatlari = PCA(n_components=2).fit_transform(olcekli)
    pca_df = pd.DataFrame(pca_koordinatlari, columns=["PC1", "PC2"])
    pca_df["class"] = y.map(dict(enumerate(sinif_adlari))).to_numpy()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class", s=65, alpha=0.8)
    plt.title("PCA ile İki Boyutlu Şarap Sınıfları")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_projection.png", dpi=140)
    plt.close()


def main() -> None:
    X, y, sinif_adlari = veriyi_yukle()
    sonuc = modeli_egit(X, y, sinif_adlari)
    gorselleri_uret(X, y, sinif_adlari, sonuc)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rapor = {
        "veri_boyutu": list(X.shape),
        "sinif_adlari": sinif_adlari,
        "test_metrikleri": sonuc["metrikler"],
        "cv_f1_macro_ortalama": round(float(sonuc["cv_f1_macro"].mean()), 4),
        "cv_f1_macro_std": round(float(sonuc["cv_f1_macro"].std()), 4),
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as dosya:
        json.dump(rapor, dosya, ensure_ascii=False, indent=2)
    sonuc["katsayi_uzun"].to_csv(RESULTS / "coefficients.csv", index=False)
    sonuc["sinif_raporu"].to_csv(RESULTS / "classification_report.csv")

    print("ŞARAP TÜRÜ SINIFLANDIRMASI — LOGISTIC REGRESSION")
    print("=" * 52)
    print(f"Veri boyutu: {X.shape}")
    print(f"Sınıf dağılımı: {y.value_counts().sort_index().to_dict()}")
    print(f"Test metrikleri: {sonuc['metrikler']}")
    print(f"5-fold CV Macro-F1: {sonuc['cv_f1_macro'].mean():.4f} ± {sonuc['cv_f1_macro'].std():.4f}")
    print("\nHer sınıf için en etkili özellik:")
    for sinif_adi in sinif_adlari:
        seri = sonuc["katsayilar"][sinif_adi]
        ozellik = seri.abs().idxmax()
        print(f"- {sinif_adi}: {ozellik} ({seri.loc[ozellik]:+.3f})")
    print(f"\nGrafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
