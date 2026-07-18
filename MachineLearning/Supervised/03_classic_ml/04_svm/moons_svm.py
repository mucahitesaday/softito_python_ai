"""Doğrusal olmayan iki sınıfta ayrıntılı SVM kernel analizi."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42


def veriyi_uret(n: int = 700):
    X, y = make_moons(n_samples=n, noise=0.27, random_state=RANDOM_STATE)
    return pd.DataFrame(X, columns=["sensor_x", "sensor_y"]), pd.Series(y, name="class")


def veriyi_incele(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "shape": X.shape,
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "class_counts": y.value_counts().sort_index(),
        "description": X.describe().round(3),
        "correlation": X.corr().round(3),
    }


def svm_pipeline(kernel: str, **kwargs) -> Pipeline:
    return Pipeline(
        [
            ("olcekleme", StandardScaler()),
            ("model", SVC(kernel=kernel, probability=True, random_state=RANDOM_STATE, **kwargs)),
        ]
    )


def modelleri_egit(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    modeller = {
        "linear": svm_pipeline("linear", C=1),
        "polynomial": svm_pipeline("poly", degree=3, C=1),
        "rbf": svm_pipeline("rbf", C=1, gamma="scale"),
    }
    satirlar = []
    olasiliklar = {}
    tahminler = {}
    raporlar = {}

    for ad, model in modeller.items():
        model.fit(X_train, y_train)
        tahmin = model.predict(X_test)
        olasilik = model.predict_proba(X_test)[:, 1]
        tahminler[ad] = tahmin
        olasiliklar[ad] = olasilik
        raporlar[ad] = pd.DataFrame(classification_report(y_test, tahmin, output_dict=True)).T
        satirlar.append(
            {
                "model": ad,
                "train_accuracy": model.score(X_train, y_train),
                "test_accuracy": accuracy_score(y_test, tahmin),
                "accuracy": accuracy_score(y_test, tahmin),
                "f1": f1_score(y_test, tahmin),
                "roc_auc": roc_auc_score(y_test, olasilik),
                "support_vectors": int(model.named_steps["model"].n_support_.sum()),
            }
        )

    parametreler = {
        "model__C": [0.1, 1, 10, 50],
        "model__gamma": [0.1, 0.5, 1, 2],
    }
    arama = GridSearchCV(
        svm_pipeline("rbf"),
        parametreler,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
    )
    arama.fit(X_train, y_train)
    tahmin = arama.predict(X_test)
    olasilik = arama.predict_proba(X_test)[:, 1]
    modeller["rbf_tuned"] = arama.best_estimator_
    tahminler["rbf_tuned"] = tahmin
    olasiliklar["rbf_tuned"] = olasilik
    raporlar["rbf_tuned"] = pd.DataFrame(classification_report(y_test, tahmin, output_dict=True)).T
    satirlar.append(
        {
            "model": "rbf_tuned",
            "train_accuracy": arama.best_estimator_.score(X_train, y_train),
            "test_accuracy": accuracy_score(y_test, tahmin),
            "accuracy": accuracy_score(y_test, tahmin),
            "f1": f1_score(y_test, tahmin),
            "roc_auc": roc_auc_score(y_test, olasilik),
            "support_vectors": int(arama.best_estimator_.named_steps["model"].n_support_.sum()),
        }
    )
    cv_sonuclari = pd.DataFrame(arama.cv_results_)
    return {
        "models": modeller,
        "metrics": pd.DataFrame(satirlar),
        "search": arama,
        "cv_results": cv_sonuclari,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": tahminler,
        "probabilities": olasiliklar,
        "reports": raporlar,
    }


def gorselleri_uret(X: pd.DataFrame, y: pd.Series, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=X.assign(sinif=y), x="sensor_x", y="sensor_y", hue="sinif", palette="coolwarm")
    plt.title("Doğrusal Olmayan Moons Veri Seti")
    plt.tight_layout()
    plt.savefig(FIGURES / "data_overview.png", dpi=140)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x1 = np.linspace(X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5, 250)
    x2 = np.linspace(X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5, 250)
    xx, yy = np.meshgrid(x1, x2)
    grid = pd.DataFrame({"sensor_x": xx.ravel(), "sensor_y": yy.ravel()})
    for ax, ad in zip(axes, ["linear", "polynomial", "rbf_tuned"]):
        karar = sonuc["models"][ad].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, karar, alpha=0.25, cmap="coolwarm")
        ax.scatter(X["sensor_x"], X["sensor_y"], c=y, cmap="coolwarm", s=12, alpha=0.65)
        ax.set_title(ad)
    fig.suptitle("SVM Kernel Karar Sınırları")
    fig.tight_layout()
    fig.savefig(FIGURES / "decision_boundaries.png", dpi=140)
    plt.close(fig)

    plt.figure(figsize=(7, 5))
    for ad in ["linear", "polynomial", "rbf_tuned"]:
        fpr, tpr, _ = roc_curve(sonuc["y_test"], sonuc["probabilities"][ad])
        auc = roc_auc_score(sonuc["y_test"], sonuc["probabilities"][ad])
        plt.plot(fpr, tpr, label=f"{ad} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Kernel ROC Eğrileri")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_comparison.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        sonuc["y_test"], sonuc["predictions"]["rbf_tuned"], cmap="Oranges", colorbar=False
    )
    plt.title("Optimize RBF Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()


def main() -> None:
    X, y = veriyi_uret()
    inceleme = veriyi_incele(X, y)
    sonuc = modelleri_egit(X, y)
    gorselleri_uret(X, y, sonuc)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X.assign(class_label=y).to_csv(DATA / "moons_sensor_data.csv", index=False)
    sonuc["metrics"].round(4).to_csv(RESULTS / "kernel_comparison.csv", index=False)
    sonuc["reports"]["rbf_tuned"].round(4).to_csv(RESULTS / "classification_report.csv")
    sonuc["cv_results"].to_csv(RESULTS / "grid_search_results.csv", index=False)
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(
            {"rows": len(X), "best_params": sonuc["search"].best_params_},
            dosya,
            indent=2,
        )

    print("SVM KERNEL KARŞILAŞTIRMASI")
    print("=" * 31)
    print(f"Veri boyutu: {inceleme['shape']}")
    print(f"Eksik değer: {inceleme['missing_values']} | Tekrarlı satır: {inceleme['duplicate_rows']}")
    print("\nSayısal özet:\n", inceleme["description"].to_string())
    print("\nSınıf dağılımı:\n", inceleme["class_counts"].to_string())
    print("\nKernel karşılaştırması:\n", sonuc["metrics"].round(4).to_string(index=False))
    print(f"\nEn iyi RBF parametreleri: {sonuc['search'].best_params_}")
    print("\nOptimize RBF sınıflandırma raporu:\n", sonuc["reports"]["rbf_tuned"].round(4).to_string())
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
