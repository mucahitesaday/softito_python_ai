"""Makine sensörlerinde AdaBoost, Gradient Boosting ve opsiyonel XGBoost analizi."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "oil_quality",
    "load",
    "age_months",
    "error_count",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
]


def veriyi_uret(n: int = 1800):
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.72, 0.28],
        class_sep=1.05,
        flip_y=0.035,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=FEATURES)
    # İlk sekiz sütunu sensörlerin okunabilir aralıklarına dönüştürüyoruz.
    df["temperature"] = 70 + 12 * df["temperature"]
    df["vibration"] = np.clip(4 + 1.5 * df["vibration"], 0, None)
    df["pressure"] = 100 + 10 * df["pressure"]
    df["rpm"] = 1500 + 240 * df["rpm"]
    df["oil_quality"] = np.clip(70 + 12 * df["oil_quality"], 0, 100)
    df["load"] = np.clip(65 + 14 * df["load"], 0, 100)
    df["age_months"] = np.clip(48 + 18 * df["age_months"], 1, 144)
    df["error_count"] = np.clip(np.round(3 + 2 * df["error_count"]), 0, None)
    return df.round(3), pd.Series(y, name="failure")


def veriyi_incele(X: pd.DataFrame, y: pd.Series) -> dict:
    veri = X.assign(failure=y)
    return {
        "shape": X.shape,
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "class_counts": y.value_counts().sort_index(),
        "description": X[FEATURES[:8]].describe().round(2),
        "correlation_with_failure": (
            veri.corr(numeric_only=True)["failure"].drop("failure").sort_values(key=abs, ascending=False)
        ),
    }


def model_listesi() -> tuple[dict, bool]:
    modeller = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
            n_estimators=160,
            learning_rate=0.06,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }
    xgboost_kurulu = False
    try:
        from xgboost import XGBClassifier

        modeller["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        xgboost_kurulu = True
    except ImportError:
        pass
    return modeller, xgboost_kurulu


def modelleri_egit(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    modeller, xgboost_kurulu = model_listesi()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    satirlar = []
    egitilmis = {}
    tahminler = {}
    olasiliklar = {}
    raporlar = {}

    for ad, model in modeller.items():
        baslangic = time.perf_counter()
        model.fit(X_train, y_train)
        egitim_suresi = time.perf_counter() - baslangic
        tahmin = model.predict(X_test)
        olasilik = model.predict_proba(X_test)[:, 1]
        cv_skorlari = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        egitilmis[ad] = model
        tahminler[ad] = tahmin
        olasiliklar[ad] = olasilik
        raporlar[ad] = pd.DataFrame(classification_report(y_test, tahmin, output_dict=True)).T
        satirlar.append(
            {
                "model": ad,
                "train_accuracy": model.score(X_train, y_train),
                "test_accuracy": accuracy_score(y_test, tahmin),
                "precision": precision_score(y_test, tahmin),
                "recall": recall_score(y_test, tahmin),
                "f1": f1_score(y_test, tahmin),
                "roc_auc": roc_auc_score(y_test, olasilik),
                "cv_roc_auc_mean": cv_skorlari.mean(),
                "cv_roc_auc_std": cv_skorlari.std(),
                "fit_seconds": egitim_suresi,
            }
        )
    return {
        "models": egitilmis,
        "metrics": pd.DataFrame(satirlar),
        "predictions": tahminler,
        "probabilities": olasiliklar,
        "reports": raporlar,
        "X_test": X_test,
        "y_test": y_test,
        "xgboost_available": xgboost_kurulu,
    }


def ozellik_onemleri(sonuc: dict) -> pd.DataFrame:
    satirlar = []
    for model_adi, model in sonuc["models"].items():
        for ozellik, onem in zip(FEATURES, model.feature_importances_):
            satirlar.append({"model": model_adi, "feature": ozellik, "importance": onem})
    return pd.DataFrame(satirlar)


def gorselleri_uret(X: pd.DataFrame, y: pd.Series, sonuc: dict, onemler: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(x=y.map({0: "Normal", 1: "Arıza"}), ax=axes[0], color="#E15759")
    axes[0].set_title("Hedef Sınıf Dağılımı")
    sns.boxplot(data=X.assign(failure=y), x="failure", y="temperature", ax=axes[1])
    axes[1].set_title("Arıza Durumuna Göre Sıcaklık")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_overview.png", dpi=140)
    plt.close(fig)

    sonuc["metrics"].set_index("model")[["test_accuracy", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1)
    )
    plt.xticks(rotation=0)
    plt.title("Boosting Model Performansı")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(7, 5))
    for ad, olasilik in sonuc["probabilities"].items():
        fpr, tpr, _ = roc_curve(sonuc["y_test"], olasilik)
        auc = roc_auc_score(sonuc["y_test"], olasilik)
        plt.plot(fpr, tpr, label=f"{ad} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Boosting ROC Eğrileri")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_comparison.png", dpi=140)
    plt.close()

    en_iyi = sonuc["metrics"].loc[sonuc["metrics"]["roc_auc"].idxmax(), "model"]
    ConfusionMatrixDisplay.from_predictions(
        sonuc["y_test"], sonuc["predictions"][en_iyi], cmap="Reds", colorbar=False
    )
    plt.title(f"{en_iyi} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "best_confusion_matrix.png", dpi=140)
    plt.close()

    veri = onemler.query("model == @en_iyi").sort_values("importance").tail(10)
    plt.figure(figsize=(8, 5))
    plt.barh(veri["feature"], veri["importance"], color="#E15759")
    plt.title(f"{en_iyi} Özellik Önemleri")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=140)
    plt.close()


def main() -> None:
    X, y = veriyi_uret()
    inceleme = veriyi_incele(X, y)
    sonuc = modelleri_egit(X, y)
    onemler = ozellik_onemleri(sonuc)
    gorselleri_uret(X, y, sonuc, onemler)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X.assign(failure=y).to_csv(DATA / "machine_sensor_data.csv", index=False)
    sonuc["metrics"].round(4).to_csv(RESULTS / "model_comparison.csv", index=False)
    onemler.to_csv(RESULTS / "feature_importance.csv", index=False)
    for ad, rapor in sonuc["reports"].items():
        rapor.round(4).to_csv(RESULTS / f"{ad.lower()}_classification_report.csv")
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as dosya:
        json.dump(
            {
                "rows": len(X),
                "failure_rate": round(float(y.mean()), 4),
                "xgboost_available": sonuc["xgboost_available"],
            },
            dosya,
            indent=2,
        )

    print("BOOSTING — MAKİNE ARIZA RİSKİ")
    print("=" * 34)
    print(f"Veri boyutu: {inceleme['shape']}")
    print(f"Eksik değer: {inceleme['missing_values']} | Tekrarlı satır: {inceleme['duplicate_rows']}")
    print("\nSensör özeti:\n", inceleme["description"].to_string())
    print("\nArızayla en güçlü korelasyonlar:\n", inceleme["correlation_with_failure"].head(8).round(3).to_string())
    print("\nModel karşılaştırması:\n", sonuc["metrics"].round(4).to_string(index=False))
    print(f"\nXGBoost kurulu mu? {sonuc['xgboost_available']}")
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
