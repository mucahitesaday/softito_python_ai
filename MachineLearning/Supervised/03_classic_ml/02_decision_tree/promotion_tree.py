"""Çalışan verisi üzerinde ayrıntılı Decision Tree sınıflandırması."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

KLASOR = Path(__file__).resolve().parent
DATA = KLASOR / "data"
FIGURES = KLASOR / "figures"
RESULTS = KLASOR / "results"
RANDOM_STATE = 42
SAYISAL = ["experience_years", "performance_score", "training_hours", "projects_completed"]
KATEGORIK = ["department", "education"]


def veriyi_uret(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    departmanlar = np.array(["engineering", "sales", "operations", "finance"])
    egitimler = np.array(["high_school", "bachelor", "master"])
    df = pd.DataFrame(
        {
            "experience_years": rng.integers(0, 21, n),
            "performance_score": rng.uniform(40, 100, n).round(2),
            "training_hours": rng.integers(0, 121, n),
            "projects_completed": rng.integers(1, 16, n),
            "department": rng.choice(departmanlar, n, p=[0.35, 0.25, 0.25, 0.15]),
            "education": rng.choice(egitimler, n, p=[0.20, 0.55, 0.25]),
        }
    )
    gizli_skor = (
        0.07 * (df["performance_score"] - 70)
        + 0.09 * df["experience_years"]
        + 0.018 * df["training_hours"]
        + 0.11 * df["projects_completed"]
        + (df["education"] == "master") * 0.7
        + (df["department"] == "engineering") * 0.35
        + rng.normal(0, 1.15, n)
        - 3.0
    )
    df["promoted"] = (gizli_skor > 0).astype(int)
    return df


def veriyi_incele(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "promotion_counts": df["promoted"].value_counts().sort_index(),
        "numeric_summary": df[SAYISAL].describe().round(2),
        "department_rates": df.groupby("department")["promoted"].mean().sort_values(ascending=False),
        "education_rates": df.groupby("education")["promoted"].mean().sort_values(ascending=False),
    }


def pipeline_olustur() -> Pipeline:
    donusturucu = ColumnTransformer(
        [("kategori", OneHotEncoder(handle_unknown="ignore", sparse_output=False), KATEGORIK)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("donusturucu", donusturucu),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )


def modeli_egit(df: pd.DataFrame) -> dict:
    X = df.drop(columns="promoted")
    y = df["promoted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    parametreler = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3, 4, 5, 7, None],
        "model__min_samples_leaf": [5, 15, 30],
    }
    arama = GridSearchCV(
        pipeline_olustur(),
        parametreler,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
    )
    arama.fit(X_train, y_train)
    tahmin = arama.predict(X_test)
    olasilik = arama.predict_proba(X_test)[:, 1]
    egitim_tahmini = arama.predict(X_train)

    adlar = arama.best_estimator_.named_steps["donusturucu"].get_feature_names_out()
    agac = arama.best_estimator_.named_steps["model"]
    onem = pd.DataFrame(
        {"feature": adlar, "importance": agac.feature_importances_}
    ).sort_values("importance", ascending=False)
    rapor = pd.DataFrame(classification_report(y_test, tahmin, output_dict=True)).T
    return {
        "search": arama,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prediction": tahmin,
        "metrics": {
            "train_accuracy": accuracy_score(y_train, egitim_tahmini),
            "test_accuracy": accuracy_score(y_test, tahmin),
            "f1": f1_score(y_test, tahmin),
            "roc_auc": roc_auc_score(y_test, olasilik),
        },
        "classification_report": rapor,
        "importance": onem,
        "feature_names": adlar,
        "rules": export_text(agac, feature_names=list(adlar)),
    }


def gorselleri_uret(df: pd.DataFrame, sonuc: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=df, x="performance_score", hue="promoted", kde=True, ax=axes[0])
    axes[0].set_title("Performans Skoru ve Terfi")
    oran = df.groupby("department")["promoted"].mean().sort_values()
    oran.plot(kind="barh", ax=axes[1], color="#59A14F")
    axes[1].set_title("Departmana Göre Terfi Oranı")
    axes[1].set_xlabel("Oran")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_overview.png", dpi=140)
    plt.close(fig)

    ConfusionMatrixDisplay.from_predictions(
        sonuc["y_test"],
        sonuc["prediction"],
        display_labels=["Terfi yok", "Terfi"],
        cmap="Greens",
        colorbar=False,
    )
    plt.title("Terfi Tahmini Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close()

    veri = sonuc["importance"].head(10).sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(veri["feature"], veri["importance"], color="#59A14F")
    plt.title("Decision Tree Özellik Önemleri")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=140)
    plt.close()

    plt.figure(figsize=(20, 9))
    plot_tree(
        sonuc["search"].best_estimator_.named_steps["model"],
        feature_names=sonuc["feature_names"],
        class_names=["Hayır", "Evet"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Karar Ağacının İlk Dört Seviyesi")
    plt.tight_layout()
    plt.savefig(FIGURES / "tree_structure.png", dpi=140)
    plt.close()


def main() -> None:
    df = veriyi_uret()
    inceleme = veriyi_incele(df)
    sonuc = modeli_egit(df)
    gorselleri_uret(df, sonuc)

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "employee_promotion.csv", index=False)
    sonuc["importance"].to_csv(RESULTS / "feature_importance.csv", index=False)
    sonuc["classification_report"].round(4).to_csv(RESULTS / "classification_report.csv")
    (RESULTS / "decision_rules.txt").write_text(sonuc["rules"], encoding="utf-8")
    rapor = {
        "rows": len(df),
        "promotion_rate": round(float(df["promoted"].mean()), 4),
        "best_params": sonuc["search"].best_params_,
        "metrics": {k: round(float(v), 4) for k, v in sonuc["metrics"].items()},
    }
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as dosya:
        json.dump(rapor, dosya, ensure_ascii=False, indent=2)

    print("DECISION TREE — ÇALIŞAN TERFİ TAHMİNİ")
    print("=" * 43)
    print(f"Veri boyutu: {inceleme['shape']}")
    print(f"Eksik değer: {inceleme['missing_values']} | Tekrarlı satır: {inceleme['duplicate_rows']}")
    print("\nSayısal özet:\n", inceleme["numeric_summary"].to_string())
    print("\nDepartmana göre terfi oranı:\n", inceleme["department_rates"].round(3).to_string())
    print(f"\nEn iyi parametreler: {sonuc['search'].best_params_}")
    print(f"Metrikler: {rapor['metrics']}")
    print("\nSınıflandırma raporu:\n", sonuc["classification_report"].round(4).to_string())
    print("\nEn önemli özellikler:\n", sonuc["importance"].head(8).to_string(index=False))
    print("\nKarar kurallarının ilk bölümü:\n", sonuc["rules"][:1200])
    print(f"Grafikler: {FIGURES}")
    print(f"Sonuçlar: {RESULTS}")


if __name__ == "__main__":
    main()
