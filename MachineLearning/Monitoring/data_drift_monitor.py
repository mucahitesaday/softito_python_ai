"""İstatistiksel testler ve sınıflandırıcıyla uçtan uca data drift izleme."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

KLASOR = Path(__file__).resolve().parent
DATA, FIGURES, RESULTS = KLASOR / "data", KLASOR / "figures", KLASOR / "results"
RANDOM_STATE = 42
SAYISAL = ["age", "monthly_income", "monthly_transactions", "average_basket", "engagement_score"]
KATEGORIK = ["channel", "region", "membership"]


def veri_uret(adet: int = 2500, drift_seviyesi: float = 0.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38 + 4 * drift_seviyesi, 11, adet), 18, 75)
    income = rng.lognormal(10.35 + 0.22 * drift_seviyesi, 0.38, adet)
    transactions = rng.poisson(7 + 4 * drift_seviyesi, adet) + 1
    basket = np.clip(320 + income / 260 + rng.normal(0, 90 + 20 * drift_seviyesi, adet), 30, None)
    engagement = np.clip(62 - 12 * drift_seviyesi + 0.7 * transactions + rng.normal(0, 10, adet), 0, 100)
    channel = rng.choice(["Mobile", "Web", "Store"], adet, p=np.array([0.48 + .20*drift_seviyesi, .37 - .12*drift_seviyesi, .15 - .08*drift_seviyesi]))
    region = rng.choice(["Marmara", "Ege", "Ic_Anadolu", "Akdeniz"], adet, p=[.42, .22, .21, .15])
    membership = rng.choice(["Standard", "Silver", "Gold"], adet, p=[.57 + .10*drift_seviyesi, .29, .14 - .10*drift_seviyesi])
    return pd.DataFrame({"age":age,"monthly_income":income,"monthly_transactions":transactions,"average_basket":basket,"engagement_score":engagement,"channel":channel,"region":region,"membership":membership}).round(3)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    sinirlar = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    sinirlar[0], sinirlar[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=sinirlar)[0] / len(reference)
    cur = np.histogram(current, bins=sinirlar)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur-ref)*np.log(cur/ref)))


def js_divergence(reference: pd.Series, current: pd.Series) -> float:
    kategoriler = sorted(set(reference.astype(str)) | set(current.astype(str)))
    p = reference.astype(str).value_counts(normalize=True).reindex(kategoriler, fill_value=0).to_numpy()
    q = current.astype(str).value_counts(normalize=True).reindex(kategoriler, fill_value=0).to_numpy()
    return float(jensenshannon(p, q, base=2) ** 2)


def drift_raporu(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    satirlar=[]
    for col in SAYISAL:
        ks, p = ks_2samp(reference[col], current[col])
        psideger = psi(reference[col], current[col])
        satirlar.append({"feature":col,"type":"numeric","test_statistic":ks,"p_value":p,"psi":psideger,"js_divergence":np.nan,"drift":bool(p<.05 and psideger>=.10)})
    for col in KATEGORIK:
        cats=sorted(set(reference[col])|set(current[col]))
        tablo=np.vstack([reference[col].value_counts().reindex(cats,fill_value=0),current[col].value_counts().reindex(cats,fill_value=0)])
        chi,p,_,_=chi2_contingency(tablo)
        js=js_divergence(reference[col],current[col])
        satirlar.append({"feature":col,"type":"categorical","test_statistic":chi,"p_value":p,"psi":np.nan,"js_divergence":js,"drift":bool(p<.05 and js>=.01)})
    return pd.DataFrame(satirlar)


def domain_classifier(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    X=pd.concat([reference,current],ignore_index=True); y=np.r_[np.zeros(len(reference)),np.ones(len(current))]
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),KATEGORIK)],remainder="passthrough")
    model=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=220,max_depth=7,random_state=RANDOM_STATE,n_jobs=-1))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=RANDOM_STATE)
    model.fit(Xtr,ytr); probability=model.predict_proba(Xte)[:,1]
    return {"model":model,"roc_auc":float(roc_auc_score(yte,probability))}


def aylik_izleme() -> pd.DataFrame:
    ref=veri_uret(seed=100); rows=[]
    for ay,seviye in enumerate([0,.05,.12,.22,.38,.60],start=1):
        cur=veri_uret(drift_seviyesi=seviye,seed=100+ay)
        rapor=drift_raporu(ref,cur)
        rows.append({"month":ay,"drift_level":seviye,"drifted_features":int(rapor.drift.sum()),"mean_numeric_psi":rapor.psi.mean(),"domain_auc":domain_classifier(ref,cur)["roc_auc"]})
    return pd.DataFrame(rows)


def gorseller(reference,current,report,monthly):
    FIGURES.mkdir(parents=True,exist_ok=True); sns.set_theme(style="whitegrid")
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,col in zip(axes.flat,SAYISAL):
        sns.kdeplot(reference[col],ax=ax,label="Reference"); sns.kdeplot(current[col],ax=ax,label="Production"); ax.set_title(col); ax.legend()
    axes.flat[-1].axis("off"); plt.tight_layout(); plt.savefig(FIGURES/"numeric_distribution_shift.png",dpi=140); plt.close()
    plt.figure(figsize=(10,4)); sns.barplot(data=report,x="feature",y="test_statistic",hue="drift"); plt.xticks(rotation=25); plt.title("Özellik Bazında Drift Testleri"); plt.tight_layout(); plt.savefig(FIGURES/"feature_drift_tests.png",dpi=140); plt.close()
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(monthly.month,monthly.domain_auc,"o-",label="Domain AUC"); ax.plot(monthly.month,monthly.mean_numeric_psi,"o-",label="Ortalama PSI"); ax.axhline(.75,color="red",ls="--",label="AUC alarm"); ax.legend(); ax.set(xlabel="Ay",title="Aylık Drift İzleme"); plt.tight_layout(); plt.savefig(FIGURES/"monthly_drift_monitoring.png",dpi=140); plt.close()


def main():
    reference=veri_uret(seed=42); production=veri_uret(drift_seviyesi=.45,seed=43)
    report=drift_raporu(reference,production); domain=domain_classifier(reference,production); monthly=aylik_izleme(); gorseller(reference,production,report,monthly)
    DATA.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    reference.to_csv(DATA/"reference_customers.csv",index=False); production.to_csv(DATA/"production_customers.csv",index=False)
    report.round(5).to_csv(RESULTS/"feature_drift_report.csv",index=False); monthly.round(5).to_csv(RESULTS/"monthly_monitoring.csv",index=False)
    with open(RESULTS/"summary.json","w",encoding="utf-8") as f: json.dump({"reference_shape":list(reference.shape),"production_shape":list(production.shape),"drifted_features":report.query("drift").feature.tolist(),"domain_classifier_auc":round(domain["roc_auc"],4)},f,ensure_ascii=False,indent=2)
    print("DATA DRIFT İZLEME"); print("="*35); print(f"Reference: {reference.shape} | Production: {production.shape}"); print("\nÖzellik raporu:\n",report.round(4).to_string(index=False)); print(f"\nDomain classifier ROC-AUC: {domain['roc_auc']:.4f}"); print("\nAylık izleme:\n",monthly.round(4).to_string(index=False)); print(f"\nGrafikler: {FIGURES}")

if __name__=="__main__": main()
