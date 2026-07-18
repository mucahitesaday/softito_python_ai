"""Sentetik çalışan verisiyle Decision Tree sınıflandırması."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42

def veriyi_uret(n:int=1200)->pd.DataFrame:
    rng=np.random.default_rng(RANDOM_STATE); departments=np.array(["engineering","sales","operations","finance"]); education=np.array(["high_school","bachelor","master"])
    df=pd.DataFrame({"experience_years":rng.integers(0,21,n),"performance_score":rng.uniform(40,100,n),"training_hours":rng.integers(0,121,n),"projects_completed":rng.integers(1,16,n),"department":rng.choice(departments,n,p=[.35,.25,.25,.15]),"education":rng.choice(education,n,p=[.2,.55,.25])})
    skor=.07*(df.performance_score-70)+.09*df.experience_years+.018*df.training_hours+.11*df.projects_completed+(df.education=="master")*.7+(df.department=="engineering")*.35+rng.normal(0,1.15,n)-3.0
    df["promoted"]=(skor>0).astype(int); return df

def modeli_egit(df:pd.DataFrame)->dict:
    X=df.drop(columns="promoted"); y=df.promoted; sayisal=["experience_years","performance_score","training_hours","projects_completed"]; kategorik=["department","education"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_STATE)
    donusturucu=ColumnTransformer([("kategori",OneHotEncoder(handle_unknown="ignore",sparse_output=False),kategorik)],remainder="passthrough",verbose_feature_names_out=False)
    pipe=Pipeline([("donusturucu",donusturucu),("model",DecisionTreeClassifier(random_state=RANDOM_STATE,class_weight="balanced"))])
    arama=GridSearchCV(pipe,{"model__max_depth":[2,3,4,5,7,None],"model__min_samples_leaf":[5,15,30]},cv=5,scoring="f1",n_jobs=-1).fit(X_train,y_train)
    tahmin=arama.predict(X_test); olasilik=arama.predict_proba(X_test)[:,1]; adlar=arama.best_estimator_.named_steps["donusturucu"].get_feature_names_out(); agac=arama.best_estimator_.named_steps["model"]
    onem=pd.DataFrame({"feature":adlar,"importance":agac.feature_importances_}).sort_values("importance",ascending=False)
    return {"search":arama,"X_test":X_test,"y_test":y_test,"prediction":tahmin,"metrics":{"accuracy":accuracy_score(y_test,tahmin),"f1":f1_score(y_test,tahmin),"roc_auc":roc_auc_score(y_test,olasilik)},"importance":onem,"feature_names":adlar}

def gorselleri_uret(sonuc:dict)->None:
    FIGURES.mkdir(parents=True,exist_ok=True); ConfusionMatrixDisplay.from_predictions(sonuc["y_test"],sonuc["prediction"],cmap="Greens",colorbar=False); plt.title("Terfi Tahmini Confusion Matrix"); plt.tight_layout(); plt.savefig(FIGURES/"confusion_matrix.png",dpi=140); plt.close()
    veri=sonuc["importance"].head(10).sort_values("importance"); plt.figure(figsize=(8,5)); plt.barh(veri.feature,veri.importance,color="#59A14F"); plt.title("Decision Tree Özellik Önemleri"); plt.tight_layout(); plt.savefig(FIGURES/"feature_importance.png",dpi=140); plt.close()
    plt.figure(figsize=(18,8)); plot_tree(sonuc["search"].best_estimator_.named_steps["model"],feature_names=sonuc["feature_names"],class_names=["Hayır","Evet"],filled=True,max_depth=3,fontsize=7); plt.title("Karar Ağacının İlk Dört Seviyesi"); plt.tight_layout(); plt.savefig(FIGURES/"tree_structure.png",dpi=140); plt.close()

def main()->None:
    df=veriyi_uret(); sonuc=modeli_egit(df); gorselleri_uret(sonuc); RESULTS.mkdir(parents=True,exist_ok=True); sonuc["importance"].to_csv(RESULTS/"feature_importance.csv",index=False)
    rapor={"rows":len(df),"promotion_rate":round(float(df.promoted.mean()),4),"best_params":sonuc["search"].best_params_,"metrics":{k:round(float(v),4) for k,v in sonuc["metrics"].items()}}
    with (RESULTS/"metrics.json").open("w",encoding="utf-8") as f: json.dump(rapor,f,ensure_ascii=False,indent=2)
    print("DECISION TREE — ÇALIŞAN TERFİ TAHMİNİ"); print("="*43); print(f"Veri boyutu: {df.shape}"); print(f"Terfi oranı: %{df.promoted.mean()*100:.1f}"); print(f"En iyi parametreler: {sonuc['search'].best_params_}"); print(f"Metrikler: {rapor['metrics']}"); print("\nÖzellik önemleri:\n",sonuc["importance"].head().to_string(index=False)); print(f"\nGrafikler: {FIGURES}")
if __name__=="__main__": main()
