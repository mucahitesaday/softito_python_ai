"""Makine sensörlerinde AdaBoost, Gradient Boosting ve opsiyonel XGBoost."""

from __future__ import annotations
import json,time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42
FEATURES=["temperature","vibration","pressure","rpm","oil_quality","load","age_months","error_count","sensor_9","sensor_10","sensor_11","sensor_12"]
def veriyi_uret(n:int=1800):
    X,y=make_classification(n_samples=n,n_features=12,n_informative=8,n_redundant=2,weights=[.72,.28],class_sep=1.05,flip_y=.035,random_state=RANDOM_STATE); return pd.DataFrame(X,columns=FEATURES),pd.Series(y,name="failure")
def modelleri_egit(X,y)->dict:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_STATE); models={"AdaBoost":AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,random_state=RANDOM_STATE),n_estimators=160,learning_rate=.06,random_state=RANDOM_STATE),"GradientBoosting":GradientBoostingClassifier(n_estimators=160,learning_rate=.06,max_depth=2,random_state=RANDOM_STATE)}; xgb=False
    try:
        from xgboost import XGBClassifier
        models["XGBoost"]=XGBClassifier(n_estimators=160,max_depth=3,learning_rate=.06,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=RANDOM_STATE); xgb=True
    except ImportError: pass
    rows=[]; fitted={}; predictions={}
    for ad,m in models.items(): basla=time.perf_counter(); m.fit(X_train,y_train); fit_s=time.perf_counter()-basla; p=m.predict(X_test); proba=m.predict_proba(X_test)[:,1]; fitted[ad]=m; predictions[ad]=p; rows.append({"model":ad,"accuracy":accuracy_score(y_test,p),"f1":f1_score(y_test,p),"roc_auc":roc_auc_score(y_test,proba),"fit_seconds":fit_s})
    return {"models":fitted,"metrics":pd.DataFrame(rows),"predictions":predictions,"X_test":X_test,"y_test":y_test,"xgboost_available":xgb}
def gorselleri_uret(sonuc)->None:
    FIGURES.mkdir(parents=True,exist_ok=True); sonuc["metrics"].set_index("model")[["accuracy","f1","roc_auc"]].plot(kind="bar",figsize=(8,5),ylim=(0,1)); plt.xticks(rotation=0); plt.title("Boosting Model Performansı"); plt.tight_layout(); plt.savefig(FIGURES/"model_comparison.png",dpi=140); plt.close()
    en_iyi=sonuc["metrics"].loc[sonuc["metrics"].roc_auc.idxmax(),"model"]; ConfusionMatrixDisplay.from_predictions(sonuc["y_test"],sonuc["predictions"][en_iyi],cmap="Reds",colorbar=False); plt.title(f"{en_iyi} Confusion Matrix"); plt.tight_layout(); plt.savefig(FIGURES/"best_confusion_matrix.png",dpi=140); plt.close()
    model=sonuc["models"][en_iyi]; onem=pd.Series(model.feature_importances_,index=FEATURES).sort_values().tail(10); onem.plot(kind="barh",figsize=(8,5),color="#E15759"); plt.title(f"{en_iyi} Özellik Önemleri"); plt.tight_layout(); plt.savefig(FIGURES/"feature_importance.png",dpi=140); plt.close()
def main()->None:
    X,y=veriyi_uret(); sonuc=modelleri_egit(X,y); gorselleri_uret(sonuc); RESULTS.mkdir(parents=True,exist_ok=True); sonuc["metrics"].round(4).to_csv(RESULTS/"model_comparison.csv",index=False); rapor={"rows":len(X),"failure_rate":round(float(y.mean()),4),"xgboost_available":sonuc["xgboost_available"]}
    with (RESULTS/"summary.json").open("w",encoding="utf-8") as f: json.dump(rapor,f,indent=2)
    print("BOOSTING — MAKİNE ARIZA RİSKİ"); print("="*34); print(f"Veri boyutu: {X.shape}"); print(f"Arıza oranı: %{y.mean()*100:.1f}"); print(sonuc["metrics"].round(4).to_string(index=False)); print(f"XGBoost kurulu mu? {sonuc['xgboost_available']}"); print(f"Grafikler: {FIGURES}")
if __name__=="__main__": main()
