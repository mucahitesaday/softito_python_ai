"""Doğrusal olmayan iki sınıfta SVM kernel karşılaştırması."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42
def veriyi_uret(n:int=700):
    X,y=make_moons(n_samples=n,noise=.27,random_state=RANDOM_STATE); return pd.DataFrame(X,columns=["sensor_x","sensor_y"]),pd.Series(y,name="class")
def pipe(kernel:str,**kwargs): return Pipeline([("scale",StandardScaler()),("model",SVC(kernel=kernel,probability=True,random_state=RANDOM_STATE,**kwargs))])
def modelleri_egit(X,y)->dict:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_STATE); modeller={"linear":pipe("linear",C=1),"polynomial":pipe("poly",degree=3,C=1),"rbf":pipe("rbf",C=1,gamma="scale")}; rows=[]
    for ad,m in modeller.items(): m.fit(X_train,y_train); p=m.predict(X_test); proba=m.predict_proba(X_test)[:,1]; rows.append({"model":ad,"accuracy":accuracy_score(y_test,p),"f1":f1_score(y_test,p),"roc_auc":roc_auc_score(y_test,proba)})
    arama=GridSearchCV(pipe("rbf"),{"model__C":[.1,1,10,50],"model__gamma":[.1,.5,1,2]},cv=5,scoring="f1",n_jobs=-1).fit(X_train,y_train); p=arama.predict(X_test); proba=arama.predict_proba(X_test)[:,1]; rows.append({"model":"rbf_tuned","accuracy":accuracy_score(y_test,p),"f1":f1_score(y_test,p),"roc_auc":roc_auc_score(y_test,proba)}); modeller["rbf_tuned"]=arama.best_estimator_
    return {"models":modeller,"metrics":pd.DataFrame(rows),"search":arama,"X_test":X_test,"y_test":y_test,"prediction":p}
def gorselleri_uret(X,y,sonuc)->None:
    FIGURES.mkdir(parents=True,exist_ok=True); fig,axes=plt.subplots(1,3,figsize=(16,5)); x1=np.linspace(X.iloc[:,0].min()-.5,X.iloc[:,0].max()+.5,250); x2=np.linspace(X.iloc[:,1].min()-.5,X.iloc[:,1].max()+.5,250); xx,yy=np.meshgrid(x1,x2); grid=pd.DataFrame({"sensor_x":xx.ravel(),"sensor_y":yy.ravel()})
    for ax,ad in zip(axes,["linear","polynomial","rbf_tuned"]): z=sonuc["models"][ad].predict(grid).reshape(xx.shape); ax.contourf(xx,yy,z,alpha=.25,cmap="coolwarm"); ax.scatter(X.sensor_x,X.sensor_y,c=y,cmap="coolwarm",s=12,alpha=.65); ax.set_title(ad)
    fig.suptitle("SVM Kernel Karar Sınırları"); fig.tight_layout(); fig.savefig(FIGURES/"decision_boundaries.png",dpi=140); plt.close(fig)
    ConfusionMatrixDisplay.from_predictions(sonuc["y_test"],sonuc["prediction"],cmap="Oranges",colorbar=False); plt.title("Optimize RBF Confusion Matrix"); plt.tight_layout(); plt.savefig(FIGURES/"confusion_matrix.png",dpi=140); plt.close()
def main()->None:
    X,y=veriyi_uret(); sonuc=modelleri_egit(X,y); gorselleri_uret(X,y,sonuc); RESULTS.mkdir(parents=True,exist_ok=True); sonuc["metrics"].round(4).to_csv(RESULTS/"kernel_comparison.csv",index=False); rapor={"rows":len(X),"best_params":sonuc["search"].best_params_}
    with (RESULTS/"summary.json").open("w",encoding="utf-8") as f: json.dump(rapor,f,indent=2)
    print("SVM KERNEL KARŞILAŞTIRMASI"); print("="*31); print(f"Veri boyutu: {X.shape}"); print(sonuc["metrics"].round(4).to_string(index=False)); print(f"En iyi RBF parametreleri: {sonuc['search'].best_params_}"); print(f"Grafikler: {FIGURES}")
if __name__=="__main__": main()
