"""Digits veri setinde KNN ve Gaussian Naive Bayes karşılaştırması."""

from __future__ import annotations
import json,time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42
def veriyi_yukle():
    d=load_digits(as_frame=True); return d.data,d.target,d.images
def modeli_egit(X,y)->dict:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=RANDOM_STATE); k_sonuclari=[]; modeller={}
    for k in (1,3,5,7,9):
        m=Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=k,weights="distance"))]).fit(X_train,y_train); p=m.predict(X_test); k_sonuclari.append({"k":k,"accuracy":accuracy_score(y_test,p),"f1_macro":f1_score(y_test,p,average="macro")}); modeller[k]=m
    k_df=pd.DataFrame(k_sonuclari); en_iyi=int(k_df.loc[k_df.accuracy.idxmax(),"k"]); knn=modeller[en_iyi]; nb=GaussianNB().fit(X_train,y_train); satirlar=[]; tahminler={}
    for ad,m in ((f"KNN (k={en_iyi})",knn),("GaussianNB",nb)):
        basla=time.perf_counter(); p=m.predict(X_test); sure=(time.perf_counter()-basla)*1000; tahminler[ad]=p; satirlar.append({"model":ad,"accuracy":accuracy_score(y_test,p),"f1_macro":f1_score(y_test,p,average="macro"),"prediction_ms":sure})
    return {"X_test":X_test,"y_test":y_test,"k_results":k_df,"best_k":en_iyi,"comparison":pd.DataFrame(satirlar),"predictions":tahminler}
def gorselleri_uret(sonuc:dict)->None:
    FIGURES.mkdir(parents=True,exist_ok=True); sonuc["k_results"].plot(x="k",y="accuracy",marker="o",legend=False,figsize=(7,4)); plt.ylim(.85,1); plt.title("K Değerine Göre KNN Accuracy"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIGURES/"k_selection.png",dpi=140); plt.close()
    fig,axes=plt.subplots(1,2,figsize=(13,5));
    for ax,(ad,p) in zip(axes,sonuc["predictions"].items()): ConfusionMatrixDisplay.from_predictions(sonuc["y_test"],p,ax=ax,cmap="Blues",colorbar=False); ax.set_title(ad)
    fig.tight_layout(); fig.savefig(FIGURES/"confusion_matrices.png",dpi=140); plt.close(fig)
def main()->None:
    X,y,_=veriyi_yukle(); sonuc=modeli_egit(X,y); gorselleri_uret(sonuc); RESULTS.mkdir(parents=True,exist_ok=True); sonuc["k_results"].round(4).to_csv(RESULTS/"k_results.csv",index=False); sonuc["comparison"].round(4).to_csv(RESULTS/"model_comparison.csv",index=False)
    with (RESULTS/"summary.json").open("w",encoding="utf-8") as f: json.dump({"rows":len(X),"best_k":sonuc["best_k"]},f,indent=2)
    print("KNN vs NAIVE BAYES — RAKAM TANIMA"); print("="*38); print(f"Veri boyutu: {X.shape}"); print(f"En iyi k: {sonuc['best_k']}"); print(sonuc["comparison"].round(4).to_string(index=False)); print(f"Grafikler: {FIGURES}")
if __name__=="__main__": main()
