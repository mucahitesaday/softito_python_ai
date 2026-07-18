"""Sıcaklığa göre enerji talebinde polinom derecesi karşılaştırması."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42

def veriyi_uret(n:int=360)->pd.DataFrame:
    rng=np.random.default_rng(RANDOM_STATE)
    sicaklik=rng.uniform(-5,40,n)
    talep=92+0.42*(sicaklik-19)**2+0.003*(sicaklik-19)**3+rng.normal(0,8,n)
    return pd.DataFrame({"temperature_c":sicaklik,"energy_demand_mwh":talep})

def model_olustur(derece:int)->Pipeline:
    return Pipeline([("polinom",PolynomialFeatures(degree=derece,include_bias=False)),("olcekleme",StandardScaler()),("model",LinearRegression())])

def modelleri_karsilastir(df:pd.DataFrame)->dict:
    X=df[["temperature_c"]]; y=df["energy_demand_mwh"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=RANDOM_STATE)
    satirlar=[]; modeller={}
    for derece in (1,2,3,5,8):
        model=model_olustur(derece).fit(X_train,y_train); modeller[derece]=model
        for bolum,X_b,y_b in (("train",X_train,y_train),("test",X_test,y_test)):
            tahmin=model.predict(X_b)
            satirlar.append({"degree":derece,"split":bolum,"rmse":float(np.sqrt(mean_squared_error(y_b,tahmin))),"r2":float(r2_score(y_b,tahmin))})
    metrikler=pd.DataFrame(satirlar)
    test=metrikler.query("split == 'test'"); en_iyi=int(test.loc[test.rmse.idxmin(),"degree"])
    return {"models":modeller,"metrics":metrikler,"best_degree":en_iyi,"X_test":X_test,"y_test":y_test}

def gorselleri_uret(df:pd.DataFrame,sonuc:dict)->None:
    FIGURES.mkdir(parents=True,exist_ok=True)
    x=np.linspace(df.temperature_c.min(),df.temperature_c.max(),400); model=sonuc["models"][sonuc["best_degree"]]
    plt.figure(figsize=(8,5)); plt.scatter(df.temperature_c,df.energy_demand_mwh,s=15,alpha=.35); plt.plot(x,model.predict(pd.DataFrame({"temperature_c":x})),color="crimson",lw=2); plt.xlabel("Sıcaklık (°C)"); plt.ylabel("Enerji talebi (MWh)"); plt.title(f"En İyi Polinom Modeli — Derece {sonuc['best_degree']}"); plt.tight_layout(); plt.savefig(FIGURES/"best_polynomial_fit.png",dpi=140); plt.close()
    pivot=sonuc["metrics"].pivot(index="degree",columns="split",values="rmse"); pivot.plot(marker="o",figsize=(7,5)); plt.ylabel("RMSE"); plt.title("Polinom Derecesine Göre Hata"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"degree_comparison.png",dpi=140); plt.close()

def main()->None:
    df=veriyi_uret(); sonuc=modelleri_karsilastir(df); gorselleri_uret(df,sonuc); RESULTS.mkdir(parents=True,exist_ok=True)
    sonuc["metrics"].round(4).to_csv(RESULTS/"degree_metrics.csv",index=False)
    with (RESULTS/"summary.json").open("w",encoding="utf-8") as f: json.dump({"rows":len(df),"best_degree":sonuc["best_degree"]},f,indent=2)
    print("POLYNOMIAL REGRESSION — ENERJİ TALEBİ"); print("="*44); print(f"Veri boyutu: {df.shape}"); print(sonuc["metrics"].round(4).to_string(index=False)); print(f"En iyi derece: {sonuc['best_degree']}"); print(f"Grafikler: {FIGURES}"); print(f"Sonuçlar: {RESULTS}")
if __name__=="__main__": main()
