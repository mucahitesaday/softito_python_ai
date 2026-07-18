"""Genetik Algoritma, CMA-ES ve PSO'nun aynı benchmark üzerinde karşılaştırılması."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KLASOR=Path(__file__).resolve().parent; FIGURES=KLASOR/"figures"; RESULTS=KLASOR/"results"; RANDOM_STATE=42

def rastrigin(x):
    x=np.asarray(x); return 10*x.shape[-1]+np.sum(x*x-10*np.cos(2*np.pi*x),axis=-1)

def genetic_algorithm(dim=5,pop_size=80,generations=160,seed=42):
    rng=np.random.default_rng(seed); pop=rng.uniform(-5.12,5.12,(pop_size,dim)); history=[]
    for _ in range(generations):
        fitness=rastrigin(pop); elite=pop[np.argsort(fitness)[:max(4,pop_size//5)]]; history.append(float(fitness.min()))
        children=[elite[0].copy()]
        while len(children)<pop_size:
            a,b=elite[rng.integers(len(elite),size=2)]; mask=rng.random(dim)<.5; child=np.where(mask,a,b); child+=rng.normal(0,.12,dim)*(rng.random(dim)<.22); children.append(np.clip(child,-5.12,5.12))
        pop=np.asarray(children)
    idx=np.argmin(rastrigin(pop)); return pop[idx],float(rastrigin(pop[idx])),history

def cma_es(dim=5,pop_size=36,generations=160,seed=42):
    rng=np.random.default_rng(seed); mean=rng.uniform(-3,3,dim); sigma=2.; cov=np.eye(dim); mu=pop_size//2; weights=np.log(mu+.5)-np.log(np.arange(1,mu+1)); weights/=weights.sum(); history=[]
    for _ in range(generations):
        samples=rng.multivariate_normal(mean,sigma*sigma*cov,size=pop_size); samples=np.clip(samples,-5.12,5.12); order=np.argsort(rastrigin(samples)); selected=samples[order[:mu]]; old=mean.copy(); mean=np.sum(selected*weights[:,None],axis=0); centered=selected-old; cov=.82*cov+.18*sum(w*np.outer(v,v)/(sigma*sigma+1e-12) for w,v in zip(weights,centered)); sigma=np.clip(sigma*np.exp(.12*(np.mean(rastrigin(selected))<np.mean(rastrigin(samples)))-.04),.03,3.); history.append(float(rastrigin(samples[order[0]])))
    return mean,float(rastrigin(mean)),history

def particle_swarm(dim=5,particles=60,iterations=160,seed=42):
    rng=np.random.default_rng(seed); x=rng.uniform(-5.12,5.12,(particles,dim)); v=rng.uniform(-.5,.5,(particles,dim)); p=x.copy(); pval=rastrigin(p); g=p[np.argmin(pval)].copy(); history=[]
    for t in range(iterations):
        w=.9-.5*t/iterations; v=w*v+1.7*rng.random((particles,dim))*(p-x)+1.7*rng.random((particles,dim))*(g-x); x=np.clip(x+v,-5.12,5.12); vals=rastrigin(x); better=vals<pval; p[better]=x[better]; pval[better]=vals[better]; g=p[np.argmin(pval)].copy(); history.append(float(pval.min()))
    return g,float(rastrigin(g)),history

def karsilastir(seeds=(21,42,84)):
    funcs={"GeneticAlgorithm":genetic_algorithm,"CMA_ES":cma_es,"PSO":particle_swarm}; rows=[]; histories={}
    for name,func in funcs.items():
        for seed in seeds:
            point,value,history=func(seed=seed); rows.append({"algorithm":name,"seed":seed,"best_value":value,"distance_to_zero":float(np.linalg.norm(point))}); histories[f"{name}_{seed}"]=history
    return pd.DataFrame(rows),histories

def gorseller(df,histories):
    FIGURES.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(9,5))
    for name in df.algorithm.unique(): plt.plot(histories[f"{name}_42"],label=name)
    plt.yscale("log"); plt.xlabel("İterasyon"); plt.ylabel("En iyi Rastrigin değeri"); plt.title("Optimizasyon Yakınsama Eğrileri"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES/"convergence_comparison.png",dpi=140); plt.close()
    plt.figure(figsize=(8,5)); df.boxplot(column="best_value",by="algorithm"); plt.yscale("log"); plt.title("Farklı Seed Sonuçları"); plt.suptitle(""); plt.ylabel("En iyi değer"); plt.tight_layout(); plt.savefig(FIGURES/"robustness_comparison.png",dpi=140); plt.close()

def main():
    df,histories=karsilastir(); gorseller(df,histories); RESULTS.mkdir(parents=True,exist_ok=True); df.to_csv(RESULTS/"algorithm_runs.csv",index=False); summary=df.groupby("algorithm").best_value.agg(["mean","std","min","max"]).round(6); summary.to_csv(RESULTS/"algorithm_summary.csv")
    with open(RESULTS/"summary.json","w") as f: json.dump({"benchmark":"Rastrigin 5D","global_minimum":0,"best_algorithm":df.loc[df.best_value.idxmin(),"algorithm"],"best_value":float(df.best_value.min())},f,indent=2)
    print("EVRİMSEL OPTİMİZASYON KARŞILAŞTIRMASI"); print("="*45); print(df.round(6).to_string(index=False)); print("\nÖzet:\n",summary.to_string()); print(f"\nGrafikler: {FIGURES}")

if __name__=="__main__": main()
