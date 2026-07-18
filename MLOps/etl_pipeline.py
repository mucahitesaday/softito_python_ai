"""Airflow mantığını sade görev fonksiyonlarıyla gösteren ETL pipeline."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUTPUT=ROOT/"outputs"
def extract(): return pd.DataFrame({"customer_id":[1,2,2,3,4],"amount":[120,85,85,None,240],"status":["paid","paid","paid","paid","refund"]})
def validate(df): return {"rows":len(df),"missing":int(df.isna().sum().sum()),"duplicates":int(df.duplicated().sum())}
def transform(df):
    clean=df.drop_duplicates().copy(); clean["amount"]=clean["amount"].fillna(clean["amount"].median()); clean["net_amount"]=clean["amount"].where(clean["status"]=="paid",-clean["amount"]); return clean
def load(df): OUTPUT.mkdir(exist_ok=True); target=OUTPUT/"daily_sales.csv"; df.to_csv(target,index=False); return target
def run_pipeline():
    raw=extract(); before=validate(raw); clean=transform(raw); target=load(clean); manifest={"pipeline":"daily_sales_etl","tasks":["extract","validate","transform","load"],"quality_before":before,"quality_after":validate(clean),"artifact":str(target)}; (OUTPUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest
if __name__=="__main__": print(run_pipeline())
