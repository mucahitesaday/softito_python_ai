"""TF-IDF, dağıtımsal kelime vektörü ve haber sentiment örneği."""
from pathlib import Path
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"
NEWS=[("Şirket güçlü büyüme ve rekor kâr açıkladı",1),("Piyasalar yükselişle günü tamamladı",1),("Yeni yatırım istihdamı artıracak",1),("İhracat beklentilerin üzerine çıktı",1),("Endeks sert düşüşle kapandı",0),("Şirket zarar ve borç artışı açıkladı",0),("Üretimde daralma endişesi büyüyor",0),("Satışlar beklentilerin altında kaldı",0)]*8
def clean(text): return " ".join(re.findall(r"[a-zçğıöşü]+",text.lower()))
def cooccurrence_embeddings(documents,window=2):
    tokens=[clean(d).split() for d in documents]; vocabulary=sorted({w for doc in tokens for w in doc}); index={w:i for i,w in enumerate(vocabulary)}; matrix=np.zeros((len(vocabulary),len(vocabulary)))
    for doc in tokens:
        for i,word in enumerate(doc):
            for context in doc[max(0,i-window):i]+doc[i+1:i+window+1]: matrix[index[word],index[context]]+=1
    u,s,_=np.linalg.svd(matrix,full_matrices=False); return vocabulary,u[:,:min(8,len(vocabulary))]*np.sqrt(s[:min(8,len(vocabulary))])
def main():
    texts=[x[0] for x in NEWS]; labels=np.array([x[1] for x in NEWS]); train_x,test_x,train_y,test_y=train_test_split(texts,labels,test_size=.25,random_state=42,stratify=labels); vectorizer=TfidfVectorizer(preprocessor=clean,ngram_range=(1,2)); x_train=vectorizer.fit_transform(train_x); model=LogisticRegression(random_state=42).fit(x_train,train_y); accuracy=accuracy_score(test_y,model.predict(vectorizer.transform(test_x))); vocabulary,embeddings=cooccurrence_embeddings(texts); result={"document_count":len(texts),"tfidf_features":len(vectorizer.get_feature_names_out()),"embedding_shape":list(embeddings.shape),"vocabulary_sample":vocabulary[:8],"sentiment_accuracy":float(accuracy)}; RESULTS.mkdir(exist_ok=True); (RESULTS/"nlp_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print("NLP ANALİZİ\n"+"="*25); print(result)
if __name__=="__main__": main()
