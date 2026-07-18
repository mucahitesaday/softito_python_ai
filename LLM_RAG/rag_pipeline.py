"""Harici API gerektirmeyen küçük bir RAG ve LLM optimizasyon laboratuvarı."""
from pathlib import Path
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"
DOCUMENTS=["RAG, dil modelinin cevabını güvenilir belgelerle destekler.","LoRA büyük ağırlıkları dondurup düşük rank matrislerini eğitir.","Quantization model ağırlıklarını daha az bit ile saklayarak belleği azaltır.","Vektör veritabanı embedding benzerliğine göre belge arar."]
def preprocess(text): return " ".join(re.findall(r"[a-zçğıöşü0-9]+",text.lower()))
def chunk(text,size=12,overlap=3):
    words=text.split(); step=max(1,size-overlap); return [" ".join(words[i:i+size]) for i in range(0,len(words),step) if words[i:i+size]]
class VectorStore:
    def __init__(self,documents):
        self.documents=documents; self.vectorizer=TfidfVectorizer(preprocessor=preprocess); self.matrix=self.vectorizer.fit_transform(documents)
    def search(self,query,k=2):
        scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix)[0]; order=np.argsort(scores)[::-1][:k]; return [{"text":self.documents[i],"score":float(scores[i])} for i in order]
def lora_parameter_count(input_size,output_size,rank): return rank*(input_size+output_size)
def quantize(values,bits=8):
    qmax=2**(bits-1)-1; scale=np.max(np.abs(values))/qmax or 1.; quantized=np.round(values/scale).astype(np.int8); return quantized,scale
def rag_answer(question,store):
    context=store.search(question,2); return {"question":question,"context":context,"answer":context[0]["text"]}
def main():
    store=VectorStore(DOCUMENTS); weights=np.array([-.8,-.2,.1,.65,1.]); q,scale=quantize(weights); result={"rag":rag_answer("Model belleğini nasıl azaltırım?",store),"fine_tuning_record":{"instruction":"RAG nedir?","response":DOCUMENTS[0]},"lora_parameters":lora_parameter_count(768,768,8),"full_parameters":768*768,"quantized":q.tolist(),"quantization_scale":float(scale)}; RESULTS.mkdir(exist_ok=True); (RESULTS/"rag_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print("LLM VE RAG\n"+"="*25); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
