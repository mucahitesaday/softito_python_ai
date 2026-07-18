"""RNN, LSTM, attention, Transformer ve autoregressive üretim özeti."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent; RESULTS = ROOT / "results"
def softmax(x, temperature=1.0):
    z = x / temperature; z -= np.max(z, axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def simple_rnn(sequence, hidden_size=6, seed=7):
    rng=np.random.default_rng(seed); wx=rng.normal(0,.3,(1,hidden_size)); wh=rng.normal(0,.3,(hidden_size,hidden_size)); h=np.zeros(hidden_size); states=[]
    for value in sequence: h=np.tanh(value*wx[0]+h@wh); states.append(h.copy())
    return np.asarray(states)

def lstm(sequence, hidden_size=6, seed=8):
    rng=np.random.default_rng(seed); weights=rng.normal(0,.25,(hidden_size+1,4*hidden_size)); h=np.zeros(hidden_size); c=np.zeros(hidden_size); states=[]
    for value in sequence:
        gates=np.r_[value,h]@weights; i,f,o=sigmoid(gates[:hidden_size]),sigmoid(gates[hidden_size:2*hidden_size]),sigmoid(gates[2*hidden_size:3*hidden_size]); candidate=np.tanh(gates[3*hidden_size:]); c=f*c+i*candidate; h=o*np.tanh(c); states.append(h.copy())
    return np.asarray(states)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def scaled_dot_attention(x):
    scores=x@x.T/np.sqrt(x.shape[1]); weights=softmax(scores); return weights@x,weights
def positional_encoding(length, dimension):
    pos=np.arange(length)[:,None]; i=np.arange(dimension)[None,:]; angles=pos/np.power(10000,(2*(i//2))/dimension); encoding=np.zeros_like(angles,dtype=float); encoding[:,0::2]=np.sin(angles[:,0::2]); encoding[:,1::2]=np.cos(angles[:,1::2]); return encoding
def autoregressive_generate(seed_tokens, transition, steps=8, temperature=.8, seed=42):
    rng=np.random.default_rng(seed); tokens=list(seed_tokens)
    for _ in range(steps): tokens.append(int(rng.choice(len(transition),p=softmax(transition[tokens[-1]],temperature))))
    return tokens
def main():
    sequence=np.sin(np.linspace(0,3,12)); rnn=simple_rnn(sequence); lstm_states=lstm(sequence); transformer_input=rnn+positional_encoding(len(rnn),rnn.shape[1]); _,attention=scaled_dot_attention(transformer_input); transition=np.array([[3,1,.2],[.3,3,1],[1,.4,3]])
    result={"rnn_shape":list(rnn.shape),"lstm_shape":list(lstm_states.shape),"attention_row_sum":float(attention[0].sum()),"generated_tokens":autoregressive_generate([0],transition)}
    RESULTS.mkdir(exist_ok=True); (RESULTS/"sequence_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print("SEQUENCE MODELLERİ\n"+"="*30); print(result)
if __name__ == "__main__": main()
