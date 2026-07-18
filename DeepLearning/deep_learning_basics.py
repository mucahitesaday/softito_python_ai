"""ANN temelleri: aktivasyonlar, ileri yayılım ve gradient descent."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def relu(x):
    return np.maximum(0, x)

def veri_uret(n=500, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = ((x[:, 0] ** 2 + x[:, 1] > 0.6)).astype(float).reshape(-1, 1)
    return x, y

def egit(x, y, hidden=12, epochs=1200, lr=0.08, seed=42):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.4, (x.shape[1], hidden)); b1 = np.zeros((1, hidden))
    w2 = rng.normal(0, 0.4, (hidden, 1)); b2 = np.zeros((1, 1)); history = []
    for epoch in range(epochs):
        z1 = x @ w1 + b1; h = relu(z1); probability = sigmoid(h @ w2 + b2)
        loss = -np.mean(y*np.log(probability+1e-9)+(1-y)*np.log(1-probability+1e-9))
        output_gradient = (probability-y)/len(x)
        w2_gradient = h.T @ output_gradient; b2_gradient = output_gradient.sum(axis=0, keepdims=True)
        hidden_gradient = (output_gradient @ w2.T) * (z1 > 0)
        w1 -= lr * (x.T @ hidden_gradient); b1 -= lr * hidden_gradient.sum(axis=0, keepdims=True)
        w2 -= lr * w2_gradient; b2 -= lr * b2_gradient
        if epoch % 100 == 0: history.append(float(loss))
    accuracy = float(np.mean((probability >= .5) == y))
    return {"accuracy": accuracy, "final_loss": float(loss), "loss_history": history}

def main():
    x, y = veri_uret(); result = egit(x, y); RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ann_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TEMEL YAPAY SİNİR AĞI\n" + "="*30); print(result)

if __name__ == "__main__": main()
