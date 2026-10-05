"""2-layer MLP with hand-derived backprop (NumPy), verified against PyTorch autograd.

Forward : z1 = XW1+b1 ; a1 = relu(z1) ; z2 = a1W2+b2 ; p = softmax(z2)
Loss    : L = -mean(log p[y])
Backward: dz2 = (p - onehot(y))/N ; dW2 = a1^T dz2 ; db2 = sum dz2
          da1 = dz2 W2^T ; dz1 = da1 * (z1>0) ; dW1 = X^T dz1 ; db1 = sum dz1
"""
import numpy as np


def xavier(n_in, n_out, rng):
    return rng.normal(0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))


def he(n_in, n_out, rng):
    return rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))


def forward(X, y, P):
    z1 = X @ P["W1"] + P["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ P["W2"] + P["b2"]
    e = np.exp(z2 - z2.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return loss, (X, z1, a1, p)


def backward(y, P, cache):
    X, z1, a1, p = cache
    N = len(y)
    dz2 = p.copy()
    dz2[np.arange(N), y] -= 1
    dz2 /= N
    dW2, db2 = a1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ P["W2"].T) * (z1 > 0)
    return {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": dW2, "b2": db2}


def clip_grads(g, max_norm=1.0):
    norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
    s = min(1.0, max_norm / (norm + 1e-6))
    return {k: v * s for k, v in g.items()}


def train(epochs=200, lr=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like
    P = {"W1": he(2, 16, rng), "b1": np.zeros(16), "W2": xavier(16, 2, rng), "b2": np.zeros(2)}
    for ep in range(epochs):
        loss, cache = forward(X, y, P)
        g = clip_grads(backward(y, P, cache))
        for k in P:
            P[k] -= lr * g[k]
        if ep % 50 == 0:
            print(f"epoch {ep:3d} loss {loss:.4f}")
    return P, X, y


def gradcheck():
    import torch

    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(8, 3)), rng.integers(0, 2, 8)
    P = {"W1": he(3, 4, rng), "b1": np.zeros(4), "W2": xavier(4, 2, rng), "b2": np.zeros(2)}
    _, cache = forward(X, y, P)
    g = backward(y, P, cache)
    T = {k: torch.tensor(v, requires_grad=True) for k, v in P.items()}
    out = torch.relu(torch.tensor(X) @ T["W1"] + T["b1"]) @ T["W2"] + T["b2"]
    torch.nn.functional.cross_entropy(out, torch.tensor(y)).backward()
    for k in P:
        assert np.allclose(g[k], T[k].grad.numpy(), atol=1e-6), k
    print("Manual gradients match PyTorch autograd ✔")


if __name__ == "__main__":
    train()
    try:
        gradcheck()
    except ImportError:
        print("install torch to run gradcheck")
