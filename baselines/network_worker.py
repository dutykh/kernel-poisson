# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Multilayer perceptron for the log-rate with a Poisson likelihood (PyTorch), run as

    python -I network_worker.py INPUT.npz OUTPUT.npz

The network g(z) with two hidden layers of `width` tanh units gives the natural parameter
eta = log e + g(z), and the loss is the mean Poisson negative log-likelihood
mean(exp(eta) - y eta). Adam with weight decay minimises it on mini-batches of the training
part of a split (85 per cent of the training fold, the rest for validation); training stops when
the validation loss has not improved for `patience` epochs and the weights of the best epoch are
kept. Every width of `widths` is trained on the same split and the one with the smallest
validation loss is returned. The output bias starts at log(sum y / sum e), the intercept-only fit.

INPUT holds Ztr, ytr, otr (offsets), Zte, ote and the scalars seed, threads, val_fraction,
weight_decay, lr, max_epochs, patience, batch; OUTPUT receives mu_test, mu_train and a JSON
string info.
"""
import json
import sys
import time

import numpy as np
import torch


def build(p, width, bias0):
    net = torch.nn.Sequential(torch.nn.Linear(p, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, 1))
    with torch.no_grad():
        net[-1].bias.fill_(bias0)
    return net


def poisson_nll(eta, y):
    return torch.mean(torch.exp(eta) - y * eta)


def train(Z, y, o, tr, va, width, cfg, generator):
    p = Z.shape[1]
    bias0 = float(np.log(max(y[tr].sum(), 1e-12) / np.exp(o[tr]).sum()))
    net = build(p, width, bias0).double()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    Zt, yt, ot = (torch.as_tensor(a, dtype=torch.float64) for a in (Z, y, o))
    tr_t = torch.as_tensor(tr)
    best, best_state, wait, epochs = np.inf, None, 0, 0
    for epoch in range(cfg["max_epochs"]):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=generator)]
        for start in range(0, len(perm), cfg["batch"]):
            b = perm[start:start + cfg["batch"]]
            opt.zero_grad()
            eta = ot[b] + net(Zt[b]).squeeze(-1)
            loss = poisson_nll(eta, yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(poisson_nll(ot[va] + net(Zt[va]).squeeze(-1), yt[va]))
        epochs = epoch + 1
        if val < best - 1e-10:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    return net, best, epochs


def main(src, dst):
    a = np.load(src, allow_pickle=False)
    cfg = {k: a[k].item() for k in ("seed", "threads", "val_fraction", "weight_decay", "lr",
                                    "max_epochs", "patience", "batch")}
    widths = [int(w) for w in a["widths"]]
    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(int(cfg["seed"]))
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    Ztr, ytr, otr, Zte, ote = (a[k] for k in ("Ztr", "ytr", "otr", "Zte", "ote"))
    n = len(ytr)
    rng = np.random.default_rng(int(cfg["seed"]))
    perm = rng.permutation(n)
    n_val = max(1, int(round(cfg["val_fraction"] * n)))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    cfg["max_epochs"], cfg["patience"], cfg["batch"] = (int(cfg["max_epochs"]), int(cfg["patience"]),
                                                        int(cfg["batch"]))
    t0 = time.time()
    results = []
    for w in widths:
        net, val, epochs = train(Ztr, ytr, otr, tr, va, w, cfg, gen)
        results.append((val, w, epochs, net))
    val, width, epochs, net = min(results, key=lambda r: r[0])
    with torch.no_grad():
        eta_te = torch.as_tensor(ote) + net(torch.as_tensor(Zte)).squeeze(-1)
        eta_tr = torch.as_tensor(otr) + net(torch.as_tensor(Ztr)).squeeze(-1)
    info = dict(width=width, epochs=epochs, val_loss=val,
                val_losses={str(r[1]): r[0] for r in results}, seconds=time.time() - t0)
    np.savez(dst, mu_test=np.exp(np.minimum(eta_te.numpy(), 60.0)),
             mu_train=np.exp(np.minimum(eta_tr.numpy(), 60.0)), info=np.array(json.dumps(info)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
