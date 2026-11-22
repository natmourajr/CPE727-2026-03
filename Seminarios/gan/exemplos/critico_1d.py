"""D (BCE) contra critico (WGAN-GP) com p_g parada longe de p_data, em 1D: o sinal que chega ao falso.

Mostra o que Arjovsky et al. (2017, figura 2) mostram: treinado ate o fim, o D satura e o sinal
do custo minimax some onde as amostras falsas estao; o critico continua inclinado.
So CPU, menos de 1 minuto. MLP 2 camadas ocultas de 128, Adam 1e-3, 3.000 passos, sementes 0 a 4.

    python critico_1d.py    # grava resultados/critico_1d.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
SAIDA = Path(__file__).parent / "resultados" / "critico_1d.json"


def rede():
    return nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))


def treina(tipo, semente, passos=3000, lote=256):
    torch.manual_seed(semente)
    d, bce = rede(), nn.BCEWithLogitsLoss()
    o = torch.optim.Adam(d.parameters(), 1e-3, betas=(0.5, 0.9))
    for _ in range(passos):
        r, f = -2 + 0.3 * torch.randn(lote, 1), 2 + 0.3 * torch.randn(lote, 1)
        if tipo == "bce":
            sr, sf = d(r), d(f)
            l = bce(sr, torch.ones_like(sr)) + bce(sf, torch.zeros_like(sf))
        else:
            e = torch.rand(lote, 1)
            m = (e * r + (1 - e) * f).requires_grad_(True)
            g = torch.autograd.grad(d(m).sum(), m, create_graph=True)[0]
            l = d(f).mean() - d(r).mean() + 10 * ((g.norm(2, dim=1) - 1) ** 2).mean()
        o.zero_grad(); l.backward(); o.step()
    return d


def sinal(fn, d, x):
    """|d fn(saida) / dx|: o que chega ao gerador pela regra da cadeia, antes de dG/dtheta."""
    x = x.clone().requires_grad_(True)
    return np.abs(torch.autograd.grad(fn(d(x)).sum(), x)[0].numpy().ravel())


def main():
    x = torch.linspace(-4, 4, 401)[:, None]
    falso = ((x > 1.1) & (x < 2.9)).numpy().ravel()
    # logsigmoid evita o arredondamento de log(1 - sigmoid) quando D fica muito perto de 0
    minimax = lambda a: nn.functional.logsigmoid(-a)
    bce, wgan = [treina("bce", s) for s in range(5)], [treina("wgan", s) for s in range(5)]
    res = {"x": x.numpy().ravel().tolist()}
    with torch.no_grad():
        res["D"] = [torch.sigmoid(d(x)).numpy().ravel().tolist() for d in bce]
        res["C"] = [d(x).numpy().ravel().tolist() for d in wgan]
    res["inclinacao_D"] = [float(sinal(torch.sigmoid, d, x)[falso].mean()) for d in bce]
    res["minimax"] = [float(sinal(minimax, d, x)[falso].mean()) for d in bce]
    res["critico"] = [float(sinal(lambda a: a, d, x)[falso].mean()) for d in wgan]
    for k in ("inclinacao_D", "minimax", "critico"):
        print(k, " ".join(f"{v:.3g}" for v in res[k]), "| media %.3g" % np.mean(res[k]))
    # Autoverificacao: o minimax morre e o critico nao; a inclinacao do critico bate com 1 + W/(2 lambda)
    assert np.mean(res["minimax"]) < 0.05 < 0.5 < np.mean(res["critico"])
    assert abs(np.mean(res["critico"]) - 1.2) < 0.05
    SAIDA.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
