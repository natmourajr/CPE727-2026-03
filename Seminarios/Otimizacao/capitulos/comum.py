"""Dados e gráficos comuns aos experimentos do seminário."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def semente(valor: int) -> None:
    random.seed(valor)
    np.random.seed(valor)
    torch.manual_seed(valor)


def salvar(fig, pasta: Path, nome: str) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    alvo = pasta / nome
    fig.savefig(alvo, bbox_inches='tight', dpi=170, facecolor='white')
    plt.close(fig)
    print(f'[figura] {alvo}')
    return alvo


def luas(n=480, ruido=0.12, seed=42):
    """Duas luas (sem sklearn ou downloads), classes balanceadas."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    a0 = rng.uniform(0, np.pi, n0)
    a1 = rng.uniform(0, np.pi, n1)
    c0 = np.stack([np.cos(a0), np.sin(a0)], axis=1)
    c1 = np.stack([1-np.cos(a1), 0.5-np.sin(a1)], axis=1)
    x = np.vstack([c0, c1]).astype(np.float32)
    x += rng.normal(0, ruido, size=x.shape).astype(np.float32)
    y = np.r_[np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)]
    perm = rng.permutation(n)
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def figura_curvas(series, titulo, eixo='Perda', escala_log=False):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for rotulo, vals in series.items():
        ax.plot(np.arange(1, len(vals)+1), vals, label=rotulo, linewidth=1.9)
    ax.set(title=titulo, xlabel='Iteração / época', ylabel=eixo)
    if escala_log:
        ax.set_yscale('log')
    ax.grid(alpha=.25)
    ax.legend()
    return fig


def funcao_quadratica_numpy(xs, ys, matriz):
    return 0.5 * (matriz[0,0]*xs**2 + 2*matriz[0,1]*xs*ys + matriz[1,1]*ys**2)


def contornos_caminhos(caminhos, matriz, titulo, limites=(-3.1, 3.1)):
    x = np.linspace(*limites, 175)
    xx, yy = np.meshgrid(x, x)
    zz = funcao_quadratica_numpy(xx, yy, matriz)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.contour(xx, yy, zz, levels=np.geomspace(max(.003,zz.min()+.003), max(.3, zz.max()), 23),
               cmap='Greys', linewidths=.8, alpha=.7)
    for k, (nome, traj) in enumerate(caminhos.items()):
        traj = np.asarray(traj)
        ax.plot(traj[:,0], traj[:,1], '-o', ms=3, lw=1.9,
                label=nome, alpha=.9)
        ax.scatter(traj[0,0], traj[0,1], marker='s', s=48, zorder=4)
    ax.scatter([0], [0], marker='*', s=160, c='black', zorder=5, label='Ótimo')
    ax.set(title=titulo, xlabel=r'$\theta_1$', ylabel=r'$\theta_2$', xlim=limites, ylim=limites)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9)
    ax.grid(alpha=.16)
    return fig


def treinar_classifier(model, x, y, epochs=30, lr=.01, batch=64, seed=42):
    """Treinamento em CPU, com ordem determinística e número fixo de épocas."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        p = rng.permutation(len(x))
        model.train()
        for ids in np.array_split(p, max(1, int(np.ceil(len(p)/batch)))):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            losses.append(float(torch.nn.functional.cross_entropy(model(x), y)))
    return losses
