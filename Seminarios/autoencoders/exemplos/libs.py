"""Funções auxiliares do notebook de autoencoders (CPE727).

Dados principais: Optical Recognition of Handwritten Digits (UCI), via sklearn.datasets.load_digits.
Imagens 8x8 em tons de cinza (64 atributos), valores originais de 0 a 16.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

DIR_FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# cores usadas nas figuras dos slides
CORES = {"pca": "#C84B4B", "claro": "#9DB3CF", "azul": "#5B8DBE", "verde": "#41AF8B",
         "cinza": "#8C8C8C", "laranja": "#E39B3F"}


def fixar_semente(semente=0):
    np.random.seed(semente)
    torch.manual_seed(semente)


# ---------------------------------------------------------------- dados

def carregar_digits(semente=0):
    """Retorna (X_treino, X_teste, y_treino, y_teste) como tensores float32 em [0, 1]."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=semente, stratify=y)
    paraT = lambda a: torch.tensor(a, dtype=torch.float32)
    return paraT(Xtr), paraT(Xte), torch.tensor(ytr), torch.tensor(yte)


def dados_arco(n=500, ruido=0.04, semente=0):
    """Pontos 2D em torno de um arco de parábola (uma curva de dimensão 1)."""
    g = torch.Generator().manual_seed(semente)
    t = 2 * torch.rand(n, generator=g) - 1
    X = torch.stack([t, 0.6 * (1 - t ** 2)], dim=1)
    return X + ruido * torch.randn(n, 2, generator=g)


def dados_circulo(n=2000, semente=0):
    """Pontos sobre um círculo de raio 0,3 centrado em (0,5; 0,5)."""
    g = torch.Generator().manual_seed(semente)
    t = 2 * np.pi * torch.rand(n, generator=g)
    return torch.stack([0.5 + 0.3 * torch.cos(t), 0.5 + 0.3 * torch.sin(t)], dim=1)


def ruido_gaussiano(x, sigma):
    """Corrupção usada no denoising: soma ruído gaussiano e corta em [0, 1]."""
    return (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)


# ---------------------------------------------------------------- modelo

ATIVACOES = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "linear": nn.Identity}


class Autoencoder(nn.Module):
    """Autoencoder denso e simétrico: h = f(x), r = g(h).

    ocultas:   camadas ocultas do encoder (o decoder usa a ordem inversa).
    ativacao:  'relu', 'tanh', 'sigmoid' ou 'linear'.
    ativ_codigo: ativação aplicada ao código h (padrão: nenhuma).
    saida:     'sigmoide' (dados em [0, 1]) ou 'linear'.
    """

    def __init__(self, n_entrada=64, n_codigo=8, ocultas=(), ativacao="relu",
                 ativ_codigo="linear", saida="sigmoide"):
        super().__init__()
        A = ATIVACOES[ativacao]

        enc, n = [], n_entrada
        for m in ocultas:
            enc += [nn.Linear(n, m), A()]
            n = m
        enc += [nn.Linear(n, n_codigo), ATIVACOES[ativ_codigo]()]
        self.codificador = nn.Sequential(*enc)

        dec, n = [], n_codigo
        for m in reversed(ocultas):
            dec += [nn.Linear(n, m), A()]
            n = m
        dec += [nn.Linear(n, n_entrada)]
        if saida == "sigmoide":
            dec += [nn.Sigmoid()]
        self.decodificador = nn.Sequential(*dec)

    def forward(self, x):
        h = self.codificador(x)
        return self.decodificador(h), h


def jacobiano_codigo(modelo, x):
    """Jacobiano dh/dx (N, n_codigo, n_entrada) de um encoder de UMA camada com sigmoide."""
    lin = modelo.codificador[0]
    h = torch.sigmoid(lin(x))
    return (h * (1 - h)).unsqueeze(2) * lin.weight.unsqueeze(0)


def treinar(modelo, X, epocas=300, lr=1e-3, tam_lote=128, sigma_ruido=0.0, lambda_l1=0.0,
            lambda_contr=0.0, semente=0):
    """Treina minimizando ||x - g(f(x~))||^2 + penalidades.

    sigma_ruido  > 0 -> denoising (a entrada é corrompida, o alvo é o x limpo).
    lambda_l1    > 0 -> esparso (L1 no código h).
    lambda_contr > 0 -> contractive (norma de Frobenius do jacobiano dh/dx; encoder de 1 camada sigmoide).
    """
    fixar_semente(semente)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    mse = nn.MSELoss()
    n = X.shape[0]
    for _ in range(epocas):
        perm = torch.randperm(n)
        for i in range(0, n, tam_lote):
            x = X[perm[i:i + tam_lote]]
            x_in = ruido_gaussiano(x, sigma_ruido) if sigma_ruido > 0 else x
            r, h = modelo(x_in)
            perda = mse(r, x)
            if lambda_l1 > 0:
                perda = perda + lambda_l1 * h.abs().sum(dim=1).mean()
            if lambda_contr > 0:
                perda = perda + lambda_contr * jacobiano_codigo(modelo, x_in).pow(2).sum(dim=(1, 2)).mean()
            opt.zero_grad()
            perda.backward()
            opt.step()
    modelo.eval()
    return modelo


def varias_rodadas(criar_modelo, X, n_rodadas=5, avaliar=None, **kw_treino):
    """Treina n_rodadas modelos (sementes 0, 1, ...) e devolve (modelos, metricas).

    criar_modelo: função sem argumentos que devolve um modelo novo.
    avaliar:      função modelo -> número ou tupla de números (opcional).
    """
    modelos, metricas = [], []
    for s in range(n_rodadas):
        torch.manual_seed(s)
        m = treinar(criar_modelo(), X, semente=s, **kw_treino)
        modelos.append(m)
        if avaliar is not None:
            metricas.append(avaliar(m))
    return modelos, np.array(metricas)


def media_dp(v):
    v = np.asarray(v)
    return v.mean(axis=0), v.std(axis=0, ddof=1) if len(v) > 1 else np.zeros_like(v.mean(axis=0))


@torch.no_grad()
def erro_reconstrucao(modelo, X, X_entrada=None):
    """MSE médio por pixel entre X e a reconstrução de X_entrada (padrão: o próprio X)."""
    r, _ = modelo(X if X_entrada is None else X_entrada)
    return nn.functional.mse_loss(r, X).item()


@torch.no_grad()
def codificar(modelo, X):
    return modelo.codificador(X)


@torch.no_grad()
def reconstruir(modelo, X):
    return modelo(X)[0]


# ---------------------------------------------------------------- figuras

def salvar(fig, nome, pasta="exemplos"):
    """Salva a figura em ../figures/<pasta>/<nome>.png."""
    destino = os.path.join(DIR_FIGURAS, pasta)
    os.makedirs(destino, exist_ok=True)
    fig.savefig(os.path.join(destino, nome + ".png"), dpi=200, bbox_inches="tight")


def limpar_eixos(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mostrar_imagens(linhas, titulos, n=10, tamanho=1.0):
    """Grade de dígitos: cada elemento de `linhas` é um tensor (N, 64)."""
    fig, eixos = plt.subplots(len(linhas), n, figsize=(n * tamanho + 1.5, len(linhas) * tamanho * 1.1))
    eixos = np.atleast_2d(eixos)
    for i, (imgs, titulo) in enumerate(zip(linhas, titulos)):
        for j in range(n):
            ax = eixos[i, j]
            ax.imshow(imgs[j].reshape(8, 8).numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.axis("off")
        eixos[i, 0].text(-0.6, 0.5, titulo, transform=eixos[i, 0].transAxes,
                         ha="right", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def barras_com_erro(ax, nomes, medias, dps, cores, fmt="{:.3f}"):
    """Gráfico de barras com barra de erro (média ± desvio padrão) e o valor em cima."""
    x = np.arange(len(nomes))
    ax.bar(x, medias, color=cores)
    tem_erro = np.asarray(dps) > 0                       # sem barra quando não há variação (ex.: PCA)
    ax.errorbar(x[tem_erro], np.asarray(medias)[tem_erro], yerr=np.asarray(dps)[tem_erro],
                fmt="none", capsize=5, lw=1.2, ecolor="#333333")
    for xi, m, d in zip(x, medias, dps):
        ax.text(xi, m + d, fmt.format(m), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    limpar_eixos(ax)


def dispersao_codigo(ax, H, y, titulo):
    """Espaço latente 2D colorido pelo dígito."""
    H = H.numpy() if torch.is_tensor(H) else H
    sc = ax.scatter(H[:, 0], H[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
    ax.set_title(titulo)
    ax.set_xticks([]); ax.set_yticks([])
    return sc
