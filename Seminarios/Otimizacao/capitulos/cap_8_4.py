"""8.4: estratégias de inicialização de parâmetros."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from .comum import semente, luas, salvar, figura_curvas, treinar_classifier


class MLPProfunda(nn.Module):
    def __init__(self, largura=48, camadas=5):
        super().__init__()
        blocos = []
        dim = 2
        for _ in range(camadas):
            blocos += [nn.Linear(dim, largura), nn.ReLU()]
            dim = largura
        blocos += [nn.Linear(dim, 2)]
        self.rede = nn.Sequential(*blocos)

    def forward(self, x):
        return self.rede(x)


def inicializar(modelo, modo):
    for m in modelo.modules():
        if not isinstance(m, nn.Linear):
            continue
        nn.init.zeros_(m.bias)
        if modo == 'zeros':
            nn.init.zeros_(m.weight)
        elif modo == 'normal_002':
            nn.init.normal_(m.weight, mean=0, std=.02)
        elif modo == 'xavier':
            nn.init.xavier_uniform_(m.weight)
        elif modo == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            raise ValueError(modo)


def executar(pasta: Path, rapido=False, seed=42):
    semente(seed)
    x, y = luas(n=320 if rapido else 600, seed=seed)
    nomes = {'zeros':'Todos zero', 'normal_002':'Normal σ=0,02',
             'xavier':'Xavier/Glorot', 'kaiming':'Kaiming/He'}
    hist = {}
    grad = {}
    for i, (chave, nome) in enumerate(nomes.items()):
        semente(seed+10)
        modelo = MLPProfunda(camadas=4, largura=36)
        inicializar(modelo, chave)
        modelo.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(modelo(x), y)
        loss.backward()
        grad[nome] = float(modelo.rede[0].weight.grad.norm().item())
        hist[nome] = treinar_classifier(modelo, x, y, epochs=4 if rapido else 35,
                                       lr=.008, batch=64, seed=seed+77)
    salvar(figura_curvas(hist, '8.4 — Inicialização e convergência da rede'),
           pasta, '8_4_01_inicializacao_convergencia.png')
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(list(grad), list(grad.values()))
    ax.set(title='8.4 — Norma do gradiente na primeira camada, antes do treino',
           ylabel='Norma L2 do gradiente (cross-entropy)')
    ax.tick_params(axis='x', rotation=16)
    ax.grid(axis='y', alpha=.2)
    for b in bars:
        ax.annotate(f'{b.get_height():.2g}', (b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0,3), textcoords='offset points', ha='center', fontsize=9)
    salvar(fig, pasta, '8_4_02_inicializacao_gradientes.png')
    print('[8.4] Gradiente inicial (primeira camada):', {k:round(v,6) for k,v in grad.items()})
    return {'gradiente_inicial':grad,'perda_final':{k:v[-1] for k,v in hist.items()}}
