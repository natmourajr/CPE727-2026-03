"""8.5.1–8.5.4: AdaGrad, RMSProp, Adam e escolha de otimizador."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from .comum import semente, luas, salvar, figura_curvas, contornos_caminhos


def _quadratica(steps, init):
    """Compara algoritmos concretos do torch.optim na mesma função convexa."""
    H = torch.tensor([[28., 0.], [0., 1.]], dtype=torch.float64)
    taxa = {'SGD':.035, 'AdaGrad':.35, 'RMSProp':.06, 'Adam':.15}
    historico, perdas, taxas_efetivas = {}, {}, {}
    for nome, lr in taxa.items():
        param = nn.Parameter(torch.tensor(init,dtype=torch.float64))
        optim = {'SGD':torch.optim.SGD,
                 'AdaGrad':torch.optim.Adagrad,
                 'RMSProp':torch.optim.RMSprop,
                 'Adam':torch.optim.Adam}[nome]([param], lr=lr)
        traj, values, eff = [param.detach().numpy().copy()], [], []
        for _ in range(steps):
            optim.zero_grad()
            loss = .5 * param @ H @ param
            loss.backward()
            optim.step()
            traj.append(param.detach().numpy().copy())
            values.append(float((.5 * param @ H @ param).detach()))
            estado = optim.state[param]
            if nome == 'AdaGrad':
                eff.append((lr/(torch.sqrt(estado['sum']) + optim.defaults['eps'])).detach().numpy().copy())
            elif nome == 'RMSProp':
                eff.append((lr/(torch.sqrt(estado['square_avg']) + optim.defaults['eps'])).detach().numpy().copy())
        historico[nome] = np.asarray(traj)
        perdas[nome] = values
        if eff: taxas_efetivas[nome] = np.asarray(eff)
    return H.numpy(), historico, perdas, taxas_efetivas


def _comparar_classificacao(steps, seed):
    xt, yt = luas(n=420,seed=seed)
    xv, yv = luas(n=240,seed=seed+1)
    curvas = {}
    validacao = {}
    configs = [('SGD',torch.optim.SGD,.18),('AdaGrad',torch.optim.Adagrad,.08),
               ('RMSProp',torch.optim.RMSprop,.008),('Adam',torch.optim.Adam,.012)]
    for nome, opt_class, lr in configs:
        semente(seed+91)
        model = nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
        opt = opt_class(model.parameters(),lr=lr)
        rng = np.random.default_rng(seed+7)
        hist=[]
        for _ in range(steps):
            for idx in np.array_split(rng.permutation(len(xt)), 7):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(xt[idx]),yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad(): hist.append(float(nn.functional.cross_entropy(model(xv),yv)))
            model.train()
        curvas[nome] = hist
        validacao[nome] = hist[-1]
    return curvas, validacao


def executar(pasta: Path, rapido=False, seed=42):
    semente(seed)
    H, traj, perdas, taxas = _quadratica(16 if rapido else 95, (2.5, 2.5))
    salvar(contornos_caminhos(traj, H, '8.5 — Trajetórias dos otimizadores na função convexa'),
           pasta, '8_5_01_trajetorias_ada_rmsprop_adam.png')
    salvar(figura_curvas({k:[max(z,1e-12) for z in v] for k,v in perdas.items()},
                         '8.5 — Comparação de perda em função convexa', escala_log=True),
           pasta, '8_5_02_perdas_otimizadores.png')
    fig, ax = plt.subplots(figsize=(8,4.8))
    for nome, dados in taxas.items():
        for c in range(2):
            ax.plot(np.arange(1,len(dados)+1),dados[:,c],
                    label=f'{nome}: coordenada {c+1}', lw=1.8)
    ax.set(title='8.5.1 / 8.5.2 — Taxa de aprendizagem efetiva por parâmetro',
           xlabel='Atualização',ylabel='Taxa efetiva aproximada')
    ax.grid(alpha=.25); ax.legend()
    salvar(fig,pasta,'8_5_03_adagrad_vs_rmsprop_taxas.png')
    curvas, val = _comparar_classificacao(3 if rapido else 24, seed)
    salvar(figura_curvas(curvas,'8.5.4 — Escolha de otimizador: perda de validação',
                         eixo='Cross-entropy de validação'),
           pasta,'8_5_04_selecao_otimizador_validacao.png')
    print('[8.5] Perda final de validação por otimizador:', {k:round(v,4) for k,v in val.items()})
    return {'perda_quadratica':{k:v[-1] for k,v in perdas.items()},'validacao':val}
