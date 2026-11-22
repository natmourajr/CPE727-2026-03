"""8.7.1–8.7.6: estratégias/meta-algoritmos de otimização.

Experimentos independentes, em CPU e com dados sintéticos.
Resultados são ilustrações controladas, não benchmarks científicos.
"""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from .comum import semente, luas, salvar, figura_curvas, contornos_caminhos


def _rede_bn(com_bn=False):
    blocos=[]
    d=2
    for _ in range(5):
        blocos.append(nn.Linear(d,32))
        if com_bn: blocos.append(nn.BatchNorm1d(32))
        blocos.append(nn.ReLU())
        d=32
    blocos.append(nn.Linear(d,2))
    return nn.Sequential(*blocos)


def batch_norm(pasta,rapido,seed):
    xt,yt=luas(400 if rapido else 640,seed=seed)
    xv,yv=luas(200 if rapido else 320,seed=seed+1)
    logs={}
    for com_bn in (False,True):
        semente(seed+15)
        model=_rede_bn(com_bn)
        optim=torch.optim.SGD(model.parameters(),lr=.10)
        perdas=[]
        rng=np.random.default_rng(seed+22)
        for _ in range(4 if rapido else 27):
            model.train()
            for ids in np.array_split(rng.permutation(len(xt)),max(1,len(xt)//64)):
                optim.zero_grad()
                loss=F.cross_entropy(model(xt[ids]),yt[ids]);loss.backward();optim.step()
            model.eval()
            with torch.no_grad():perdas.append(float(F.cross_entropy(model(xv),yv)))
        logs['Com BatchNorm' if com_bn else 'Sem BatchNorm']=perdas
    salvar(figura_curvas(logs,'8.7.1 — BatchNorm: perda de validação',
                         eixo='Cross-entropy de validação'),pasta,
           '8_7_1_batch_normalization.png')
    return {k:v[-1] for k,v in logs.items()}


def coordinate_descent(pasta,rapido):
    H=np.array([[10.,5.],[5.,5.]])
    paths={};losses={}
    for nome in ('Descida por coordenadas','Gradiente (passo fixo)'):
        p=np.array([2.6,2.1])
        vals=[float(.5*p@H@p)];traj=[p.copy()]
        for it in range(8 if rapido else 30):
            g=H@p
            if nome=='Descida por coordenadas':
                i=it%2
                p[i]-=g[i]/H[i,i]  # minimização exata na coordenada
            else:
                p=p-.07*g
            traj.append(p.copy());vals.append(float(.5*p@H@p))
        paths[nome]=np.asarray(traj);losses[nome]=vals
    salvar(contornos_caminhos(paths,H,'8.7.2 — Coordinate Descent vs gradiente'),
           pasta,'8_7_2_coordinate_descent_trajetorias.png')
    salvar(figura_curvas(losses,'8.7.2 — Redução do custo por atualização',
                         eixo='Custo quadrático'),pasta,'8_7_2_coordinate_descent_perdas.png')
    return {k:v[-1] for k,v in losses.items()}


def polyak(pasta,rapido,seed):
    rng=np.random.default_rng(seed+19)
    H=np.diag([4.,1.])
    p=np.array([2.2,1.7]); traj=[]; medias=[]; perdas_sgd=[];perdas_avg=[]
    media=None; n=0
    for it in range(38 if rapido else 260):
        ruido=rng.normal(0,3.1,2)
        p=p-.12*(H@p+ruido)
        # Polyak-Ruppert: média dos últimos 75% dos iterados (burn-in descartado).
        if it >= (9 if rapido else 65):
            if n == 0:
                media=p.copy();n=1
            else:
                n+=1;media += (p-media)/n
        else:
            media=p.copy()
        traj.append(p.copy());medias.append(media.copy())
        perdas_sgd.append(.5*p@H@p)
        perdas_avg.append(.5*media@H@media)
    salvar(figura_curvas({'SGD (parâmetro atual)':perdas_sgd,
                          'Polyak (média dos parâmetros)':perdas_avg},
                         '8.7.3 — Polyak Averaging reduz variância',
                         eixo='Custo verdadeiro (sem ruído)'),pasta,
           '8_7_3_polyak_perdas.png')
    fig,ax=plt.subplots(figsize=(7.2,5.6))
    aa=np.asarray(traj);mm=np.asarray(medias)
    ax.plot(aa[:,0],aa[:,1],'-',lw=1,alpha=.5,label='SGD com ruído')
    ax.plot(mm[:,0],mm[:,1],'-',lw=2,label='Média de Polyak')
    ax.scatter([0],[0],marker='*',s=120,c='black',label='Ótimo')
    ax.set(title='8.7.3 — Trajetórias dos parâmetros',xlabel='θ1',ylabel='θ2')
    ax.grid(alpha=.2);ax.legend()
    salvar(fig,pasta,'8_7_3_polyak_trajetorias.png')
    return {'custo_SGD_final':float(perdas_sgd[-1]),'custo_Polyak_final':float(perdas_avg[-1])}


class Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,32);self.fc2=nn.Linear(32,32);self.out=nn.Linear(32,2)

    def forward(self,x):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


class Deeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,32);self.fc2=nn.Linear(32,32)
        self.extra=nn.Linear(32,32);self.out=nn.Linear(32,2)

    def forward(self,x):
        h=F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.out(F.relu(self.extra(h)))


def _treino_val(model,xt,yt,xv,yv,epochs,seed,lr=.01):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    rng=np.random.default_rng(seed)
    logs=[]
    for _ in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(xt)), max(1,len(xt)//64)):
            opt.zero_grad();loss=F.cross_entropy(model(xt[ids]),yt[ids]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():logs.append(float(F.cross_entropy(model(xv),yv)))
    return logs


def supervised_pretraining(pasta,rapido,seed):
    xt,yt=luas(500,ruido=.14,seed=seed)
    xv,yv=luas(220,ruido=.14,seed=seed+1)
    semente(seed+50)
    base=Shallow()
    pre=_treino_val(base,xt,yt,xv,yv,epochs=4 if rapido else 15,seed=seed+40)
    warm=Deeper()
    with torch.no_grad():
        warm.fc1.load_state_dict(base.fc1.state_dict())
        warm.fc2.load_state_dict(base.fc2.state_dict())
        warm.out.load_state_dict(base.out.state_dict())
        # A entrada é ReLU>=0; camada identidade preserva a função aprendida.
        nn.init.eye_(warm.extra.weight)
        nn.init.zeros_(warm.extra.bias)
    semente(seed+51)
    scratch=Deeper()
    stages=4 if rapido else 15
    log_warm=_treino_val(warm,xt,yt,xv,yv,epochs=stages,seed=seed+44)
    log_scratch=_treino_val(scratch,xt,yt,xv,yv,epochs=stages,seed=seed+44)
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(range(1,len(pre)+1),pre,label='Pré-treinamento raso')
    xa=np.arange(len(pre),len(pre)+stages)
    ax.plot(xa,log_warm,label='Modelo profundo (transferência + fine-tuning)',lw=2)
    ax.plot(xa,log_scratch,label='Modelo profundo (do zero)',lw=2)
    ax.axvline(len(pre),ls='--',c='gray',alpha=.6)
    ax.set(title='8.7.4 — Pré-treinamento supervisionado e expansão da rede',
           xlabel='Etapas de otimização (modelo raso + profundo)',
           ylabel='Perda de validação')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    salvar(fig,pasta,'8_7_4_pretraining_supervisionado.png')
    return {'pretrain_ultimo':pre[-1],'warm_final':log_warm[-1],
            'scratch_final':log_scratch[-1]}


class RegressorProfundo(nn.Module):
    def __init__(self,residual):
        super().__init__()
        self.residual=residual
        self.entrada=nn.Linear(2,32)
        self.blocos=nn.ModuleList([nn.Linear(32,32) for _ in range(7)])
        self.saida=nn.Linear(32,1)

    def forward(self,x):
        x=F.relu(self.entrada(x))
        for layer in self.blocos:
            h=F.relu(layer(x))
            x=x+0.3*h if self.residual else h
        return self.saida(x).flatten()


def design_for_optimization(pasta,rapido,seed):
    semente(seed+55)
    x=torch.randn(560,2)
    y=torch.sin(2.7*x[:,0]) +.45*x[:,1]**2 - .2*x[:,0]*x[:,1]
    logs={}
    norms={}
    for skip in (False,True):
        semente(seed+56)
        model=RegressorProfundo(skip)
        opt=torch.optim.Adam(model.parameters(),lr=.008)
        losses=[]
        for t in range(6 if rapido else 90):
            opt.zero_grad()
            loss=F.mse_loss(model(x),y)
            loss.backward()
            if t==0:
                norms['Residual (skip)' if skip else 'Sem skip']=float(model.entrada.weight.grad.norm())
            opt.step()
            with torch.no_grad():losses.append(float(F.mse_loss(model(x),y)))
        logs['Residual (skip)' if skip else 'Sem skip']=losses
    salvar(figura_curvas(logs,'8.7.5 — Rede profunda: conexões residuais vs convencional',
                         eixo='MSE de treino'),pasta,
           '8_7_5_residual_conexoes_skip.png')
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    ax.bar(list(norms),list(norms.values()))
    ax.set(title='8.7.5 — Norma inicial do gradiente na primeira camada',ylabel='Norma L2')
    ax.grid(axis='y',alpha=.2)
    salvar(fig,pasta,'8_7_5_gradiente_primeira_camada.png')
    return {'perda_final':{k:v[-1] for k,v in logs.items()},'gradiente':norms}


def curriculum_learning(pasta,rapido,seed):
    xt,yt=luas(720,ruido=.055,seed=seed)
    xv,yv=luas(340,ruido=.18,seed=seed+1)
    rng=np.random.default_rng(seed+70)
    dif=rng.permutation(np.arange(len(xt))%3)
    xt=xt + torch.from_numpy((rng.normal(0,1,xt.shape)*
         np.where(dif==0,.025,np.where(dif==1,.13,.32))[:,None]).astype(np.float32))
    logs={}
    epochs=9 if rapido else 48
    for cur in (False,True):
        semente(seed+80)
        model=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2))
        opt=torch.optim.Adam(model.parameters(),lr=.008)
        rng2=np.random.default_rng(seed+88)
        vals=[]
        for ep in range(epochs):
            etapa = min(2,3*ep//epochs)
            elegiveis=np.flatnonzero(dif<=etapa) if cur else np.arange(len(xt))
            # Mesmo número de atualizações e batch size, comparações de amostragem.
            for _ in range(8):
                ids=rng2.choice(elegiveis,size=64,replace=True)
                opt.zero_grad()
                loss=F.cross_entropy(model(xt[ids]),yt[ids]);loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                pred=model(xv).argmax(dim=1)
                vals.append(float((pred==yv).float().mean()))
            model.train()
        logs['Currículo fácil → difícil' if cur else 'Amostragem aleatória (todos)']=vals
    fig=figura_curvas(logs,'8.7.6 — Curriculum learning: acurácia de validação',
                      eixo='Acurácia de validação')
    ax=fig.axes[0];ax.set_ylim(0,1.01)
    for p in (epochs//3,2*epochs//3):ax.axvline(p,ls=':',c='gray',alpha=.6)
    salvar(fig,pasta,'8_7_6_curriculum_learning.png')
    return {k:v[-1] for k,v in logs.items()}


def executar(pasta: Path, rapido=False, seed=42):
    relatorio={}
    planos=[('8.7.1',batch_norm),('8.7.2',coordinate_descent),
            ('8.7.3',polyak),('8.7.4',supervised_pretraining),
            ('8.7.5',design_for_optimization),('8.7.6',curriculum_learning)]
    for nome,funcao in planos:
        print(f'[{nome}] Executando {funcao.__name__}...')
        if nome=='8.7.2':relatorio[nome]=funcao(pasta,rapido)
        else:relatorio[nome]=funcao(pasta,rapido,seed)
    return relatorio
