#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera figuras e animações do capítulo 8.1.1–8.1.3 de Goodfellow.
Modo padrão usa dados sintéticos locais, sem acesso à internet.
"""

# ============================================================================
# AMBIENTE PYTHON / DOCKER
# Sem importar qualquer arquivo .py e sem precisar carregar dados locais.
# Requisitos: PyTorch, NumPy, Matplotlib e Pillow.
# Para usar MNIST ou Fashion-MNIST, use dataset='mnist'/'fashion_mnist',
# download=True; isso precisará de internet somente para obter as imagens.
# ============================================================================
import argparse
import copy
import io
import json
import math
import random
import zipfile
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # Geração de gráficos sem interface gráfica (Docker)
import matplotlib.pyplot as plt
from matplotlib import animation

# Opções de execução: 'shapes' (autossuficiente), 'mnist', 'fashion_mnist',
# 'folder' (pasta inserida no Colab), ou 'npz' (arquivo inserido no Colab).
# O tamanho de imagem, o modelo e os hiperparâmetros podem ser alterados abaixo.
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10,
                     'savefig.facecolor': 'white'})

print('PyTorch', torch.__version__, '| CUDA disponível:', torch.cuda.is_available())

@dataclass
class AppConfig:
    """Edite estes valores, ou configure-os em configs/exemplo.json.

    - dataset='shapes': nenhuma conexão com a internet necessária.
    - Para MNIST/FashionMNIST use class_pair=(0, 1), por exemplo (3, 8).
    - Para 'folder': data_path/train/<classe> e data_path/val/<classe>
      (data_path/test/<classe> é opcional).
    - Para 'npz': X_train, y_train, X_val, y_val e opcional X_test, y_test.
    """
    dataset: str = 'shapes'
    data_path: str = 'dados'
    download: bool = False
    class_pair: tuple[int, int] = (0, 1)
    image_size: int = 20
    seed: int = 42
    n_train: int = 120
    n_val: int = 260
    n_test: int = 260
    image_noise: float = 0.28
    flip_train: float = 0.25  # altera SOMENTE os rótulos do TREINO
    model: str = 'mlp'        # 'mlp' ou 'cnn' nas experiências de ERM/early stopping
    hidden: int = 96
    epochs_erm: int = 110
    epochs_early: int = 180
    epochs_batch: int = 45
    batch_size: int = 20
    patience: int = 19
    lr_erm: float = 0.14
    lr_early: float = 0.15
    device: str = 'cpu'
    outdir: str = 'graficos'

@dataclass
class ImageSplits:
    """Imagens grayscale [N,1,H,W] em [0,1], labels binários [N]."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor | None = None
    y_test: torch.Tensor | None = None

    def __post_init__(self):
        for prefix in ('train', 'val', 'test'):
            x, y = getattr(self, f'x_{prefix}'), getattr(self, f'y_{prefix}')
            if (x is None) != (y is None):
                raise ValueError(f'x_{prefix} e y_{prefix} devem existir juntos.')
            if x is None:
                continue
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
            # Aceita grayscale [N,H,W], [N,1,H,W] e RGB [N,H,W,3] / [N,3,H,W].
            # Converte RGB em grayscale para usar a mesma arquitetura com um canal.
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim == 4 and x.shape[1] not in (1,3,4) and x.shape[-1] in (1,3,4):
                x = x.permute(0,3,1,2)
            if x.ndim == 4 and x.shape[1] in (3,4):
                x = (x[:,0:1]*.299 + x[:,1:2]*.587 + x[:,2:3]*.114)
            if x.ndim != 4 or x.shape[1] != 1:
                raise ValueError(f'x_{prefix}: esperado [N,H,W], [N,C,H,W] ou [N,H,W,C] em cinza/RGB.')
            if x.numel() and x.max() > 1.0:  # imagens externas em escala 0..255
                x = x / 255.0
            if len(x) != len(y):
                raise ValueError(f'Quantidade de imagens/rótulos inconsistente em {prefix}.')
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError('Demonstração binária: rótulos devem ser 0 ou 1.')
            setattr(self, f'x_{prefix}', x.clamp(0, 1).contiguous())
            setattr(self, f'y_{prefix}', y.contiguous())
        if self.x_train.shape[1:] != self.x_val.shape[1:]:
            raise ValueError('Treino e validação devem ter a mesma resolução.')
        if self.x_test is not None and self.x_test.shape[1:] != self.x_train.shape[1:]:
            raise ValueError('Teste deve ter a mesma resolução de treino.')



def set_seed(seed: int):
    """Reprodutibilidade aproximada, independente da disponibilidade de GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def generate_shapes(n: int, size: int, noise: float, rng: np.random.Generator):
    """Cria imagens: classe 0 = barra vertical; classe 1 = barra horizontal.

    Deslocamentos, espessuras, ruído e elementos distratores tornam o problema
    menos trivial. É um EXEMPLO DIDÁTICO, não reconhecimento de dígitos reais.
    """
    if size < 12:
        raise ValueError('Para shapes, image_size deve ser pelo menos 12.')
    y = rng.integers(0, 2, size=n).astype(np.float32)
    x = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        yc, xc = size // 2 + rng.integers(-2, 3, size=2)
        half = rng.integers(max(3, size//4), max(4, size//2))
        thick = rng.integers(1, 4)
        if y[i] == 0:
            x[i, max(0,yc-half):min(size,yc+half+1), max(0,xc-thick//2):min(size,xc+(thick+1)//2)] = rng.uniform(.70, 1.)
        else:
            x[i, max(0,yc-thick//2):min(size,yc+(thick+1)//2), max(0,xc-half):min(size,xc+half+1)] = rng.uniform(.70, 1.)
        # Linha/distrator aleatório leve: ajuda a visualizar incerteza.
        if rng.random() < 0.34:
            yy, xx = rng.integers(1,size-1,size=2)
            x[i, yy, max(0,xx-2):min(size,xx+3)] += rng.uniform(.15,.45)
    x += rng.normal(0, noise, x.shape).astype(np.float32)
    return np.clip(x, 0, 1), y



def make_shapes(cfg: AppConfig) -> ImageSplits:
    """Cada split usa amostras independentes de uma mesma distribuição geradora."""
    rng = np.random.default_rng(cfg.seed)
    xt, yt = generate_shapes(cfg.n_train, cfg.image_size, cfg.image_noise, rng)
    xv, yv = generate_shapes(cfg.n_val, cfg.image_size, cfg.image_noise, rng)
    xe, ye = generate_shapes(cfg.n_test, cfg.image_size, cfg.image_noise, rng)
    count = round(len(yt) * cfg.flip_train)
    if count:
        idx = rng.choice(len(yt), count, replace=False)
        yt[idx] = 1 - yt[idx]
    return ImageSplits(xt, yt, xv, yv, xe, ye)



def _from_torchvision(cfg: AppConfig) -> ImageSplits:
    """MNIST ou FashionMNIST, binário. Baixa só com download=True."""
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise RuntimeError('Instale torchvision para MNIST/FashionMNIST.') from e
    klass = {'mnist': datasets.MNIST,
             'fashion_mnist': datasets.FashionMNIST}[cfg.dataset]
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()])
    train = klass(cfg.data_path, train=True, download=cfg.download, transform=transform)
    test = klass(cfg.data_path, train=False, download=cfg.download, transform=transform)
    a, b = cfg.class_pair
    if a == b or not (0 <= a <= 9 and 0 <= b <= 9):
        raise ValueError('class_pair deve ter dois dígitos distintos entre 0 e 9.')
    def subset(ds, take, seed):
        targets = np.asarray(ds.targets)
        possible = np.flatnonzero((targets == a) | (targets == b))
        rng = np.random.default_rng(seed)
        rng.shuffle(possible)
        chosen = possible[:take]
        xx = torch.stack([ds[int(i)][0] for i in chosen])
        yy = torch.tensor([float(targets[i] == b) for i in chosen])
        return xx, yy
    # Escolhas disjuntas na base oficial de treino; teste oficial separado.
    labels = np.asarray(train.targets)
    possible = np.flatnonzero((labels == a) | (labels == b))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(possible)
    needed = cfg.n_train + cfg.n_val
    if len(possible) < needed:
        raise ValueError(f'Treino MNIST/Fashion insuficiente: {len(possible)} < {needed}.')
    def gather(ds, indices):
        return (torch.stack([ds[int(i)][0] for i in indices]),
                torch.tensor([float(int(ds.targets[int(i)]) == b) for i in indices]))
    xt, yt = gather(train, possible[:cfg.n_train])
    xv, yv = gather(train, possible[cfg.n_train:needed])
    xe, ye = subset(test, cfg.n_test, cfg.seed + 1)
    if cfg.flip_train:
        r = np.random.default_rng(cfg.seed + 2)
        indices = r.choice(len(yt), round(len(yt)*cfg.flip_train), replace=False)
        yt[indices] = 1 - yt[indices]
    return ImageSplits(xt, yt, xv, yv, xe, ye)



def _from_folder(cfg: AppConfig) -> ImageSplits:
    """Subpastas train/{nomeA,nomeB}, val/{...}, test/{...} opcional."""
    from torchvision import datasets, transforms
    root = Path(cfg.data_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()])
    datasets_found = {}
    for name in ('train', 'val', 'test'):
        folder = root/name
        if folder.is_dir():
            ds = datasets.ImageFolder(folder, transform=transform)
            if len(ds.classes) != 2:
                raise ValueError(f'{folder}: forneça exatamente 2 subpastas de classes.')
            datasets_found[name] = ds
    if not {'train', 'val'}.issubset(datasets_found):
        raise FileNotFoundError('Esperados dados/train/<classe> e dados/val/<classe>.')
    cls = datasets_found['train'].classes
    for name, ds in datasets_found.items():
        if ds.classes != cls:
            raise ValueError(f'Classes de {name} diferentes de train: {ds.classes} != {cls}.')
    def tensorize(ds):
        return (torch.stack([ds[i][0] for i in range(len(ds))]),
                torch.tensor([float(ds[i][1]) for i in range(len(ds))]))
    xt, yt = tensorize(datasets_found['train'])
    xv, yv = tensorize(datasets_found['val'])
    xe, ye = tensorize(datasets_found['test']) if 'test' in datasets_found else (None,None)
    return ImageSplits(xt, yt, xv, yv, xe, ye)



def _from_npz(cfg: AppConfig) -> ImageSplits:
    with np.load(cfg.data_path, allow_pickle=False) as v:
        names = ('x_train','y_train','x_val','y_val','x_test','y_test')
        values = {k: v[k] if k in v else None for k in names}
        # Aceita convenções uppercase: X_train, y_train.
        for name in ('train','val','test'):
            if values['x_'+name] is None and 'X_'+name in v:
                values['x_'+name] = v['X_'+name]
    return ImageSplits(**values)



def load_images(cfg: AppConfig) -> ImageSplits:
    """Único ponto para trocar o conjunto de imagens sem alterar experimentos."""
    set_seed(cfg.seed)
    if cfg.dataset == 'shapes':
        return make_shapes(cfg)
    if cfg.dataset in ('mnist', 'fashion_mnist'):
        return _from_torchvision(cfg)
    if cfg.dataset == 'folder':
        return _from_folder(cfg)
    if cfg.dataset == 'npz':
        return _from_npz(cfg)
    raise ValueError(f'Dataset desconhecido: {cfg.dataset}')



class PairProjector:
    """Resume cada imagem em dois atributos interpretáveis para gráfico 2D.

    f1 = intensidade na faixa vertical central; f2 = na faixa horizontal.
    O ajuste do centramento/escala ocorre SOMENTE com X_train.
    Para outros tipos de imagem, substitua extract() mantendo saída [N,2].
    Por exemplo MNIST pode precisar de outro extrator para ter acurácia boa:
    o projeto ilustra otimização, NÃO busca SOTA de visão computacional.
    """
    def __init__(self):
        self.mean, self.std = None, None

    def extract(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        h, w = x.shape[-2:]
        sw = max(1, w//7)
        sh = max(1, h//7)
        vertical = x[:, :, :, max(0,w//2-sw):min(w,w//2+sw+1)].mean((1,2,3))
        horizontal = x[:, :, max(0,h//2-sh):min(h,h//2+sh+1), :].mean((1,2,3))
        return torch.stack((vertical, horizontal), dim=-1)

    def fit(self, x):
        f = self.extract(x)
        self.mean = f.mean(0)
        self.std = f.std(0, unbiased=False).clamp(min=1e-3)
        return self

    def transform(self, x):
        if self.mean is None:
            raise ValueError('Chame fit(X_train) antes de transform().')
        return (self.extract(x)-self.mean)/self.std



def as_projected(splits: ImageSplits):
    """Converte imagens para atributos 2D SEM usar estatísticas de validação/teste."""
    proj = PairProjector().fit(splits.x_train)
    return proj, {
        'train': (proj.transform(splits.x_train), splits.y_train),
        'val': (proj.transform(splits.x_val), splits.y_val),
        'test': ((proj.transform(splits.x_test), splits.y_test)
                 if splits.x_test is not None else None)
    }


def build_model(kind: str, image_size: int=20, hidden: int=96, image_shape: tuple[int,int] | None=None):
    """Troque esta função para experimentar outros modelos PyTorch.

    linear2d: recebe [N,2] e possui APENAS 2 pesos; permite loss contour.
    mlp:      recebe imagens [N,1,H,W], sobreajusta com facilidade.
    cnn:      recebe imagens e aprende filtros convolucionais; tamanho flexível.
    """
    if kind == 'linear2d':
        return nn.Linear(2, 1, bias=False)
    if kind == 'mlp':
        h, w = image_shape if image_shape is not None else (image_size,image_size)
        return nn.Sequential(nn.Flatten(), nn.Linear(h*w, hidden),
                             nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                             nn.Linear(hidden, 1))
    if kind == 'cnn':
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                             nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
                             nn.Linear(16*4*4, hidden), nn.ReLU(), nn.Linear(hidden,1))
    raise ValueError(f'Modelo desconhecido: {kind}')



@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device='cpu'):
    """Avalia o modelo nos dados FORNECIDOS; nunca atualiza seus parâmetros."""
    model.eval()
    xx, yy = x.to(device), y.to(device)
    logits = model(xx).flatten()
    loss = F.binary_cross_entropy_with_logits(logits, yy).item()
    accuracy = ((logits >= 0).float() == yy).float().mean().item()
    return {'loss': loss, 'acc': accuracy}



def fit(model: nn.Module, train: tuple, val: tuple, *, epochs=100, lr=.1,
        batch_size=32, patience: int | None=None, seed=42, device='cpu',
        record_updates=False, max_record=6000):
    """Treino genérico PyTorch + validação + early stopping opcional.

    - Cada mini-batch calcula gradients por loss.backward(), atualiza via SGD.
    - Salva métricas A CADA ÉPOCA medindo TODOS os exemplos de treino/val.
    - Para 2 parâmetros, registra trajetória de pesos a cada UPDATE (para GIF).
    - patience=None => treino completo, mesmo após iniciar overfitting.
    - patience=int => interrompe e RESTAURA melhor checkpoint de validação.
    - Datasets NÃO podem ser usados para retreinar após avaliar no teste.
    """
    x_train,y_train = train
    x_val,y_val = val
    if batch_size < 1 or epochs < 1 or lr <= 0:
        raise ValueError('batch_size/epochs devem ser positivos; lr > 0.')
    set_seed(seed)
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {'epoch': [], 'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'grad_norm': [],
               'updates_total': [], 'samples_seen': [], 'theta_epoch': [],
               'theta_updates': [], 'update_examples': []}
    two_params = sum(p.numel() for p in model.parameters()) == 2
    if two_params:
        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()
        history['theta_updates'].append(theta0.copy())
        history['update_examples'].append(0)
    best_val, best_epoch, bad_epochs = float('inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    updates, processed = 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        norms = []
        for xb,yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).flatten()
            loss = loss_fn(logits, yb)
            loss.backward()  # autograd calcula gradiente da perda substituta
            gnorm = math.sqrt(sum((p.grad.detach()**2).sum().item()
                                  for p in model.parameters() if p.grad is not None))
            norms.append(gnorm)
            optimizer.step()  # SGD: theta <- theta - learning_rate * gradiente
            updates += 1
            processed += len(yb)
            if two_params and record_updates and updates <= max_record:
                theta = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy()
                history['theta_updates'].append(theta.copy())
                history['update_examples'].append(processed)
        tr = evaluate(model, x_train, y_train, device=device)
        va = evaluate(model, x_val, y_val, device=device)
        for key, v in [('epoch',ep),('train_loss',tr['loss']),('val_loss',va['loss']),
                       ('train_acc',tr['acc']),('val_acc',va['acc']),
                       ('grad_norm',float(np.mean(norms))),('updates_total',updates),
                       ('samples_seen',processed)]:
            history[key].append(v)
        if two_params:
            history['theta_epoch'].append(torch.cat([
                p.detach().flatten() for p in model.parameters()]).cpu().numpy().copy())
        # Checkpoint definido APENAS pelo conjunto de validação.
        if va['loss'] < best_val - 1e-6:
            best_val, best_epoch, bad_epochs = va['loss'], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if patience is not None and bad_epochs >= patience:
            break
    history['best_epoch'] = best_epoch
    history['stopped_epoch'] = history['epoch'][-1]
    history['best_val_loss'] = best_val
    history['early_stopping'] = patience is not None
    if patience is not None:
        model.load_state_dict(best_state)  # devolve pesos da melhor época
    return model, history



def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path



def plot_image_gallery(splits: ImageSplits, n=12):
    """Mostra os rótulos de treino (que podem ter flips artificiais)."""
    n = min(n, len(splits.y_train))
    fig, ax = plt.subplots(2, math.ceil(n/2), figsize=(13,4.7))
    ax = np.asarray(ax).reshape(-1)
    for i, a in enumerate(ax):
        a.axis('off')
        if i < n:
            a.imshow(splits.x_train[i,0].numpy(), cmap='gray', vmin=0,vmax=1)
            a.set_title(f'Imagem {i+1} | rótulo {int(splits.y_train[i])}', fontsize=9)
    fig.suptitle('Entrada visual do experimento (rótulo 0/1)', fontweight='bold')
    fig.tight_layout()
    return fig



def loss_surface(features: torch.Tensor, labels: torch.Tensor, theta_history,
                 ngrid=85, padding=1.3):
    """Calcula superfície real da BCE do classificador linear com 2 pesos."""
    path = np.asarray(theta_history, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError('A trajetória deve ter exatamente dois parâmetros.')
    lo = np.minimum(path.min(0)-padding, np.array([-2.,-2.]))
    hi = np.maximum(path.max(0)+padding, np.array([2.,2.]))
    xcoords = torch.linspace(float(lo[0]),float(hi[0]),ngrid)
    ycoords = torch.linspace(float(lo[1]),float(hi[1]),ngrid)
    gx,gy = torch.meshgrid(xcoords,ycoords,indexing='xy')
    theta = torch.stack([gx.reshape(-1), gy.reshape(-1)],1)
    x = features.float().cpu()
    y = labels.float().cpu()
    # Divide em blocos: evita matrizes gigantes em datasets grandes.
    parts=[]
    with torch.no_grad():
        for chunk in theta.split(1500):
            logits = chunk @ x.T
            cost = F.softplus(logits) - logits*y.unsqueeze(0)
            parts.append(cost.mean(1))
    grid = torch.cat(parts).reshape(ngrid,ngrid).numpy()
    return gx.numpy(),gy.numpy(),grid



def plot_descent(data, history, title='8.1.1 | Descida de gradiente em duas dimensões'):
    x,y = data
    trace = np.vstack((history['theta_updates'] if len(history['theta_updates'])>1
                       else history['theta_epoch']))
    gx,gy,z = loss_surface(x,y,trace)
    fig, ax = plt.subplots(figsize=(8.1,5.5))
    lev = np.linspace(float(z.min()),float(np.quantile(z,.87)),21)
    ax.contourf(gx,gy,z,levels=lev,extend='max',cmap='Blues',alpha=.85)
    ax.contour(gx,gy,z,levels=lev[::3],colors='gray',alpha=.4,linewidths=.6)
    ax.plot(trace[:,0],trace[:,1],'-o',ms=2.7,lw=1.6,color='#ee6c4d',alpha=.9)
    ax.scatter(*trace[0],c='black',s=85,marker='o',label='início',zorder=5)
    ax.scatter(*trace[-1],c='#e63946',s=105,marker='*',label='fim',zorder=5)
    ax.set(xlabel='Peso w₁',ylabel='Peso w₂',title=title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig



def animate_descent(data, history, *, frames=55, interval=140, save_gif=None):
    """Retorna FuncAnimation para ver CADA PASSO na interface do Jupyter.

    Para salvar GIF: save_gif='...gif'.
    O movimento é dos dois pesos sobre a superfície da BCE, NÃO é o movimento
    de pixels ou da saída de uma rede profunda em espaço de alta dimensão.
    """
    trace = np.asarray(history['theta_updates'] if len(history['theta_updates'])>1
                       else history['theta_epoch'])
    gx,gy,z = loss_surface(*data,trace,ngrid=75)
    fig,ax = plt.subplots(figsize=(7.3,5.0))
    lev=np.linspace(float(z.min()),float(np.quantile(z,.87)),18)
    ax.contourf(gx,gy,z,levels=lev,extend='max',cmap='Blues',alpha=.87)
    ax.contour(gx,gy,z,levels=lev[::3],colors='gray',linewidths=.6,alpha=.5)
    line,=ax.plot([],[],'-o',color='#e63946',ms=2,lw=2)
    dot,=ax.plot([],[],'o',color='#c1121f',ms=10)
    marker=ax.text(.02,.96,'',transform=ax.transAxes,va='top',fontsize=11,
                   bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
    ax.set(xlabel='Peso w₁',ylabel='Peso w₂',title='Gradiente: trajetória dos pesos sobre a perda')
    indices=np.unique(np.linspace(0,len(trace)-1,min(frames,len(trace)),dtype=int))
    def draw(k):
        i=indices[k]
        line.set_data(trace[:i+1,0],trace[:i+1,1])
        dot.set_data([trace[i,0]],[trace[i,1]])
        marker.set_text(f'Atualização {i} de {len(trace)-1}')
        return line,dot,marker
    anim=animation.FuncAnimation(fig,draw,frames=len(indices),interval=interval,
                                 blit=True,repeat=True)
    if save_gif:
        Path(save_gif).parent.mkdir(parents=True,exist_ok=True)
        anim.save(save_gif,writer=animation.PillowWriter(fps=max(1,int(1000/interval))))
    plt.close(fig)
    return anim



def plot_erm(history, title='8.1.1 | Risco empírico versus generalização'):
    fig, (a,b)=plt.subplots(1,2,figsize=(13.2,4.3))
    ep=history['epoch']
    a.plot(ep,history['train_loss'],label='Treino (risco empírico)',lw=2)
    a.plot(ep,history['val_loss'],label='Validação (aprox. generalização)',lw=2)
    a.set(xlabel='Época',ylabel='Binary cross-entropy',title='Perda nos dois conjuntos')
    a.legend(); a.grid(alpha=.25)
    b.plot(ep,history['train_acc'],label='Treino')
    b.plot(ep,history['val_acc'],label='Validação')
    b.set(xlabel='Época',ylabel='Acurácia',title='Acurácia da classificação')
    b.set_ylim(0,1.04); b.grid(alpha=.25); b.legend()
    fig.suptitle(title,fontweight='bold'); fig.tight_layout()
    return fig



def plot_surrogates():
    """Comparação em função da MARGEM z=y_assinado*logit.

    0-1: salto, derivada nula quase em todo lugar -> difícil otimizar.
    Logística: softplus(-z), diferenciável; hinge: max(0,1-z), subgradiente.
    """
    m=np.linspace(-4,4,700)
    zero_one=(m<=0).astype(float)
    bce=np.logaddexp(0,-m)
    hinge=np.maximum(0,1-m)
    fig,ax=plt.subplots(figsize=(9,5.1))
    ax.step(m,zero_one,where='post',label='0–1 (não diferenciável)',lw=2.3)
    ax.plot(m,bce,label='Logística / BCE (PyTorch)',lw=2.3)
    ax.plot(m,hinge,label='Hinge (SVM)',lw=2,ls='--')
    ax.axvline(0,ls=':',c='gray',lw=1)
    ax.set(xlabel='Margem correta: (2y−1) × logit',ylabel='Perda por observação',
           title='8.1.2 | Funções substitutas versus perda 0–1',ylim=(-.1,4.2))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();return fig



def plot_early_stopping(history):
    ep=history['epoch']
    fig,(a,b)=plt.subplots(1,2,figsize=(13,4.4))
    a.plot(ep,history['train_loss'],label='Treino',lw=2)
    a.plot(ep,history['val_loss'],label='Validação',lw=2)
    a.axvline(history['best_epoch'],color='crimson',ls='--',lw=2,
              label=f'Melhor época: {history["best_epoch"]}')
    a.axvline(history['stopped_epoch'],color='gray',ls=':',
              label=f'Parada: {history["stopped_epoch"]}')
    a.set(xlabel='Época',ylabel='BCE',title='Treino vs. validação');a.grid(alpha=.22);a.legend()
    b.plot(ep,history['train_acc'],label='Treino')
    b.plot(ep,history['val_acc'],label='Validação')
    b.axvline(history['best_epoch'],color='crimson',ls='--',lw=2)
    b.set(xlabel='Época',ylabel='Acurácia',title='Generalização',ylim=(0,1.03))
    b.grid(alpha=.22);b.legend()
    fig.suptitle('8.1.2 | Early stopping: monitorar validação, restaurar melhor época',
                 fontweight='bold');fig.tight_layout();return fig


def compare_batch(data2d, cfg: AppConfig):
    """Três variantes com MESMO início/mesmos exemplos e épocas fixas.

    OBS.: mesma quantidade de épocas != mesmo número de atualizações;
    o plot de exemplos processados ajuda, mas custo computacional difere.
    """
    train,val=data2d['train'],data2d['val']
    methods = {
        'Batch completo': (len(train[1]), .38),
        'Mini-batch (16)': (min(16,len(train[1])), .085),
        'SGD (1)': (1, .018),
    }
    set_seed(cfg.seed+77)
    init=build_model('linear2d')
    # Partimos de um ponto deliberadamente distante do ótimo para enxergar as trajetórias.
    # Para mudar o cenário, altere os dois números abaixo.
    with torch.no_grad():
        init.weight.copy_(torch.tensor([[3.0, -3.0]]))
    state=copy.deepcopy(init.state_dict())
    results={}
    for name,(bs,lr) in methods.items():
        model=build_model('linear2d')
        model.load_state_dict(state)
        _,hist=fit(model,train,val,epochs=cfg.epochs_batch,lr=lr,batch_size=bs,
                   seed=cfg.seed+100,patience=None,record_updates=True,
                   max_record=9000,device=cfg.device)
        results[name]=hist
    return results



def plot_batch_comparison(results,data2d):
    fig,axes=plt.subplots(2,2,figsize=(13.3,9.3))
    a,b,c,d=axes.flat
    colors=['#275dad','#f39c12','#c23b22']
    for (name,h),color in zip(results.items(),colors):
        a.plot(h['epoch'],h['train_loss'],label=name,lw=2,c=color)
        b.plot(h['samples_seen'],h['train_loss'],label=name,lw=2,c=color)
        c.plot(h['epoch'],h['grad_norm'],label=name,lw=1.8,c=color)
    a.set(title='Mesmas épocas (atualizações diferentes)',xlabel='Época',ylabel='BCE de treino')
    b.set(title='Orçamento: exemplos processados',xlabel='Imagens processadas',ylabel='BCE de treino')
    c.set(title='Norma média dos gradientes por época',xlabel='Época',ylabel='||gradiente||')
    traces=[np.asarray(h['theta_updates']) for h in results.values()]
    lo=np.vstack(traces)
    gx,gy,z=loss_surface(*data2d['train'],lo,ngrid=80)
    lev=np.linspace(float(z.min()),float(np.quantile(z,.82)),17)
    d.contourf(gx,gy,z,levels=lev,extend='max',cmap='Blues',alpha=.7)
    for (name,h),color in zip(results.items(),colors):
        trace=np.asarray(h['theta_updates'])
        take=np.unique(np.linspace(0,len(trace)-1,min(230,len(trace)),dtype=int))
        d.plot(trace[take,0],trace[take,1],alpha=.8,c=color,lw=1.6,label=name)
        d.scatter(*trace[-1],c=color,s=45)
    d.set(xlabel='w₁',ylabel='w₂',title='Trajetória de pesos: passos mais/menos ruidosos')
    for ax in (a,b,c,d): ax.grid(alpha=.14);ax.legend(fontsize=9)
    fig.suptitle('8.1.3 | Batch GD, Mini-batch SGD e SGD (PyTorch)',fontweight='bold')
    fig.tight_layout();return fig



def gradient_variance(data2d, *, sizes=(1,4,16,32,64,128), repetitions=70, seed=42):
    """Estima E[||g_B-g_completo||²], sorteio SEM reposição por tentativa.

    Importante: quando B=N, g_B == g_completo até erro de arredondamento.
    A variância está no GRADIENTE (não na distribuição dos rótulos).
    """
    x,y=data2d['train']
    n=len(y)
    set_seed(seed)
    w = torch.tensor([[-.4,.4]],requires_grad=True)
    def get_grad(xx,yy):
        loss=F.binary_cross_entropy_with_logits((xx@w.T).flatten(),yy)
        return torch.autograd.grad(loss,w)[0].detach().flatten()
    full=get_grad(x,y)
    sizes=sorted(set(min(int(b),n) for b in sizes if int(b)>0)|{n})
    generator=torch.Generator().manual_seed(seed)
    var=[]
    for b in sizes:
        errors=[]
        for _ in range(repetitions):
            idx=torch.randperm(n,generator=generator)[:b]
            g=get_grad(x[idx],y[idx]);errors.append(((g-full)**2).sum().item())
        var.append(float(np.mean(errors)))
    return np.asarray(sizes),np.asarray(var)



def plot_gradient_variance(sizes,var):
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    ax.plot(sizes,var,'-o',c='#157b81',lw=2.2)
    ax.set(xlabel='Tamanho B do lote (sem reposição)',
           ylabel='E[||g_B − g_N||²] (estimado)',
           title='8.1.3 | Gradientes menores têm maior variância')
    ax.grid(alpha=.23);fig.tight_layout();return fig


# ============================================================================
# GRÁFICOS LIMPOS PARA OS SLIDES (separados dos gráficos analíticos do notebook)
# Os gráficos mostram valores calculados, não curvas simuladas.
# ============================================================================
def _format_slide(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.17)
    ax.tick_params(axis='both', labelsize=12)


def slide_erm(h):
    """8.1.1: risco empírico versus perda de validação (estimativa fora da amostra).
    Validação NÃO é igual ao risco populacional verdadeiro.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(h['epoch'], h['train_loss'], lw=3, color='#1769aa', label='Treino: risco empírico')
    ax.plot(h['epoch'], h['val_loss'], lw=3, color='#db6b24', label='Validação: estimativa fora da amostra')
    ax.set(xlabel='Época', ylabel='Perda média (BCE)')
    ax.legend(frameon=False, loc='center right')
    _format_slide(ax)
    fig.tight_layout()
    return fig


def slide_surrogates():
    """8.1.2: contraste de erro de classificação e BCE diferenciável."""
    z=np.linspace(-4,4,700)
    loss01=(z<=0).astype(float)
    logistic=np.logaddexp(0,-z)
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    ax.step(z,loss01,where='post',lw=3,color='#d55543',label='Erro 0–1')
    ax.plot(z,logistic,lw=3,color='#1769aa',label='Perda logística (BCE)')
    ax.axvline(0, color='#65686c', ls=':', lw=1.5)
    ax.set(xlim=(-4,4), ylim=(-.05,4.1), xlabel='Margem da classificação: (2y − 1) × logit',
           ylabel='Valor da perda')
    ax.legend(frameon=False,loc='upper right')
    _format_slide(ax)
    fig.tight_layout()
    return fig


def slide_early(h):
    """8.1.2: mínimo da validação, com checkpoint e parada efetiva."""
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    ax.plot(h['epoch'],h['train_loss'],lw=3,c='#1769aa',label='Treino')
    ax.plot(h['epoch'],h['val_loss'],lw=3,c='#db6b24',label='Validação')
    k=h['best_epoch']
    ax.scatter([k],[h['val_loss'][k-1]],s=135,marker='*',zorder=6,color='#bc2b35')
    ax.axvline(k,c='#bc2b35',ls='--',lw=2,label=f'Melhor época: {k}')
    ax.axvline(h['stopped_epoch'],c='#6d7280',ls=':',lw=2,label=f'Parada: {h["stopped_epoch"]}')
    ax.set(xlabel='Época',ylabel='Perda média (BCE)')
    ax.legend(loc='upper right',frameon=False)
    _format_slide(ax)
    fig.tight_layout()
    return fig


def slide_batch(results, data):
    """8.1.3: trajetórias REAIS dos dois pesos, 3 amostradores, MESMO início.

    Cada método usa uma taxa de aprendizado didática própria, explicitada na
    função compare_batch(); portanto não é um benchmark de eficiência.
    """
    fig,ax=plt.subplots(figsize=(9.0,5.2))
    traces=[np.asarray(h['theta_updates']) for h in results.values()]
    gx,gy,z=loss_surface(*data['train'], np.vstack(traces), ngrid=96, padding=.7)
    lev=np.linspace(float(z.min()),float(np.quantile(z,.83)),18)
    ax.contourf(gx,gy,z,levels=lev,extend='max',cmap='Blues',alpha=.8)
    ax.contour(gx,gy,z,levels=lev[::3],colors='#5b7e9b',alpha=.36,linewidths=.7)
    colors=['#101f36','#e88b22','#ca3e44']
    for (name,h),col in zip(results.items(),colors):
        t=np.asarray(h['theta_updates'])
        # Um ponto em cada etapa relevante: mantém trajetória legível no slide.
        idx=np.unique(np.linspace(0,len(t)-1,min(270,len(t)),dtype=int))
        ax.plot(t[idx,0],t[idx,1],lw=2.7,color=col,alpha=.95,label=name)
        ax.scatter(t[-1,0],t[-1,1],s=70,color=col,edgecolor='white',zorder=7)
    ax.scatter(traces[0][0,0],traces[0][0,1],s=95,color='white',edgecolor='#111',lw=2,zorder=8)
    ax.annotate('Mesmo início', xy=traces[0][0], xytext=(4.0,-2.3),
                fontsize=11, color='#101f36',arrowprops=dict(arrowstyle='->',color='#101f36'))
    ax.set(xlabel='Peso w₁',ylabel='Peso w₂', xlim=(-2.0,4.35), ylim=(-3.9,2.0))
    ax.legend(loc='upper right',facecolor='white',framealpha=.95)
    fig.tight_layout()
    return fig


def slide_variance(sizes,var):
    """8.1.3: amostragem com lote pequeno gera estimativa mais variável."""
    fig,ax=plt.subplots(figsize=(9.0,5.0))
    ax.plot(sizes,var,'-o',lw=3,ms=9,c='#15787e')
    ax.set(xscale='log', xlabel='Número de imagens em cada lote (B)',
           ylabel='Erro quadrático médio do gradiente amostral')
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(x) for x in sizes])
    ax.axhline(0,lw=1,c='#929ca7',ls=':')
    ax.annotate('SGD: uma imagem',xy=(sizes[0],var[0]),xytext=(2,var[0]*.76),
                fontsize=12,arrowprops=dict(arrowstyle='->'))
    ax.annotate('Batch completo',xy=(sizes[-1],var[-1]),xytext=(22,var[0]*.26),
                fontsize=12,arrowprops=dict(arrowstyle='->'))
    _format_slide(ax)
    fig.tight_layout()
    return fig


def plot_surface_3d(data, h):
    """Gráfico 3D opcional para explicar o formato da perda (não é recomendado
    como figura principal, pois perspectiva pode esconder a trajetória)."""
    path=np.asarray(h['theta_updates'])
    gx,gy,z=loss_surface(*data,path,ngrid=60,padding=.8)
    x,y=data
    p=torch.tensor(path,dtype=torch.float32)
    with torch.no_grad():
        logits=p@x.float().T
        pz=(F.softplus(logits)-logits*y.float().unsqueeze(0)).mean(1).numpy()
    fig=plt.figure(figsize=(10,7.0))
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(gx,gy,z,cmap='Blues',alpha=.36,rstride=1,cstride=1,linewidth=0)
    ax.plot(path[:,0],path[:,1],pz+0.12,c='#d02038',lw=3,marker='.',markersize=4)
    ax.scatter(*path[0],pz[0]+.15,c='black',s=70,label='Início')
    ax.scatter(*path[-1],pz[-1]+.15,c='#d02038',s=95,label='Fim')
    ax.set(xlabel='w₁',ylabel='w₂',zlabel='Risco empírico (BCE)')
    ax.view_init(elev=28,azim=-65)
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig


def salvar(fig, filename, *, slide=False):
    """Salva imagem em PNG sem depender de Drive; útil para Colab/PowerPoint."""
    target = SLIDE_DIR / filename if slide else OUT / filename
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return target


def mostrar_png(path):
    """Mostra exatamente a figura gravada em disco."""
    print(f"Gráfico salvo: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Experimentos Goodfellow 8.1.1–8.1.3 em PyTorch.")
    parser.add_argument("--config", type=Path, default=None, help="Arquivo JSON com campos de AppConfig")
    parser.add_argument("--outdir", type=Path, default=None, help="Diretório para PNGs, GIF e ZIPs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--dataset", choices=("shapes", "mnist", "fashion_mnist", "folder", "npz"), default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--download", action="store_true", help="Autoriza baixar MNIST/Fashion-MNIST")
    parser.add_argument("--epochs-erm", type=int, default=None)
    parser.add_argument("--epochs-early", type=int, default=None)
    parser.add_argument("--epochs-batch", type=int, default=None)
    return parser.parse_args()


def build_config(args):
    # Mantém os valores explícitos no notebook original.
    defaults = dict(dataset='shapes', download=False, class_pair=(3, 8),
                    image_size=20, seed=42, n_train=120, n_val=260, n_test=260,
                    image_noise=0.28, flip_train=0.25, model='mlp', hidden=96,
                    epochs_erm=110, epochs_early=180, epochs_batch=15,
                    batch_size=20, patience=19, lr_erm=0.14, lr_early=0.15,
                    device='cpu', outdir='graficos')
    if args.config:
        overrides = json.loads(args.config.read_text(encoding='utf-8'))
        unknown = set(overrides) - {f.name for f in fields(AppConfig)}
        if unknown:
            raise ValueError(f"Campos de configuração desconhecidos: {sorted(unknown)}")
        defaults.update(overrides)
    for attr in ('outdir','device','dataset','epochs_erm','epochs_early','epochs_batch'):
        value = getattr(args, attr)
        if value is not None:
            defaults[attr] = str(value) if isinstance(value, Path) else value
    if args.data_path is not None:
        defaults['data_path'] = args.data_path
    if args.download:
        defaults['download'] = True
    if defaults['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado, mas indisponível. Execute com --device cpu.")
    return AppConfig(**defaults)


def main():
    global OUT, SLIDE_DIR
    args = parse_args()
    cfg = build_config(args)
    OUT = Path(cfg.outdir)
    SLIDE_DIR = OUT / 'para_slides'
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    print('Dataset:',cfg.dataset,'| Modelo:',cfg.model,'| Imagens:',cfg.image_size,'x',cfg.image_size)
    # CFG preserva os nomes utilizados no notebook nas células executáveis.
    CFG = cfg

    # Imagens de entrada (célula 13 do notebook)
    splits = load_images(CFG)                  # imagens criadas no próprio notebook, sem arquivos
    proj, data2d = as_projected(splits)    # extrator ajustado somente no treino
    print('Treino:',tuple(splits.x_train.shape), '| Validação:',tuple(splits.x_val.shape),
          '| Teste:',tuple(splits.x_test.shape) if splits.x_test is not None else 'ausente')
    p = salvar(plot_image_gallery(splits), '8_1_1_imagens_entrada.png')
    mostrar_png(p)

    # Descida do gradiente (célula 15 do notebook)
    # TESTE A DESCIDA: mude o learning rate ou a inicialização e execute esta célula.
    set_seed(CFG.seed + 10)
    linear = build_model('linear2d')
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, -3.0]]))  # ponto inicial, modificável
    linear, h_grad = fit(linear, data2d['train'], data2d['val'],
                         epochs=55, lr=0.35, batch_size=len(splits.y_train),
                         seed=CFG.seed+11, record_updates=True)
    p = salvar(plot_descent(data2d['train'],h_grad), '8_1_1_descida_contornos.png')
    mostrar_png(p)
    p3 = salvar(plot_surface_3d(data2d['train'],h_grad), '8_1_1_descida_superficie_3d.png')
    mostrar_png(p3)
    print('Passos do gradiente:',h_grad['updates_total'][-1],
          '| perda após 1ª época:',round(h_grad['train_loss'][0],4),
          '| perda final:',round(h_grad['train_loss'][-1],4))

    # Animação GIF (célula 17 do notebook)
    anim = animate_descent(data2d['train'], h_grad, frames=55, interval=120,
                           save_gif=OUT/'8_1_1_descida_animada.gif')

    print('Animação salva em:',OUT/'8_1_1_descida_animada.gif')

    # Risco empírico (célula 19 do notebook)
    set_seed(CFG.seed+14)
    model_erm=build_model(CFG.model,image_shape=tuple(splits.x_train.shape[-2:]),hidden=CFG.hidden)
    _,h_erm=fit(model_erm,(splits.x_train,splits.y_train),(splits.x_val,splits.y_val),
                epochs=CFG.epochs_erm,lr=CFG.lr_erm,batch_size=CFG.batch_size,
                patience=None,seed=CFG.seed+15,device=CFG.device)
    mostrar_png(salvar(plot_erm(h_erm),'8_1_1_erm_detalhado.png'))
    mostrar_png(salvar(slide_erm(h_erm),'8_1_1_risco_empirico_vs_validacao.png',slide=True))
    print('Perda final de treino:',round(h_erm['train_loss'][-1],4),
          '| validação:',round(h_erm['val_loss'][-1],4))

    # Funções de perda substitutas (célula 21 do notebook)
    mostrar_png(salvar(plot_surrogates(),'8_1_2_surrogates_detalhado.png'))
    mostrar_png(salvar(slide_surrogates(),'8_1_2_loss_0_1_vs_logistica.png',slide=True))

    # Early stopping (célula 22 do notebook)
    set_seed(CFG.seed+14)
    model_es=build_model(CFG.model,image_shape=tuple(splits.x_train.shape[-2:]),hidden=CFG.hidden)
    model_es,h_es=fit(model_es,(splits.x_train,splits.y_train),
                       (splits.x_val,splits.y_val),epochs=CFG.epochs_early,
                       lr=CFG.lr_early,batch_size=CFG.batch_size,
                       patience=CFG.patience,seed=CFG.seed+15,device=CFG.device)
    mostrar_png(salvar(plot_early_stopping(h_es),'8_1_2_early_stopping_detalhado.png'))
    mostrar_png(salvar(slide_early(h_es),'8_1_2_early_stopping.png',slide=True))
    print(f'Melhor época: {h_es["best_epoch"]}; parada: {h_es["stopped_epoch"]}; '
          f'perda de validação restaurada: '
          f'{evaluate(model_es,splits.x_val,splits.y_val,CFG.device)["loss"]:.4f}')
    if splits.x_test is not None:
        print('Avaliação FINAL no teste:',evaluate(model_es,splits.x_test,splits.y_test,CFG.device))

    # Comparação de lotes (célula 24 do notebook)
    results_batch = compare_batch(data2d,CFG)
    mostrar_png(salvar(plot_batch_comparison(results_batch,data2d),'8_1_3_comparacao_detalhada.png'))
    mostrar_png(salvar(slide_batch(results_batch,data2d),'8_1_3_trajetorias_batch_minibatch_sgd.png',slide=True))
    for name,h in results_batch.items():
        print(f'{name}: {h["updates_total"][-1]} atualizações, '
              f'{h["samples_seen"][-1]} imagens processadas, '
              f'perda final={h["train_loss"][-1]:.4f}')

    # Variância do gradiente (célula 26 do notebook)
    sizes,var=gradient_variance(data2d,sizes=(1,4,16,32,64,120),
                                repetitions=65,seed=CFG.seed)
    mostrar_png(salvar(plot_gradient_variance(sizes,var),'8_1_3_variancia_gradiente.png'))
    mostrar_png(salvar(slide_variance(sizes,var),'8_1_3_ruido_vs_tamanho_lote.png',slide=True))

    # Arquivos ZIP para apresentação
    # Não depende de Drive, de scripts nem de outros arquivos.
    all_files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.png','.gif'}]
    for archive,files in [(OUT / 'figuras_todas.zip',all_files),
                          (OUT / 'imagens_escolhidas_para_slides.zip',sorted(SLIDE_DIR.glob('*.png')))]:
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p,arcname=str(p.relative_to(OUT)))
        print(f'{archive}: {len(files)} imagens')
    print('\nVersões para apresentações:')
    for p in sorted(SLIDE_DIR.glob('*.png')):print(' •',p)


if __name__ == "__main__":
    main()
