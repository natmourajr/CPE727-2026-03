"""8.6.1–8.6.3: Newton, gradientes conjugados e BFGS."""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from .comum import salvar, figura_curvas, contornos_caminhos


def _otimizar_quadratica(H, p0, n=12):
    """Pesquisa linear exata; somente problema quadrático convexo 2D."""
    caminhos = {}
    perdas = {}
    for nome in ['Descida mais íngreme', 'Newton', 'Gradientes conjugados (CG)', 'BFGS']:
        p = p0.astype(float).copy()
        B = np.eye(2)
        d = -H @ p
        r_anterior = None
        traj=[p.copy()]
        vals = [0.5 * p @ H @ p]
        for it in range(n):
            g = H @ p
            if np.linalg.norm(g)<1e-10: break
            if nome == 'Newton':
                # Hessiana recuperada de forma automática (exemplo 8.6.1).
                t = torch.tensor(p,dtype=torch.float64,requires_grad=True)
                matriz_h = torch.autograd.functional.hessian(
                    lambda z:.5*z @ torch.from_numpy(H) @ z,t)
                passo = -torch.linalg.solve(matriz_h,torch.from_numpy(g)).numpy()
                novo = p + passo
            elif nome == 'Descida mais íngreme':
                d = -g
                alpha = -(g @ d)/(d @ H @ d)
                novo = p + alpha*d
            elif nome == 'Gradientes conjugados (CG)':
                if it == 0:
                    d = -g
                else:
                    beta = (g@g)/(r_anterior@r_anterior)
                    d = -g + beta*d
                alpha = -(g@d)/(d@H@d)
                novo = p+alpha*d
                r_anterior = g.copy()
            elif nome == 'BFGS':
                d = -B@g
                alpha = -(g@d)/(d@H@d)
                novo = p + alpha*d
                s = novo-p
                y = H@s
                sy = s@y
                if sy > 1e-14:
                    rho = 1/sy
                    I=np.eye(len(p))
                    B=(I-rho*np.outer(s,y))@B@(I-rho*np.outer(y,s))+rho*np.outer(s,s)
            p=novo
            traj.append(p.copy())
            vals.append(max(float(.5*p@H@p),1e-16))
        caminhos[nome] = np.asarray(traj)
        perdas[nome] = vals
    return caminhos,perdas


def _selas():
    """Newton zera o gradiente e pode encontrar sela em problema não convexo."""
    p0=np.array([1.3,1.3],float)
    H=np.diag([2.,-2.])
    def v(p):return float(.5*p@H@p)
    p_newton=p0-np.linalg.solve(H,H@p0)
    pg=p0.copy(); descida=[pg.copy()]
    for _ in range(8):
        pg=pg-.22*H@pg
        descida.append(pg.copy())
    fig,ax=plt.subplots(figsize=(7,5))
    xa=np.linspace(-2.3,2.3,160)
    X,Y=np.meshgrid(xa,xa)
    Z=X**2-Y**2
    ax.contour(X,Y,Z,levels=np.linspace(-5,5,18),cmap='coolwarm',alpha=.7)
    descida=np.asarray(descida)
    ax.plot(descida[:,0],descida[:,1],'-o',label='Descida do gradiente',ms=3)
    ax.plot([p0[0],p_newton[0]],[p0[1],p_newton[1]],'-s',label='Newton → ponto de sela')
    ax.scatter([0],[0],marker='x',s=130,c='black',label='Sela (0,0)')
    ax.set(title='8.6.1 — Atenção: Newton pode convergir a um ponto de sela',
           xlabel='x',ylabel='y',xlim=(-2.3,2.3),ylim=(-2.3,2.3))
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    return fig


def executar(pasta: Path, rapido=False, seed=42):
    H=np.array([[21.,3.],[3.,2.]],dtype=np.float64)
    # SPD: autovalores positivos; mínimo exato em θ=(0,0).
    assert np.linalg.eigvalsh(H).min()>0
    caminhos,perdas=_otimizar_quadratica(H,np.array([2.6,2.3]),n=5 if rapido else 17)
    salvar(contornos_caminhos(caminhos,H,'8.6 — GD × Newton × CG × BFGS'),
           pasta,'8_6_01_trajetorias_newton_cg_bfgs.png')
    salvar(figura_curvas(perdas,'8.6 — Queda do custo quadrático',
                         eixo='Custo J(θ)',escala_log=True),
           pasta,'8_6_02_convergencia_segunda_ordem.png')
    salvar(_selas(),pasta,'8_6_03_newton_ponto_de_sela.png')
    print('[8.6] Atualizações até gradiente pequeno:',{k:len(v)-1 for k,v in caminhos.items()})
    return {'passos':{k:len(v)-1 for k,v in caminhos.items()},
            'perda_final':{k:v[-1] for k,v in perdas.items()}}
