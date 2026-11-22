"""Testes rápidos de sanidade: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from capitulos import cap_8_4,cap_8_5,cap_8_6,cap_8_7
from capitulos.comum import semente

class TestesExperimentais(unittest.TestCase):
    def test_zeros_cria_gradiente_nulo_no_inicio_da_mlp(self):
        semente(123)
        m=cap_8_4.MLPProfunda()
        cap_8_4.inicializar(m,'zeros')
        x=torch.randn(12,2)
        y=torch.randint(0,2,(12,))
        nn.functional.cross_entropy(m(x),y).backward()
        self.assertEqual(float(m.rede[0].weight.grad.norm()),0.)

    def test_otimizadores_reduzem_custo_convexo(self):
        H,caminhos,perdas,_=cap_8_5._quadratica(18,(2.5,2.5))
        ini=.5*np.array([2.5,2.5])@H@np.array([2.5,2.5])
        for k,v in perdas.items():
            with self.subTest(optimizador=k):
                self.assertTrue(np.isfinite(v).all())
                self.assertLess(v[-1],ini)
                self.assertEqual(len(caminhos[k]),len(v)+1)

    def test_newton_cg_bfgs_para_quadratica_spd(self):
        H=np.array([[21.,3.],[3.,2.]])
        caminhos,_=cap_8_6._otimizar_quadratica(H,np.array([2.6,2.3]),n=12)
        self.assertEqual(len(caminhos['Newton'])-1,1)
        for nome in ('Newton','Gradientes conjugados (CG)','BFGS'):
            with self.subTest(nome=nome):
                self.assertLess(np.linalg.norm(caminhos[nome][-1]),1e-7)

    def test_secao_87_grafico_isolado(self):
        with tempfile.TemporaryDirectory() as d:
            resultado=cap_8_7.polyak(Path(d),rapido=True,seed=55)
            self.assertTrue((Path(d)/'8_7_3_polyak_perdas.png').is_file())
            self.assertTrue((Path(d)/'8_7_3_polyak_trajetorias.png').is_file())
            self.assertTrue(all(np.isfinite(x) for x in resultado.values()))

if __name__=='__main__':unittest.main()
