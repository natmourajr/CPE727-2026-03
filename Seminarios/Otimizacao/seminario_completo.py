#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executa os experimentos preservados do 8.1 e os novos do 8.4–8.7.

Exemplos:
    python seminario_completo.py --chapters 8.4,8.5 --quick
    python seminario_completo.py --chapters 8.7 --subsection 8.7.3
    python seminario_completo.py --chapters all --outdir graficos
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from capitulos import cap_8_4, cap_8_5, cap_8_6, cap_8_7

DEFAULT_CHAPTERS=('8.1','8.4','8.5','8.6','8.7')
SUBTOPICS={
 '8.7.1':cap_8_7.batch_norm,
 '8.7.2':cap_8_7.coordinate_descent,
 '8.7.3':cap_8_7.polyak,
 '8.7.4':cap_8_7.supervised_pretraining,
 '8.7.5':cap_8_7.design_for_optimization,
 '8.7.6':cap_8_7.curriculum_learning,
}


def parse_args():
    p=argparse.ArgumentParser(description='Seminário Goodfellow cap. 8.1 e 8.4 a 8.7')
    p.add_argument('--chapters',default='all',help='all ou lista: 8.1,8.4,8.5,8.6,8.7')
    p.add_argument('--subsection',choices=sorted(SUBTOPICS),help='Uma seção 8.7.x isolada; exige --chapters 8.7')
    p.add_argument('--outdir',type=Path,default=Path('graficos'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--quick',action='store_true',help='Épocas reduzidas para teste rápido (8.1 inclui gráficos de 55 passos)')
    p.add_argument('--config-8-1',type=Path,default=None,
                   help='Configuração opcional do notebook convertido (config.json)')
    return p.parse_args()


def _gerar_zip(pasta):
    figuras=sorted(p for p in pasta.rglob('*') if p.suffix.lower() in ('.png','.gif') and p.is_file())
    arq=pasta/'figuras_todas.zip'
    with zipfile.ZipFile(arq,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in figuras:z.write(p,arcname=str(p.relative_to(pasta)))
    print(f'Arquivos gráficos: {len(figuras)} | pacote: {arq}')
    return len(figuras)


def main():
    a=parse_args()
    if a.chapters=='all':chapters=DEFAULT_CHAPTERS
    else:chapters=tuple(z.strip() for z in a.chapters.split(',') if z.strip())
    invalid=set(chapters)-set(DEFAULT_CHAPTERS)
    if invalid:raise SystemExit(f'Capítulos inválidos: {sorted(invalid)}; disponíveis: {DEFAULT_CHAPTERS}')
    if a.subsection and chapters!=('8.7',):
        raise SystemExit('--subsection exige --chapters 8.7')
    pasta=a.outdir.resolve()
    pasta.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    relatorio={'capitulos':list(chapters),'subsecao':a.subsection,'seed':a.seed,
               'quick':a.quick,'resultados':{}}
    for cap in chapters:
        print(f'\n=== Capítulo {cap} ===',flush=True)
        if cap=='8.1':
            arquivo=Path(__file__).resolve().parent/'seminario_8_1.py'
            cmd=[sys.executable,str(arquivo),'--outdir',str(pasta)]
            if a.config_8_1:cmd.extend(['--config',str(a.config_8_1.resolve())])
            if a.quick:cmd.extend(['--epochs-erm','4','--epochs-early','4','--epochs-batch','3'])
            subprocess.run(cmd,check=True)
            relatorio['resultados'][cap]={'status':'executado (script original)'}
        elif cap=='8.4':relatorio['resultados'][cap]=cap_8_4.executar(pasta,a.quick,a.seed)
        elif cap=='8.5':relatorio['resultados'][cap]=cap_8_5.executar(pasta,a.quick,a.seed)
        elif cap=='8.6':relatorio['resultados'][cap]=cap_8_6.executar(pasta,a.quick,a.seed)
        else:
            if a.subsection:
                fn=SUBTOPICS[a.subsection]
                if a.subsection=='8.7.2':resultado=fn(pasta,a.quick)
                else:resultado=fn(pasta,a.quick,a.seed)
                relatorio['resultados'][a.subsection]=resultado
            else:relatorio['resultados'][cap]=cap_8_7.executar(pasta,a.quick,a.seed)
    relatorio['figuras_geradas']=_gerar_zip(pasta)
    relatorio['duracao_segundos']=round(time.monotonic()-start,2)
    report_path=pasta/'relatorio_execucao.json'
    report_path.write_text(json.dumps(relatorio,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\nConcluído: {report_path} | duração={relatorio["duracao_segundos"]}s')


if __name__=='__main__':main()
