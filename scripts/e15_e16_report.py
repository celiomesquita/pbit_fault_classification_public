"""Verified readable reports and static figures for quality and ablation stages."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['E15','E16'],required=True);args=parser.parse_args()
    stage=args.stage;folder=ROOT/('data/e15_global_v1' if stage=='E15' else 'data/e16_empirical_v1')
    verified=json.loads((folder/'verification.json').read_text());assert verified['passed']
    summary=pd.read_csv(folder/'summary.csv');primary=summary[np.isclose(summary.target,.90)&np.isclose(summary.budget,.40)].set_index('arm')
    if stage=='E15':
        arms=['Q0_F246','Q1_F246','Q0_empirical','Q1_empirical'];baseline=None
        title='E15.g — filtragem global do treino e avaliação externa comum'
        scope='Quatro braços e 1.200 ajustes novos. Todos foram avaliados nos mesmos 1.196 ciclos processáveis das 54 unidades de desenvolvimento, incluindo suspeitos. Q1 retira suspeitos somente do treino; Q0 os mantém.'
        caveat='A população muda em relação a E14: inclui 1107/N02 como suspeito, sem declará-lo aquisição fisicamente válida ou inválida. 931/949 ficam fora por cobertura insuficiente. Não atribuir diferenças para E14 exclusivamente à filtragem.'
        protocol='../docs/protocolos/E15_global.md';record='../docs/registro_E15_E16_2026-10-29.md'
    else:
        arms=['M2_reference','only_R1','only_R2','only_R3','without_R1','without_R2','without_R3','limits_minus1','limits_plus1'];baseline='M2_reference'
        title='E16.e — informação preditiva por região numérica'
        scope='Oito ablações/perturbações, 2.400 ajustes novos e referência M2/E14 reaproveitada e verificada. A população é a mesma de E14: 1.195 ciclos/54 SN. Não misturar seus denominadores com os 1.196 de E15.'
        caveat='R1/R2/R3 são regiões numéricas aprendidas nos treinos, não extensão/inversão/contração identificadas. Retirar uma região remove todos os seus contrastes; também altera dimensionalidade. Perturbações de ±1 posição são controles artificiais estreitos, sem interpretação como tolerância física.'
        protocol='../docs/protocolos/E16_empirico.md';record='../docs/registro_E15_E16_2026-10-29.md'
    cols=['FA0','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06']
    lines=[f'# {title}','','29/10/2026. **Rodada concluída e verificada.** '+scope,'',caveat,'',
        '## Cenário principal','',
        'Médias de dez sementes pareadas; as sementes reutilizam as unidades e não constituem réplicas independentes. Critério: detecção ≥90% em cada F01–F05 e FA0 ≤40%; F06 exploratória.','',
        '| Braço | FA0 | F01 | F02 | F03 | F04 | F05 | F06 exploratória | Admissível interno | Atendimento externo |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        r=primary.loc[arm]
        lines.append('| '+arm+' | '+' | '.join(f'{100*r[c]:.2f}%' for c in cols)+f' | {int(r.inner_feasible_folds)}/50 | {int(r.external_feasible_seeds)}/10 |')
    lines += ['',f"Na grade completa houve {int(summary.external_feasible_seeds.sum())} atendimentos externos e "+
        f"{int(summary.inner_feasible_folds.sum())} admissibilidades internas nas {len(summary)} combinações braço–cenário. "+
        'Essas combinações compartilham unidades, modelos e scores; contagens não são tamanho de amostra independente.','',
        '## Contrastes e alcance','']
    pairs=[('Q1_F246','Q0_F246'),('Q1_empirical','Q0_empirical')] if stage=='E15' else [(a,baseline) for a in arms if a!=baseline]
    for left,right in pairs:
        d=primary.loc[left,cols]-primary.loc[right,cols]
        lines.append(f'- {left} menos {right}: ΔFA0 {100*d.FA0:+.2f} pontos percentuais; Δrecall F02 {100*d.detect_F02:+.2f}; Δrecall F05 {100*d.detect_F05:+.2f}.')
    if stage=='E15':
        lines += ['',
            'Os contrastes Q1−Q0 usam os mesmos ciclos externos, portanto não ganham por eliminar casos da avaliação. Ainda assim, retirar suspeitos muda composição e quantidade do treino, augmentation e limites empíricos; o resultado é do procedimento completo e não demonstra remoção de ruído instrumental. Nenhum SN inteiro foi excluído: não existe corte fundamentado para Q2.','',
            'A [tabela condicional](e15_global_v1/conditional_seed_metrics.csv) separa os ciclos externos com/sem flags, com denominadores e endpoints indefinidos preservados. Desempenho entre aceites não substitui o resultado na população comum. Flags são indicadores numéricos definidos antes deste experimento, sem rótulos ou erros na decisão de qualidade.']
    else:
        lines += ['',
            'Recalls e FA0 devem ser considerados conjuntamente. Piora de recall com redução de falsos alarmes pode expressar mudança de decisão; melhora de uma classe não basta para atender às cinco restrições. Ausência de perda ao omitir uma região pode refletir redundância. Um trecho isolado eventualmente melhor não é selecionado como vencedor confirmatório. A comparação é descritiva, com limites de cobertura independente de F05/F06.']
    lines += ['',
        'Nenhum resultado identifica mecanismo físico, valida fases, confirma individualmente os diagnósticos ou estabelece utilidade operacional. A validação em unidades independentes permanece distinta desta rodada sobre a população já inspecionada.','',
        '## Verificação e artefatos','',
        'Os hashes de entradas/fontes/saídas, scores, seleção interna e decisões foram verificados independentemente. '+
        f"Erro escalar máximo: {verified['max_scalar_error']:.3g}. "+
        f"[Verificação completa]({folder.name}/verification.json).",'',
        f'[Protocolo]({protocol}) · [Registro]({record}) · '+
        f'[Grade completa]({folder.name}/summary.csv) · [Métricas por semente]({folder.name}/seed_metrics.csv) · '+
        f'[Diferenças pareadas]({folder.name}/paired_seed_differences.csv) · [Por SN/classe]({folder.name}/primary_unit_class_metrics.csv).','']
    # Separate recall scale from false alarms: high values have different meanings.
    data=primary.loc[arms,[f'detect_F0{i}' for i in range(1,6)]].to_numpy(float)
    fig,axes=plt.subplots(1,2,figsize=(12,max(4.3,len(arms)*.48)),gridspec_kw={'width_ratios':[2,1]})
    axes[0].imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[0].set(xticks=np.arange(5),xticklabels=['F01','F02','F03','F04','F05'],yticks=np.arange(len(arms)),yticklabels=arms,title='Fault-detection recall (%)')
    for i in range(len(arms)):
        for j in range(5):axes[0].text(j,i,f'{100*data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>.6 else 'black',fontsize=9)
    axes[1].barh(np.arange(len(arms)),primary.loc[arms,'FA0'],color='#ae6543')
    axes[1].axvline(.40,color='black',linestyle='--',linewidth=1)
    axes[1].set(yticks=np.arange(len(arms)),yticklabels=['']*len(arms),xlim=(0,1),ylim=(len(arms)-.5,-.5),title='False-alarm rate',xlabel='Rate; budget 0.40')
    fig.tight_layout();fig.savefig(folder/'primary_performance.png',dpi=170);fig.savefig(folder/'primary_performance.pdf');plt.close(fig)
    lines += [f'![Desempenho principal]({folder.name}/primary_performance.png)','']
    (ROOT/f'data/{stage.lower()}_summary.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
