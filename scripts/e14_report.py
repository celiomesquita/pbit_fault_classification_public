"""Generate the readable E14.e report from verified completed outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e14_empirical_v2'


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['passed']
    summary=pd.read_csv(OUT/'summary.csv')
    primary=summary[np.isclose(summary.target,.90)&np.isclose(summary.budget,.40)].set_index('arm')
    columns=['FA0','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06']
    lines=['# E14.e — localização empírica ao longo das medições','',
        '29/10/2026. **Rodada concluída e verificada:** 50 partições, 900 ajustes novos, sem busca de hiperparâmetros. O controle F246/B1 foi reaproveitado com hashes e reprodução verificados. Nenhuma unidade foi excluída.','',
        'O autor não dispõe dos limites das fases nem do significado do canal. As regiões deste experimento são numéricas; não foram identificadas como extensão, inversão ou contração.','',
        '## Cenário principal','',
        'Médias das dez sementes pareadas. Recall mínimo de 90% em cada classe F01–F05 e FA0 ≤40%; F06 exploratória. Detecção aceita qualquer previsão de falha. Sementes não são réplicas independentes.','',
        '| Braço | FA0 | F01 | F02 | F03 | F04 | F05 | F06 exploratória | Admissível interno | Atendimento externo |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for arm in ['M0_F246','M1_uniform','M2_empirical','M3_combined']:
        r=primary.loc[arm]
        lines.append('| '+arm+' | '+' | '.join(f'{100*r[c]:.2f}%' for c in columns)+
            f" | {int(r.inner_feasible_folds)}/50 | {int(r.external_feasible_seeds)}/10 |")
    for left,right,title in [('M2_empirical','M1_uniform','Segmentação empírica menos uniforme'),
                              ('M3_combined','M0_F246','Representação combinada menos F246')]:
        delta=primary.loc[left,columns]-primary.loc[right,columns]
        lines += ['',f'**{title}:** ΔFA0 = {100*delta.FA0:+.2f} pontos percentuais; '+
            f'Δrecall F02 = {100*delta.detect_F02:+.2f}; Δrecall F05 = {100*delta.detect_F05:+.2f}. '+
            'São contrastes descritivos entre decisões com limiares escolhidos internamente, sem impor FA0 externa coincidente.']
    total=int(summary.external_feasible_seeds.sum())
    lines += ['',f'Nos nove cenários, houve {total} atendimentos em 360 combinações braço–cenário–semente. '+
        'As repetições e cenários compartilham unidades e scores; esse denominador descreve a grade, não 360 amostras independentes.','',
        '## Regiões e controles','',
        'M1 usa limites 41/82. M2 aprende dois limites comuns apenas nos sinais reais de cada treino, sem labels e com peso total igual por SN; no mínimo 15 posições por região. Minimiza erro de aproximação constante após normalização por ciclo, usada somente no alinhamento. Os 48 atributos preservam níveis absolutos. M3 combina esses atributos com F246 (294 ao todo).','']
    boundaries=pd.read_csv(OUT/'region_boundaries.csv')
    for col,label in [('first','Primeiro limite'),('second','Segundo limite')]:
        x=boundaries[col]
        lines.append(f'- {label}, nas 300 partições de aprendizagem (250 internas/50 externas): '+
            f'mínimo {int(x.min())}, mediana {x.median():.1f}, máximo {int(x.max())}.')
    at_constraint=int((boundaries['first']==15).sum())
    lines += ['',f'O primeiro limite coincidiu com o comprimento mínimo permitido em {at_constraint}/300 treinos. '+
        'Limites no mínimo da restrição não demonstram uma transição natural. A normalização pode amplificar pequenas flutuações de ciclos pouco variáveis; limites comuns também podem ignorar assinaturas heterogêneas. Nenhum desses resultados valida o mapa físico.','',
        'Cada treino rege sua própria segmentação e imputação; os SN externos não definem limites. As realizações sintéticas correspondem aos metadados históricos e ao mesmo gerador/sementes. F246 já contém segmentos uniformes e contrastes locais: o novo teste não é uma comparação entre informação global e qualquer informação local.','',
        '## Interpretação e alcance','',
        'M2−M1 examina segmentação empírica sob dimensionalidade igual. M3−M0 examina informação incremental, com maior dimensionalidade e redundância. Ganhos em uma classe acompanhados por piora de FA0 ou de outra classe não bastam para atender ao objetivo conjunto. Não escolher uma representação vencedora a partir de um endpoint isolado.','',
        'A ausência de ganho ou atendimento limita o procedimento ensaiado; não demonstra inexistência de assinaturas locais ou impossibilidade física. Qualquer ganho continua exploratório na população já inspecionada. F06 e F05 têm cobertura independente escassa. Não foi coletada coorte externa nem validada uma regra operacional.','',
        '## Rastreabilidade','',
        f"Verificação independente: {verification['input_hashes']} hashes de entrada, {verification['source_hashes']} de fontes, "+
        f"{verification['output_hashes']} de saídas por partição; {verification['independent_primary_selections']} seleções primárias "+
        f"recomputadas por decisões explícitas; {verification['independently_checked_decision_rows']} decisões e "+
        f"{verification['independent_seed_metric_rows']} linhas de métricas conferidas. "+
        f"Erro escalar máximo {verification['max_scalar_error']:.3g}. Controle reproduz E10/B1. Três testes de regiões passaram antes do piloto.",'',
        'A execução inicial v1 interrompeu antes de qualquer ajuste por uma asserção de população; v2 filtrou explicitamente o long pelos IDs de desenvolvimento, com emenda e novo congelamento. Nenhum resultado histórico foi alterado.','',
        '[Protocolo](../docs/protocolos/E14_empirico.md) · [Emenda](../docs/protocolos/E14_empirico_emenda1.md) · '+
        '[Registro](../docs/registro_E14_empirico_2026-10-29.md) · [Verificação](e14_empirical_v2/verification.json) · '+
        '[Grade completa](e14_empirical_v2/summary.csv) · [Métricas por semente](e14_empirical_v2/seed_metrics.csv) · '+
        '[Por SN/classe](e14_empirical_v2/primary_unit_class_metrics.csv) · [Diferenças pareadas](e14_empirical_v2/paired_seed_differences.csv).','',
        '![Desempenho primário](e14_empirical_v2/primary_performance.png)','',
        '![Limites aprendidos](e14_empirical_v2/region_boundaries.png)','',
        '![Diferenças por unidade](e14_empirical_v2/unit_differences.png)','']
    (ROOT/'data/e14_summary.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
