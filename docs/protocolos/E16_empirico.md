# E16.e — ablação das regiões numéricas e perturbação dos limites

29/10/2026. Definido antes de qualquer ajuste E16.e, após os resultados E14.e e antes da conclusão E15.g. Adapta a ablação física planejada à indisponibilidade do mapa: não se atribuirão nomes físicos às regiões e a análise será exploratória.

## Pergunta

Qual informação preditiva dos três trechos numéricos é utilizada pelo procedimento fixado, e o desempenho muda ao deslocar os limites em uma posição? Ablação não demonstra localização física do defeito; retirada de atributos também altera dimensionalidade, capacidade efetiva e limiar selecionado.

## Contrato e braços

População E10/E14 de 1.195 ciclos/54 SN, sem 1107 e sem teste histórico, para isolar o contraste com M2/E14. Não misturar seus denominadores com a avaliação Q0 de 1.196 ciclos E15. Mesmos folds, pesos, augmentation, configuração B1, critérios recall/FA0, nove cenários e F06 exploratória. Os limites de cada treino são os já congelados e verificados em E14, aprendidos sem labels e sem avaliação externa. Não escolher regiões pelos efeitos observados.

Reaproveitar `M2_reference` com scores/limites de E14 e executar oito braços novos:

- `only_R1`, `only_R2`, `only_R3`: 14 descritores da região, retirando todos os contrastes com outras regiões.
- `without_R1`, `without_R2`, `without_R3`: 28 descritores das duas regiões restantes e seus dois contrastes (30 atributos); retirar todo atributo/contraste que use a região omitida.
- `limits_minus1` e `limits_plus1`: deslocar simultaneamente os dois limites por −1/+1 posição, preservando ordem e regiões não vazias, e recalcular os mesmos 48 atributos. É um controle artificial estreito de alinhamento, não intervalo de incerteza física. O comprimento pode cair de 15 para 14 na perturbação; isso é deliberado e não muda o aprendizado original.

Cada braço é retreinado e recebe limiar interno próprio, nas mesmas unidades externas. Representações usam os mesmos sinais reais/sintéticos do controle E14. Verificar origem/ordem dos caches e scores, sem aprender novamente os limites em dados externos. Manter a amplitude e nível absolutos; nenhuma derivada será apresentada como velocidade/aceleração física.

Os limites comuns e a escala da augmentation foram definidos a partir dos sinais de treino completos. A ablação condiciona nesses componentes compartilhados; não equivale a treinar/adquirir somente um trecho desde a origem. Os atributos do ciclo externo entregues ao modelo isolado dependem apenas da região retida, sem normalização pelo ciclo externo completo. Não alegar independência causal de regiões ou capacidade de diagnóstico antecipado.

Orçamento: 8 braços × 50 partições × (5 ajustes internos + 1 refit) = **2.400 ajustes novos**. Referência sem refits verificada; piloto seed42/outer0 integra o orçamento, sem métricas externas. Até quatro processos/quatro threads, após concluir E15 para evitar competição de recursos. Congelamento antes do primeiro ajuste em `data/e16_empirical_v1/`; não ampliar braços/perturbações por resultado desfavorável.

## Análise e interpretação

Relatar recalls por classe, FA0, cumprimento conjunto, diferença pareada para a referência por semente/SN e mudanças de classe/limiar. Queda de recall isolada acompanhada de redução de FA0 pode expressar trade-off; não concluir mecanismo por um endpoint. Benefício de uma região isolada não habilita seleção externa de um novo vencedor confirmatório. Falta de perda ao remover região pode refletir redundância; falta de atendimento não prova ausência de informação local. Interpretar todos os braços, incluindo direções contrárias, com limites de evidência independente de F05/F06.
