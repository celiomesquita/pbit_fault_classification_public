# E1 — Emenda de análise externa (descritiva)

Versão 1, fixada em 27/10/2026, depois da rodada computacional das 50 divisões (concluída
em 14/10/2026) e **antes de calcular qualquer CMR, L̄, FA0, DR ou macro-F1 externos**.
Complementa `E1.md` e `E1_rodada_completa.md` sem os alterar. Verificação de que nenhuma
medida externa foi calculada: o plano (`Estado da execução`) e o relatório do piloto
registram que não; o script desta emenda recusa executar se `data/e1_external/` já existir.

## Por que a análise é descritiva

`E1.md` condicionava testes, intervalos, equivalência e superioridade à validação de um
procedimento inferencial pelo E3. Os três exames da grade semissintética (E3, E3b, E3c) não
validaram nenhum procedimento para a CMR por ciclo nem para L̄ (cobertura mínima 0,855–0,893,
sempre em reduções de perda com correlação alta entre braços). A pendência fica, portanto,
resolvida no sentido negativo: **a análise externa não produz testes, valores-p, ajuste de Holm,
equivalência, superioridade nem não inferioridade.** A cláusula de "melhoria do compromisso"
de `E1.md` (CMR menor com teste centrado e FA0 não inferior) não pode ser satisfeita e é
substituída pelos rótulos descritivos abaixo.

## Dados e agregação

Ler as 50 divisões de `data/e1_full_v1/completion_manifest.json` (a 42/0 vem de
`data/e1_pilot`), conferindo antes de qualquer leitura de predição os hashes de
`run_manifest`, `verification` e `model_verification` e `batch_freeze.json`. Interromper se
faltar divisão, se um ciclo aparecer mais de uma vez por semente ou se alguma SN aparecer em
mais de um fold externo da mesma semente. População: os 1.195 ciclos das 54 SN de treino
histórico (exclui 931, 949 e 1107); as 11 SN do teste histórico não participam.

Para cada semente, agrupar as predições dos cinco folds externos (cada ciclo aparece uma vez),
calcular as métricas e relatar a **média sobre as dez sementes** e o intervalo mínimo–máximo.
Sementes não são réplicas independentes. Colunas de `external_predictions.csv`: `*_base`
(predição sem limiar) e `*_b30/b40/b50` (política ao orçamento interno de FA0 0,30/0,40/0,50),
para a referência `untuned`, os quatro braços ajustados, `minimum_cost` e `metacost`.
Quando a política foi inviável internamente (`budget_failed` em `decisions.json`), a coluna já
contém a predição-base; essa substituição não é reclassificada.

## Métricas e contrastes

Endpoints, todos descritivos: CMR por ciclo (F03/F06 preditas N01/N02), **L̄** (perda média
por SN crítica, igual peso por SN, como na Seção 2.3 do manuscrito), FA0 por ciclo, FA0 média
por SN normal (secundária), DR, R3 e macro-F1 (sobre a união de classes verdadeiras e
preditas). Sempre apresentar CMR/L̄ junto com FA0.

Contrastes: cada um dos seis braços menos a referência `untuned` **ao mesmo orçamento
interno**, para os orçamentos 0,40 (primário), 0,30 e 0,50 (secundários), com a diferença dos
endpoints acima. Os três orçamentos e os seis braços são todos relatados, sem escolher o
melhor a posteriori.

Intervalos: reamostragem pareada de SN (10.000 sorteios, semente 20261027, percentis 2,5 e
97,5), aplicada às predições agrupadas e às diferenças dentro do mesmo sorteio. Aparecem como
**intervalos descritivos não calibrados**, sem valores-p. Devem ser lidos à luz do E3, do E3b
e do E3c, que mostram sub-cobertura (até cerca de 0,88) justamente em reduções de perda.

Disponibilidade da política: relatar, por braço e orçamento, em quantos dos 50 folds externos a
política foi inviável (`budget_failed`), e em quantas sementes a FA0 externa excedeu o
orçamento. Um braço inviável em pelo menos metade dos folds é rotulado "política
majoritariamente indisponível": seu contraste mede quase só a predição-base e **não é lido como
comparação de políticas sob o mesmo FA0**.

## Rótulos descritivos, fixados antes de ver resultados

Sobre as estimativas pontuais médias, com os limites de ±0,02 para CMR/L̄/DR e ±0,03 para
FA0/macro-F1 usados apenas como limiares de leitura (não são margens de equivalência):

| CMR (ou L̄) vs. referência | FA0 vs. referência | Rótulo |
|---|---|---|
| menor (<= −0,02) | maior (>= +0,03) | Deslocamento ao longo do compromisso |
| menor | semelhante ou menor | Melhoria aparente do compromisso (não confirmada) |
| semelhante | menor | Melhoria aparente do compromisso (não confirmada) |
| semelhante | semelhante | Sem diferença apreciável |
| semelhante ou maior | maior | Piora aparente |
| maior (>= +0,02) | semelhante ou menor | Deslocamento inverso |

"Semelhante" significa diferença absoluta abaixo do limiar, sem implicar equivalência.
Cada rótulo recebe um indicador de consistência, avaliado apenas nos endpoints classificados como
"menor" ou "maior" na tabela (os "semelhantes" não entram): "consistente" se o sinal da diferença
desse endpoint coincide com o da média em pelo menos 8 das 10 sementes e o intervalo descritivo de
SN exclui zero; caso contrário, "instável". "Sem diferença apreciável" não recebe indicador.
Rótulos de "melhoria aparente" só podem entrar no texto como achado exploratório, com a ressalva
do E3/E3b/E3c.

## Referência retrospectiva secundária

Para situar cada política em relação ao simples deslocamento do limiar, construir, por semente,
a curva de limiares do `untuned` sobre as probabilidades externas agrupadas (score
P(F03)+P(F06), mesma regra de promoção de `E1.md`) e reportar a menor CMR da curva com FA0 não
superior à FA0 externa atingida pela política. **Esta referência usa rótulos externos** e é
retrospectiva: serve para dizer "acima" ou "abaixo da curva", nunca para declarar melhoria. A
análise primária é a comparação com a referência de limiar interno.

## Interpretação pré-especificada de desfechos

- Nenhum braço com melhoria aparente consistente: a RQ2 recebe resposta descritiva negativa
  ("os ganhos de CMR obtidos pelos braços ajustados e de custo são deslocamentos ao longo do
  compromisso, sob seleção interna"), sem alegação de equivalência.
- Pelo menos um braço com melhoria aparente consistente em 0,40 e nos orçamentos vizinhos:
  relatar como achado exploratório não confirmado, acompanhado da disponibilidade da política e
  da posição na curva retrospectiva.
- Desfecho misto (consistência só em um orçamento, ou política indisponível): relatar todos os
  orçamentos e o rótulo "instável"; não resumir em um único vencedor.
- Qualquer outro resultado, inclusive adverso (por exemplo, piora aparente do braço ajustado
  por CMR): relatar integralmente; nenhum braço, orçamento ou métrica é omitido.

## Artefatos, testes e interrupção

`scripts/e1_external_analysis.py`, `tests/test_e1_external.py`, `data/e1_external_freeze/` (congelamento), `data/e1_external/`
(CSVs por semente/braço/orçamento, manifesto com hashes),
`data/e1_external_summary.md` e `paper/tables/tab_independent_policies.tex`. Esta emenda não
altera o manuscrito; a incorporação (R1, R4) é etapa posterior.

Testes antes de executar a análise real: métricas contra implementação escalar por laço em
dados sintéticos; classificação dos rótulos para todas as combinações da tabela; reamostragem
pareada de SN reproduzindo a diferença pontual quando todos os pesos são um; cálculo da curva
retrospectiva contra grade exaustiva. Conferência independente posterior de pelo menos um
braço e orçamento por cálculo escalar sobre `external_predictions.csv`.
Interromper perante hash divergente, divisões faltantes ou duplicadas, ou identidades
violadas, não por resultados desfavoráveis. Qualquer mudança de limiar, rótulo ou métrica
depois de ver resultados exige nova emenda declarada pós-resultado.
