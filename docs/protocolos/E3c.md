# E3c — Perda média por unidade (L̄) como estimando crítico e confirmação do WCR-C para FA0

Versão 1, fixada em 27/10/2026 depois dos resultados negativos do E3 e do E3b e antes de
implementar ou simular qualquer procedimento deste protocolo. Não altera `E3.md`, `E3b.md`,
seus congelamentos ou resultados. Decisão D1 do plano (rota "E3c com prazo + R3"), tomada
pelo autor correspondente em 27/10/2026.

## Informação já conhecida e natureza pós-resultado

No E3 e no E3b, nenhum de dez procedimentos cumpriu os critérios para a CMR ponderada por
ciclo. A falha concentra-se em reduções de CMR com correlação latente 0,8 (cobertura mínima
0,855–0,879 no E3b, delta −0,04) e em 14 SN críticas com exposições muito desiguais; a FA0 ficou
calibrada (WCR-C 0,946–0,954 na etapa A do E3b). Este protocolo foi desenhado conhecendo esse
padrão. Duas escolhas são, portanto, **pós-resultado** e assim serão registradas:

1. trocar o estimando crítico primário de CMR por ciclo para L̄ (peso igual por SN);
2. restringir a FA0 ao WCR-C, o procedimento que não falhou no E3b.

Nenhuma delas foi selecionada para obter um resultado favorável sem controle: a etapa B usa
sementes independentes e os critérios são os do E3, sem folga.

## Estimandos

- **L̄ (crítico):** para cada SN crítica g, x_g = D_g/n_g, onde D_g é a diferença entre
  braços das contagens de perdas críticas (F03/F06 preditas N01/N02), média sobre as dez
  sementes, e n_g é o número de ciclos críticos. L̄ = (1/G) · sum x_g, com G o número de SN
  com n_g > 0. É a diferença A−B da perda média por unidade L_u da Seção 2.3 do manuscrito.
  Responde a "quanto perde uma unidade crítica típica", não a "quantos ciclos críticos se
  perdem"; as duas questões diferem quando as exposições são desiguais (73 vs 1 ciclo).
- **FA0 (cycle-weighted):** razão de totais sum D_g / sum n_g sobre as SN com ciclos normais,
  como no E3b, sem mudança.

Sob o gerador do E3, E[p_A − p_B] = delta por SN, logo E[x_g] = delta e a verdade de L̄ é
exatamente delta, assim como a da razão. Este ponto é verificado por teste numérico antes
do piloto.

## Procedimentos

L̄ é a média simples de x_g, isto é, a razão do E3b com D_g → x_g e n_g → 1. Os quatro
procedimentos de `scripts/cluster_ratio_wild.py` são aplicados a (x_g, 1) sem alteração:

| ID | Procedimento (E3b) | Em L̄ |
|---|---|---|
| L-WC | WCR-C: Rademacher restrito ao nulo, CV1 | sign-flip estudentizado, SE com fator G/(G−1) |
| L-WV | WCR-V: restrito, CV3 | sign-flip estudentizado, SE jackknife |
| L-WU | WCU-B: não restrito, escores transformados | bootstrap-t selvagem |
| L-T | CV3-t com Student t(G−1) | teste t de uma amostra sobre as diferenças por SN |

Pesos, B = 1.999, p unilaterais com correção +1, p bilateral de caudas iguais, abstenção
quando o SE observado é zero ou não finito, e regras de equivalência TOST com nulos impostos
nas fronteiras: exatamente como no E3b. Para a FA0 aplica-se somente o WCR-C do E3b, com os
mesmos escores ponderados por ciclo.

## Gerador, cenários e endpoints

Reutilizar sem mudanças `setup`, `generate` e `scenarios` do script congelado do E3 (hash
conferido): mesma calibração beta, perfis histórico (54 SN; 14 críticas) e auditado (65 SN; 16
críticas), cópulas 0,8/0,2, efeitos, margens 0,02 (CMR/L̄) e 0,03 (FA0) e 40 cenários de
5.000 réplicas, oito contrastes com referência comum no nulo. Os cenários de CMR (20) alimentam
L̄; os de FA0 (20) alimentam o WCR-C.

Endpoints idênticos aos do E3b: cobertura de 95% avaliada pelo teste no delta verdadeiro,
rejeição no nulo, equivalência para m = 1/3/4/6/8, FWER de Holm para m = 3/4/6/8 sob o nulo
global, abstenções, intervalo de Wilson e erro Monte Carlo. A amplitude por inversão não é
calculada (não entra em nenhuma decisão).

## Decisão em duas etapas, fixada antes dos resultados

Critérios idênticos ao E3 e ao E3b, em todos os cenários do endpoint: cobertura >= 0,93;
falsa equivalência nas fronteiras <= 0,05 para todo m; FWER de Holm <= 0,05 para todo m;
abstenções <= 0,01.

- **Etapa A (seleção para L̄):** quatro procedimentos nos 20 cenários de CMR, semente-base
  2026102701. Entre os elegíveis, menor desvio máximo de cobertura de 0,95, depois maior poder
  de equivalência m = 4 no nulo, depois a ordem L-WC, L-WV, L-WU, L-T.
- **Etapa B (confirmação, sementes independentes):** semente-base 2026102702, mesmos tamanhos.
  Roda (i) o procedimento escolhido na etapa A nos 20 cenários de CMR, e (ii) o WCR-C nos 20
  cenários de FA0. Cada endpoint é validado apenas se cumprir de novo todos os critérios.
  O WCR-C não passa por seleção; sua confirmação é independente da semente do E3b.

Sem elegível em A, ou falha em B: registrar ausência de procedimento validado para aquele
endpoint. **Não haverá nova tentativa**: nenhum candidato, critério, gerador, B ou número de
réplicas muda sem novo protocolo declarado pós-resultado, e o plano passa a tratar a inferência
da C2 como descritiva (R3).

A etapa B controla o acaso Monte Carlo da seleção, não a especificidade do gerador. Validação
significa calibração nesta grade semissintética para L̄ e para a FA0 por ciclo; não valida a
CMR por ciclo, DR nem macro-F1, e não garante desempenho fora dela.

## Consequências

- L̄ validado: o endpoint crítico primário das comparações do E1 e do E7 passa a ser a
  diferença de L̄, com o procedimento confirmado; a CMR por ciclo permanece descritiva, sempre
  acompanhada da FA0. A decisão "melhoria do compromisso" do E1 exige redução de L̄ **e** FA0
  não inferior, ambos com procedimentos confirmados.
- FA0 não confirmado: FA0 descritiva; "melhoria do compromisso" e "equivalência conjunta"
  não podem ser declaradas, apenas "mudança de ponto de operação" descritiva.
- Nenhum validado: tudo descritivo; resultado negativo promovido aos Resultados (R3).
- A aplicação ao E1 exige emenda própria (`E1_externo.md`), fixada antes de calcular CMR, L̄
  ou FA0 externas. Tabelas históricas só são recalculadas se o procedimento for validado.

## Artefatos e testes

`scripts/cluster_unit_mean.py` (envoltório de L̄), `scripts/cluster_coverage_e3c.py`
(simulação, retomada, seleção, relatório), `tests/test_cluster_unit_mean.py`,
`data/cluster_coverage_e3c/` (congelamento, piloto funcional separado, etapas A e B,
manifestos com hashes) e `data/cluster_coverage_e3c_summary.md`.

Testes antes do piloto: L̄ = média de x_g e verdade exata delta do gerador (momentos);
CV1 e CV3 sobre (x_g, 1) coincidem com as fórmulas escalares da média; L-T igual a
`scipy.stats.ttest_1samp` bilateral; p-valores do sign-flip contra implementação escalar por
laço com os mesmos pesos; com G pequeno, p do sign-flip estudentizado igual à enumeração
exata dos 2^G sinais; WCR-C da FA0 reproduz byte a byte o E3b para a mesma entrada e pesos.
Interromper perante identidades violadas ou entradas modificadas, não por resultados negativos.
Hash e horário locais são rastreabilidade, sem pré-registro público independente.
