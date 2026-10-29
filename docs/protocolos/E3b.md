# E3b — Wild bootstrap restrito e jackknife por SN para CMR e FA0

Versão 1, fixada em 14/10/2026 depois do resultado negativo do E3 e antes de
implementar ou simular qualquer procedimento deste protocolo. É um segundo exame
da mesma grade de cenários; não altera `E3.md`, seu congelamento ou seus resultados.
Fonte lida: MacKinnon, Nielsen e Webb (2023), Journal of Applied Econometrics
38(5):671–694, doi:10.1002/jae.2969, versão arXiv 2301.04527, secções 3, 5 e 6.
Inversão de testes restritos conforme Roodman et al. (2019), Stata Journal 19(1):4–60.

## Informação já conhecida

No E3, nenhum dos seis procedimentos cumpriu os critérios. Os wild-t não
restritos (WCU-C) tiveram cobertura 0,942–0,953 em FA0, mas 0,864–0,873 em CMR
com delta −0,04 e FWER Holm até 0,069. Os candidatos abaixo foram escolhidos
conhecendo este padrão, pela recomendação publicada para poucos clusters e
tamanhos desiguais, não por ajuste aos números. Critérios e gerador ficam iguais.

## Equivalência com a regressão por SN

A razão theta = sum Dg / N é o MQO de diferenças por ciclo sobre uma constante,
agrupado por SN. Escores sg = Dg − theta*ng; CV1 com k=1 coincide com o SE do E3.
Aqui G é o número de SN com exposição positiva no endpoint: SN sem ciclos não têm
observações na regressão, não alteram estimativas, escores nem bootstrap.
Com uma única restrição sobre o único parâmetro, os escores restritos
transformados por jackknife (eq. 32–37) coincidem com os restritos. Logo WCR-S
equivale a WCR-C e WCR-B a WCR-V neste modelo; testar essa identidade.

## Quatro procedimentos

Para o nulo theta=h: ag(h)=Dg−h*ng. Jackknife theta(g)=(sum D−Dg)/(N−ng);
CV3: se3=sqrt[(G−1)/G * sum (theta(g)−theta)²]. CV1: se1=sqrt[G/(G−1)*sum rg²]/N.
Pesos Rademacher vg, B=1.999 por réplica, independentes entre réplicas, comuns
aos três bootstraps e a todos os h da mesma réplica (exigido para a inversão).

1. WCR-C: sg*=vg*ag(h); delta*=sum sg*/N; resíduos sg*−delta*·ng;
   se1* pela fórmula CV1; t*=delta*/se1*; observado t(h)=(theta−h)/se1.
2. WCR-V: mesma geração; se3 observado e se3*=sqrt[(G−1)/G*sum(delta*(g)−delta*)²],
   delta*(g)=(sum s*−sg*)/(N−ng), como na eq. 29–30.
3. WCU-B: escores transformados s'g=N(theta−theta(g)); sg*=vg*s'g; se3*;
   t*=delta*/se3*; observado t(h)=(theta−h)/se3. Não impõe o nulo.
4. CV3-t: t(h)=(theta−h)/se3 contra Student t(G−1), sem bootstrap.

p unilaterais: p_sup(h)=(1+#{t*>=t(h)})/(B+1) e p_inf(h)=(1+#{t*<=t(h)})/(B+1).
p bilateral de caudas iguais: min(1, 2·min(p_sup, p_inf)). Conjunto de 95%:
{h: p bilateral(h) > 0,05}. Teste de zero: p bilateral(0) <= 0,05.
Equivalência para família m: p_sup(−margem) <= 0,05/m e p_inf(+margem) <= 0,05/m
(TOST direto, nulos impostos nas fronteiras). Cobertura avaliada pelo teste no
delta verdadeiro. Amplitude por inversão (bisseção) nas 500 primeiras réplicas
do primeiro contraste; é descritiva e não entra na decisão.

Abster quando o SE observado é zero ou não finito: sem cobertura, p=1, sem
equivalência. Pivôs internos com SE zero e numerador não zero são ±infinito;
0/0 vira zero. Contar e reportar degenerações, sem descarte silencioso.

## Gerador, cenários e endpoints

Reutilizar sem mudanças `setup`, `generate` e `scenarios` do script congelado
do E3 (hash conferido): mesma calibração beta, perfis histórico e auditado,
cópulas 0,8/0,2, efeitos, margens 0,02/0,03 e 40 cenários com 5.000 réplicas.
Oito contrastes com referência comum no nulo. O perfil histórico representa
o E1: as exposições críticas das 54 SN do E1 são idênticas; em FA0 diferem
num ciclo (SN com 59 em vez de 60). Não criar perfil adicional.

Endpoints como no E3, acrescentando m=6 (a família primária do E1 tem seis
contrastes): cobertura 95%, rejeição no nulo, equivalência m=1/3/4/6/8,
FWER Holm m=3/4/6/8 sob o nulo global, abstenções, Wilson e erro Monte Carlo.

## Decisão em duas etapas, fixada antes dos resultados

Critérios idênticos ao E3, em todos os cenários: cobertura >= 0,93; falsa
equivalência nas fronteiras <= 0,05 para todo m; FWER Holm <= 0,05 para todo m;
abstenções <= 0,01. Etapa A (seleção): quatro procedimentos, semente-base
2026101401. Entre elegíveis, menor desvio máximo de cobertura de 0,95, depois
maior poder de equivalência m=4 no nulo, depois a ordem 1–4.
Etapa B (confirmação): somente o escolhido, semente-base independente 2026101402,
mesmos cenários e tamanhos. Validado apenas se cumprir de novo todos os critérios.
Sem elegível em A, ou falha em B: registar ausência de procedimento validado.
Não acrescentar candidatos, mudar critérios, gerador, B ou réplicas depois de
ver resultados sem novo protocolo, declarado pós-resultado.

A etapa B controla a seleção por acaso Monte Carlo, não a especificidade do
gerador. Validação aqui significa calibração nesta grade semissintética para
CMR/FA0; não garante desempenho fora dela, nem valida DR ou macro-F1.

## Consequências e artefactos

Se validado: aplicar ao E1 somente por emenda própria de análise externa,
fixada antes de calcular CMR/FA0 externas; recalcular CMR/FA0 históricas como
previsto em `E3.md`. DR/macro-F1 conservam o procedimento histórico com ressalva.
Se não validado: E1/E4/E5/E7 mantêm a pendência inferencial.

`scripts/cluster_ratio_wild.py` (procedimentos e inversão),
`scripts/cluster_coverage_e3b.py` (simulação, retomada, seleção, relatório),
`tests/test_cluster_wild.py`, `data/cluster_coverage_e3b/` (congelamento,
piloto funcional separado, etapas A e B, manifestos com hashes) e
`data/cluster_coverage_e3b_summary.md`. Testes antes do piloto: identidade
razão = MQO com constante e CV1 = SE do E3; CV3 contra laço explícito;
escores restritos transformados (eq. 32 com matrizes explícitas) iguais aos
restritos; WCU-B escores eq. 11 iguais a N(theta−theta(g)); p-valores contra
implementação escalar por laço; inversão com p bilateral = 0,05 nos limites.
Interromper perante identidades violadas ou entradas modificadas, não por
resultados negativos. Hash e horário locais são rastreabilidade, sem
pré-registo público independente.
