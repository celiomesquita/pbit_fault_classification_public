# E15.g — filtragem global de treino, com avaliação externa comum

29/10/2026. Definido após E14.e, antes dos ajustes E15.g. Alternativa sem mapa físico: flags globais são suspeitas numéricas e não diagnósticos instrumentais. Objetivo: distinguir o efeito de filtrar o treino do efeito de remover casos da avaliação.

## População e contratos

Q0: 1.196 ciclos processáveis candidatos E13, nas mesmas 54 unidades de desenvolvimento, incluindo 1107/N02 como suspeito e excluindo 931/949 pela incapacidade analítica (>50% de ausência), independentemente da classe. Isso não confirma aquisição válida nem invalidez física. Q1: retirar do treino os 65 suspeitos globais previamente fixados, restando 1.131 ciclos na população completa antes do CV. A avaliação interna/externa mantém todos os ciclos Q0 processáveis, inclusive suspeitos. Nenhum SN inteiro será segregado (Q2 não tem corte fundamentado).

IDs e classes vêm da fonte canónica; sinais completos vêm do clean, com imputação histórica rastreada por E13. Usar valores F246 canónicos para reais e o extrator histórico para os mesmos sintéticos usados nas duas representações. 1107 recebe o papel do seu SN 045 nos folds externos/internos já fixados: não redesenhar folds após acrescentá-lo. Não analisar as 11 unidades do teste histórico.

Comparar quatro braços fixados, sem escolher o vencedor E14: `Q0_F246`, `Q0_empirical`, `Q1_F246`, `Q1_empirical`. Representação empírica é a mesma de E14.e (três regiões, mínimo 15, 48 atributos), reaprendida apenas nos reais efetivamente usados em cada treino. Não chamar essas regiões de fases físicas. Os limites aprendidos variam com a filtragem; o contraste mede o procedimento completo.

Mesmos pesos freq_sev, LightGBM B1 (400 árvores etc.), augmentation apenas de treino, política q e seleção interna E10, dez sementes e cinco folds internos/externos. Critério primário: recall ≥90% em cada F01–F05 e FA0 ≤0,40; sensibilidades 85%/95% e 0,30/0,50. F06 permanece exploratória e não será sintetizada se ausente no treino. Endpoint sem denominador é indefinido. Avaliar atendimento conjunto, cobertura e composição.

## Isolamento e execução

Aplicar a filtragem somente aos reais de treino; depois gerar os sintéticos com o gerador histórico, alvo/semente originais. As duas representações de uma mesma política usam exatamente os mesmos sinais expandidos. Imputadores ajustados no treino expandido; nenhum ciclo externo define flags, limites, atributos ou limiar. As flags são as mesmas congeladas no E13; não alterar limites após observar E14 ou associações E13.d.

Salvar IDs removidos/retidos, classes de treino, SN, limites aprendidos, scores internos/externos, limiares e decisões nos nove cenários. Comparar Q1−Q0 separadamente por representação e descrever a diferença entre contrastes. Relatar adicionalmente taxas/recalls em ciclos com/sem flag, sempre com denominadores. Não contar rejeição/exclusão como detecção. A ponte para E14 muda também a população (1107), de modo que diferenças não serão atribuídas só à filtragem.

Orçamento: 4 braços × 50 partições × (5 ajustes internos + 1 refit) = **1.200 ajustes** novos, sem tuning ou seleção de representação. Piloto seed42/outer0 integra o orçamento e mede apenas execução/recursos, sem calcular métricas externas. Depois completar as 50 partições, sem ampliar orçamento por resultado desfavorável. Até quatro processos/quatro threads. Antes dos ajustes, congelar protocolo, fontes, ambiente e entradas em `data/e15_global_v1/`.

## Interpretação

Ganho na mesma avaliação externa Q0 favorece o procedimento de filtragem nesse conjunto; não prova que flags sejam erros de monitoramento. Ganho apenas entre aceites descreve mudança de alvo/cobertura. Piora pode refletir perda de informação diagnóstica ou redução de treino. Ausência de ganho limita a regra global, sem descartar critérios por fase ou outros dados. Preservar resultados negativos e classes escassas. Confirmação física/externa continua não estabelecida.
