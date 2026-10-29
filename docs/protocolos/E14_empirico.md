# E14.e — localização empírica sem nomes de fases físicas

29/10/2026. Versão 1. **Definida antes de ajustes E14.e**, após E10/E12/E13 global. O autor informou não conhecer os limites das fases nem a variável do canal. A execução já está autorizada. Esta é uma alternativa exploratória ao mapa físico indisponível; não preenche artificialmente esse mapa nem modifica os protocolos anteriores.

## Pergunta e delimitação

A identificação de regiões numéricas contíguas, aprendida sem labels apenas no treino, melhora a detecção entre SN em comparação a regiões uniformes com os mesmos atributos? Não se atribuirão os termos extensão/inversão/contração às regiões. Número de regiões K=3 é uma escolha analítica fixada, não a identificação de três estados físicos.

Mantêm-se população E10 (1.195 ciclos/54 SN), 50 partições externas, cinco folds internos por SN, dez sementes históricas, configuração LightGBM B1, pesos e augmentation históricos e política q completa. Recall principal 90% por classe F01–F05, sensibilidades 85%/95%; FA0 ≤0,40, sensibilidades 0,30/0,50; F06 exploratória no treino. Não excluir SN ou ciclos por seus erros. Não usar teste histórico.

## Alinhamento empírico, sem classe

Cada conjunto de treino real aprende **dois limites comuns**, com segmentos de no mínimo 15 posições e cobertura das 123 posições. Para aprender os limites, centrar cada ciclo na própria média e dividir pelo seu desvio; ciclos com desvio <1e-6 tornam-se zero nessa etapa. Isso é usado apenas para alinhamento; os atributos preditivos preservam os valores absolutos.

Dar peso total igual a cada SN real e igual por ciclo dentro dele. Minimizar a soma ponderada dos resíduos quadráticos de uma aproximação constante por região de cada ciclo normalizado. Calcular exatamente o custo de todos os pares de limites admissíveis; empate: escolher o par lexicograficamente menor. Essa função usa somente sinais reais e identidades de agrupamento, sem labels, previsões, dados sintéticos ou ciclos externos. Reaprender os limites em cada treino interno e no refit externo; não derivar limites únicos usando os 54 SN antes do CV.

Limites uniformes de controle: 1–41, 42–82 e 83–123. Não usar picos de um ciclo externo para adaptar sua segmentação. A identidade do SN é usada para ponderação e separação, nunca como atributo.

## Dicionário fixo e braços

Em cada região calcular 14 descritores: média, mediana, desvio, quartis 25/75, mínimo, máximo, amplitude, RMS, inclinação por índice, diferença final–inicial, média e máximo das diferenças absolutas e fração de diferenças consecutivas ≤1e-9. Acrescentar diferenças de média e desvio para cada um dos três pares de regiões: **48 atributos**. Nenhuma grandeza derivada será chamada velocidade/aceleração/energia mecânica.

| Braço | Representação | Contraste |
|---|---|---|
| M0_F246 | F246 e scores B1 históricos verificados | Referência de continuidade sem refits quando equivalência comprovada. |
| M1_uniform | 48 atributos com limites 41/82 | Controle compacto. |
| M2_empirical | Mesmos 48 atributos com limites aprendidos no treino | M2−M1 testa segmentação numérica, não alinhamento físico validado. |
| M3_combined | F246 + 48 atributos empíricos (294) | Informação incremental, com redundância e dimensionalidade maiores explicitadas. |

Reaproveitar matrizes F246 e predições B1 apenas após verificar hashes/roles/metadados dos caches E1. Regerar sinais sintéticos com o mesmo gerador, IDs, classes e sementes de cada cache e conferir correspondência com metadados salvos antes de extrair novos atributos. Não introduzir nova augmentation. Imputar atributos novos com mediana aprendida no treino expandido correspondente. Os sinais real/sintético são os mesmos entre M1/M2/M3; o alinhamento usa só reais.

## Seleção, orçamento e execução

Usar `triage_policy.curve/select_curve/predict/evaluate` de E10. Limiares dos nove cenários são fixados usando scores internos; avaliar externamente apenas depois. M0 usa probabilidades verificadas com a mesma ordem dos metadados; cotejar resultado com B1/E10. Salvar todos os scores internos/externos e decisões, não somente o cenário favorável.

Orçamento: **900 ajustes novos** = 3 braços × 50 partições × (5 ajustes internos + 1 refit), mais reutilização certificada de M0. Sem busca de hiperparâmetros. Piloto técnico seed42/outer0 integra esse orçamento; não calcular métricas externas no piloto nem alterar segmentos/descritores a partir de resultados externos. Medir tempo e memória para projetar a rodada, até quatro processos e quatro threads por processo. Parar ao completar o orçamento, preservando falhas técnicas com emendas e sem ampliar a busca por resultado desfavorável.

## Saídas e interpretação

Protocolos/fontes/ambiente/entradas congelados; limites e objetivos por treino; probabilidades e metadados por partição; seleções internas; decisões e métricas externas por cenário/semente/SN; admissibilidade interna e atendimento externo; diferenças pareadas e frequências de limites. Sementes não são réplicas independentes; relato descritivo, sem prova de superioridade ou mecanismo.

Ganho M2−M1 distribuído entre SN favorece localização empírica nesse desenho; ganho só em M3 não isola alinhamento. Ganho só após exclusão externa não será aceito como ganho geral. Resultado nulo/contrário encerra esse contraste, sem identificar uma fase por conveniência. Mesmo atendimento de recall/FA0 nesta população não é validação externa nem aceitação operacional. Inferência física continua limitada pela variável e pelo mapa desconhecidos.
