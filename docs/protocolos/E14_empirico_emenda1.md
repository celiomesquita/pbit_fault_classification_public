# E14.e — emenda 1: escopo do long congelado

29/10/2026, antes de qualquer ajuste E14.e. O arquivo `long_primary.csv` congelado contém toda a população auditada, enquanto `train_primary.csv` contém apenas os 1.195 ciclos de desenvolvimento. A asserção inicial do piloto v1 detectou a diferença antes da aprendizagem de regiões e do primeiro classificador.

Filtrar explicitamente o long pelos IDs do treino primário antes de construir a matriz. A função histórica de augmentation já recebe IDs de treino; a correspondência completa dos metadados sintéticos continua exigida. Nenhum sinal externo foi usado para aprender limites, atributos ou modelos. Preservar freeze/execução interrompida em `data/e14_empirical_v1/` e usar novo congelamento em `data/e14_empirical_v2/`. Não mudar braços, objetivos, orçamento ou métricas.
