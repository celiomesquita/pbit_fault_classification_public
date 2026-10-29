# E13 — emenda 2: separador entre metadados e medições

29/10/2026. A execução v2 interrompeu na asserção de correspondência bruto-original, antes do sumário e de qualquer treino. A inspeção do formato revelou uma linha separadora vazia: metadados nas linhas 1–3, separador na linha 4 e 123 medições nas linhas 5–127. O parser novo E13 havia lido linhas 4–126, deslocando uma posição. O long bruto não foi alterado e essa falha não demonstra inconsistência nos dados.

Corrigir o parser E13 para linhas 5–127, exigir o separador vazio e o total esperado, e acrescentar teste da primeira/última medição. Essa disposição também é documentada no leitor Orange existente (`preparar_pbit_orange.py:36-37`). Preservar v2 como execução interrompida e rodar com novo congelamento em `data/e13_quality_v3/`. As regras de qualidade não mudam. O erro estava na rotina nova de verificação; não afeta E10/E12 nem altera sinais/labels.
