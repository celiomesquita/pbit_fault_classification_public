# E13 — emenda 1: tokens de erro na exportação original

29/10/2026. A primeira execução global, preservada em `data/e13_quality_v1/`, interrompeu na conferência do CSV original: `float()` não aceita o token `#VALUE!`. Nenhum ajuste de classificador foi realizado. As tabelas parciais dessa execução não constituem auditoria concluída.

Antes de conferir a concordância com o original ou produzir o sumário final, alterar o parser para converter tokens vazios/não numéricos em ausências, preservar contagens e conteúdo dos tokens por ciclo e comparar as máscaras de ausência. Nunca converter erro em zero. A consistência bruto-original continua exigindo igualdade das máscaras e diferenças finitas ≤ 1e-12.

O parser ganhou teste específico. A nova execução será gravada em `data/e13_quality_v2/`, com novo congelamento e cópia de fontes. As regras de sinalização permanecem as mesmas. A emenda foi motivada por falha técnica de leitura, não por desempenho de modelos ou tentativa de alterar a frequência de exclusões. Tokens de erro são evidência de problema na representação/exportação; não demonstram falha física do canal.
