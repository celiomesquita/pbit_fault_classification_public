# E1 — Expansão computacional às 50 divisões

Este complemento é fixado depois do piloto 42/0 e antes dos 49 treinos restantes.
Preservar `docs/protocolos/E1.md`, seu congelamento e todos os resultados do piloto.
Não altera modelos, sementes, divisões, custos, limites ou regras de seleção.

## Informação já conhecida e alcance

O piloto fez 400 tentativas, 2.096 ajustes e levou 44,90 minutos. As verificações
passaram. Internamente, 7/7 políticas falharam no limite FA0 0,30; 6/7 em 0,40;
2/7 em 0,50. A referência falhou em 0,40. Estes resultados internos eram conhecidos
antes deste complemento. Não recalibrar o orçamento para eliminar essas falhas.
Não descrever o orçamento nominal como FA0 atingido por todas as políticas.
Desempenho externo ainda não foi calculado. A inferência permanece pendente do E3.

## Execução

Usar as cinco divisões externas para cada semente, na ordem
42, 0, 7, 13, 21, 1, 2, 3, 4, 5. Reutilizar somente o piloto completo e verificado
42/0: sua configuração é idêntica, excepto o nome do alcance computacional.
Guardar essa proveniência no manifesto; não copiar ou sobrescrever seus artefatos.

Uma divisão externa por vez, com os quatro estudos em paralelo e quatro threads
por modelo. Manter 100 tentativas por estudo, cinco folds internos, parâmetros,
pesos, amostras artificiais e regras de custo exactamente como no piloto.
As sementes de modelo, Optuna, aumento e MetaCost continuam dependentes de seed,
conforme `E1.md`. A duração estimada é 36,67 horas adicionais; não é uma garantia.

Congelar código, este complemento, protocolo original, versões e fontes antes
dos treinos. Conferir os hashes antes de cada divisão e de sua auditoria.
Não modificar as fontes congeladas durante a rodada. Resultados por divisão em
`data/e1_full_v1/partitions/seed_<seed>_outer_<fold>/attempt_<n>/`.

## Interrupção, retomada e conferência

Manter registros de progresso e logs em disco. Somente uma instância da fila.
Conferir cada divisão com os dois verificadores do piloto: fontes, hashes,
separação das SN, seleções internas e reprodução das probabilidades dos modelos.
Uma divisão conta como concluída somente depois dessas verificações.

Na primeira falha de treino ou auditoria, interromper a fila, preservar arquivos
e registrar o motivo. Não pular a divisão. Uma retomada explícita reaproveita
divisões completas cujos hashes passam; uma tentativa incompleta é preservada
e o mesmo fold começa numa nova pasta. Não alterar o algoritmo para ocultar falhas.

Registrar progresso em `progress.json` e `progress.md`; actualizar o bloco de
acompanhamento no plano nos marcos de execução. Este é um acompanhamento de
treino e conferência, sem calcular CMR/FA0 externos, testes ou intervalos.
Só depois das 50 divisões verificadas declarar a rodada computacional concluída.
E1 científico continua pendente da solução inferencial após E3 e da análise externa.
