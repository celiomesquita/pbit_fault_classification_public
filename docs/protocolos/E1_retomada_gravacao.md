# E1 — Retomada após falha de gravação do progresso

Emenda de implementação de 12/10/2026. Preservar os protocolos, fontes e manifesto
originais. Não muda treino, sementes, folds, tentativas, custos, limites ou seleção.

## Falha observada

A fila parou em 12/10/2026 às 05:34:58 UTC, com oito divisões verificadas.
O treino 0/3 terminou suas 400 tentativas e 2.096 ajustes. A conferência de
21 decisões e 90 hashes passou, mas faltava reproduzir os modelos finais.
Windows devolveu WinError 5 ao substituir `progress.json.tmp` por `progress.json`.
Não foi identificada a origem exacta da recusa de acesso; não atribuir a uma
aplicação específica. O problema observado foi na gravação do progresso.

## Correção e recuperação

Adicionar uma rotina de gravação em arquivo temporário de nome único, no mesmo
diretório, com substituição atómica e até 20 tentativas perante PermissionError.
As esperas são limitadas a um segundo cada. Se a recusa persistir, a exceção
continua visível e a fila para; não truncar o arquivo de estado anterior.

Manter `scripts/run_e1_full.py` e seu hash original. A nova entrada
`scripts/run_e1_full_resume.py` substitui somente a função de gravação no processo
da fila e acrescenta a conferência dos hashes desta emenda. Os processos de
treino e os dois verificadores não são modificados.

Completar a reprodução dos modelos 0/3, conferir seu manifesto e reaproveitar
o treino completo, sem novo ajuste. A fila original adopta essa divisão somente
após os dois verificadores passarem. Os oito resultados anteriores continuam
sujeitos à conferência de hashes na retomada. Seguir pela divisão 0/4, sem saltos.

Preservar o estado da interrupção e os logs anteriores em `resume_001_snapshots/`;
congelar código, testes e esta emenda em `resume_001_freeze.json` antes de retomar.
Os logs da retomada recebem nomes novos. Testar recusa temporária, recusa
persistente e bloqueio real de leitura no Windows. O plano deve registrar a
interrupção e a retomada. A análise externa e a inferência após E3 continuam pendentes.
