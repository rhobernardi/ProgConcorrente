# Especificação do Segundo Trabalho

# Smooth

- Especificação do Problema

A especificação completa e as dicas para a resolução do problema encontram-se anexados aqui no moodle.

- Obrigatório

    Desenvolver uma versão sequencial e outra paralela
    Na versão paralela utilizar OpenMP + OpenMPI

    Identificar e adotar os tipos de decomposição para o problema (ver Seção 3.2 do livro Introduction to Parallel Computing, Second Edition)
        Justificar a utilização de um ou vários tipos de decomposição
        Criar diagramas de decomposição de dados

    Especificar o mapeamento utilizado na implementação e justificar (ver Seção 3.4 do livro Introduction to Parallel Computing, Second Edition)
    Criar diagramas ou figuras explicativas mostrando como é o mapeamento na implementação

- Como o grupo será avaliado?

Será levado em consideração na avaliação:

    As técnicas de decomposição e mapeamento escolhidas
    A eficiência dos algoritmos utilizados para resolver o problema
    Análise crítica da aplicação desenvolvida, citando vantagens e desvantagens das opções de projeto feitas, em relação ao desempenho, portabilidade, flexibilidade, escalabilidade e granulação
    A forma de apresentação dos resultados no relatório e a escrita (utilização de referências)

- Códigos

    Tanto a versão sequencial quanto a versão paralela devem ser executadas no Cluster Cosmos. No moodle há informações de como utilizar o cluster Cosmos.
    Fazer um comparativo entre os algoritmos e entre as versões sequenciais e paralelas de cada um deles (Calcular o Speedup).
    Apresentar resultados somente em gráficos ou tabelas:
        Tempo de execução seqüencial e tempo de execução paralelo
        Speedup

    Realizar no mínimo 10 execuções e calcular o tempo de execução considerando, média e desvio padrão.
    Considerar a evolução dos resultados quando ocorre aumento ou diminuição de threads, processos, número de nós do cluster, etc.
    Ao enviar o link do repositório com os códigos, seguir o padrão:
        Trab02-GrupoX-Y
            Onde X é o número do grupo: 01, 02, ..., n
            Onde Y é a indicação da turma: A, B, Pos

- Para o Relatório

    O relatório deverá ser entregue SOMENTE no formato PDF com o seguinte nome: Relatorio2-grupoX-turmaY, onde X é o número do grupo e Y representa a turma(A, B ou PosGrad).

    Um relatório apresentando uma introdução sobre os algoritmos, os resultados obtidos e as soluções. O relatório deve ter no mínimo 6 páginas e no máximo 12 páginas desconsiderando capas, indices e bibliografia.

    Discuta as soluções, as dificuldades, os resultados obtidos, o hardware utilizado, a metodologia de execução dos experimentos, etc.

    O relatório deve ser enviado via Moodle conforme combinado no primeiro dia de aula

    O relatório deve apresentar a forma de execução dos codigos (README)

- Avaliação

A avaliação deve considerar:

    Qualidade do relatório e descrição dos resultados com base na execução dos códigos.
    Saída correta dos códigos bem como a qualidade do mesmo (comentários, identação, menus de utilização, etc).

- Imagens

Considerar as Imagens do HiRISE

http://www.uahirise.org/katalogos.php (Preferencialmente)

http://marsoweb.nas.nasa.gov/HiRISE/

    Deve ser utilizado tanto imagens em Grayscale quanto Color
    Os grupos podem escolher outras imagens. 
    Considerar as categorias de Imagens
        Pequena: 10MB <= i <= 50MB
        Média: 50MB < i <= 100MB
        Grande: 100MB < i <= 500MB
        Enorme: 500MB < i <= 1GB

- Horário para Utilização do Cluster Cosmos

Os grupos podem fazer uso do cluster em duas situações:

    Enquanto desenvolvem o trabalho, não há necessidade de agendamento. Ou seja, enquanto desenvolvem os códigos, testam as soluções, etc.
    É obrigatório agendar um horário para testar a solução final, e coletar os dados que vão para o relatório. Cada grupo tem direito à 3h de utilização a partir do momento que realizar o agendamento para utilizar o cluster. As 3h não precisam ser contínuas. Para o horário agendado, somente um grupo poderá utlizar o cluster naquele instante. Ao solicitar o agendamento, considerar no formulário:

Subject: Scheduling for using (Andromeda, Cosmos, Halley)

First Name: GrupoX (onde X é o número do grupo)

LastName: TurmaY (onde Y é A, B ou Pos)

Advisor: Other

Para realizar o agendamento, acessar o link abaixo:

http://infra.lasdpc.icmc.usp.br/contact

- Observações

    Ao enviar o relatório para o Moodle, teste se não houve problema ao fazer download.
    Relatório fora do padrão será penalizado 
    Considera-se fora do padrão, relatórios sem:
        Nome completo dos integrantes, N. USP, capa, índice, etc.

- Prazos

Relatório e Códigos

	01/11/15 - 23:55h (Turmas A, B e PosGrad)