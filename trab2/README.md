# ProgConcT2

Programação Concorrente - SSC0143 - 2 Semestre de 2015
Prof. Dr. Júlio Cezar Estrella
Trabalho 2 - Smoothing de imagem utilizando MPI+OMP

Alunos:
	Thiago Ledur Lima		- 8084214
	Rodrigo Neves Bernardi	- 8066395	


Compilação
	Para compilar é necessário ir à pasta onde se encontra o código fonte do programa e o arquivo Makefile. A compilação dos programas pode ser feita com o comando make:
		$ make 

Execução
	Para executar a versão sequencial do programa, utilize o comando make run1:
		$ make run1
	Para executar a versão paralela do programa localmente (“1 nó”), utilize o comando make run2:
		$ make run2
	Para executar a versão paralela do programa no cluster Cosmos, utilize o comando make run3:
		$ make run2