# ProgConcT3

Programação Concorrente - SSC0143 - 2 Semestre de 2015
Prof. Dr. Júlio Cezar Estrella
Trabalho 3 - Smoothing de imagem utilizando CUDA

- Alunos:
	
	Thiago Ledur Lima		- 8084214
	Rodrigo Neves Bernardi	- 8066395	


- Compilação
	
-Para compilar é necessário ir à pasta onde se encontra o código fonte do programa e o arquivo Makefile. A compilação dos programas pode ser feita com o comando make:

		$ make 

- Execução

----------->>>> OBS: No Makefile, devem ser alterados os caminhos das imagens passadas por argumento. Alterados os caminhos das imagens, basta seguir os proximos passos:	


-Para executar a versão sequencial do programa, utilize o comando:
	
		$ make seq

-Para executar a versão paralela OpenMP+OpenMPI do programa no cluster Halley, utilize o comando:
	
		$ make par	

-Para executar a versão paralela em CUDA do programa no cluster Halley, utilize o comando:
	
		$ make cud
