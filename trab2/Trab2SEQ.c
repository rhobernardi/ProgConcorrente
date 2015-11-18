/**
*	Programação Concorrente - SSC0143 - 2 Semestre de 2015
*	Prof. Dr. Júlio Cezar Estrella
*	Trabalho 2 - Smoothing de imagem (versão sequencial)
*  
*	Alunos:
*		Thiago Ledur Lima		- 8084214
*		Rodrigo Neves Bernardi	- 8066395	
**/

#include "Smoothing.c"
#include <time.h>

int main (int argc, char **argv) {
	int i, j;
	PPMImage imgIn, imgOut;
	char in[20], out[20];
	struct timespec clockStart, clockEnd;

	strcpy(in, "image.ppm");
	strcpy(out, "out_image.ppm");

	// lê imagem
	readImage(&imgIn, &imgOut, in);

	//com tudo preparado, começa a contar tempo de execução do método
	clock_gettime(CLOCK_MONOTONIC, &clockStart);

	// roda o filtro
	for (i = 0; i < imgIn.height; i++) {
		for (j = 0; j < imgIn.width; j++) {
			colorFilter (&imgIn, &imgOut, i, j, i, j);
		}
	}

	//termina a contagem do tempo de execução do método
	clock_gettime(CLOCK_MONOTONIC, &clockEnd);
	
	//opcional pra testar: printa tempo
	printf("Tempo: %.5lfs\n", ( ((double)(clockEnd.tv_nsec - clockStart.tv_nsec)/1000000000) + (clockEnd.tv_sec - clockStart.tv_sec) ) );

	// salva imagem
	saveImage(&imgOut, out);
	// libera memória
	freeData(&imgIn);
	freeData(&imgOut);  

	return 0;
}