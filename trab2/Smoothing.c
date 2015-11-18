/**
*	Programação Concorrente - SSC0143 - 2 Semestre de 2015
*	Prof. Dr. Júlio Cezar Estrella
*	Trabalho 2 - Smoothing de imagem utilizando MPI+OMP
*  
*	Alunos:
*		Thiago Ledur Lima		- 8084214
*		Rodrigo Neves Bernardi	- 8066395	
**/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Smoothing.h"

// aloca memória necessária para uma PPMImage, recebendo suas informações
void allocData(PPMImage *img, char* type, int width, int height, int maxVal) {
	int i;

	// seta o tipo, largura, altura e escala das cores
	strcpy(img->type, type);
	img->width = width;
	img->height = height;
	img->maxVal = maxVal;

	// aloca a matriz width x heigth
	img->data = (RGB **) malloc(height * sizeof(RGB*));
	for (i = 0; i < height; i++) {
		img->data[i] = (RGB *)malloc(width * sizeof(RGB));
	}
}

// libera memória usada por uma PPMImage
void freeData(PPMImage *img) {
	int i;
	for (i = 0; i < img->height; i++) {
		free(img->data[i]);
	}
	free(img->data);
}

// cria uma estrutura PPMImage, incluindo a leitura do arquivo e a chamada da função para alocar a imagem na memória
void readImage(PPMImage *imgIn, PPMImage *imgOut, char *in) {
	FILE *input;
	int i, j, ch;
	char type[3];
	int maxVal, width, height;

	input = fopen(in, "r");
	if (input == NULL){
		printf("Can't read image file.\n");
	}
	else {
		// lê tipo
		fscanf(input, "%s", type);
		// pula fim da linha
		while (getc(input) != '\n');

		// pula comentário de linha
		while (getc(input) == '#') {
			while (getc(input) != '\n');
		}

		// volta um caracter
		fseek(input, -1, SEEK_CUR);

		// lê dimensões da imagem e a escala das cores
		fscanf(input, "%d", &width);
		fscanf(input, "%d", &height);
		fscanf(input, "%d", &maxVal);

		// aloca as matrizes width x height das imagens de entrada e saída na memória
		allocData(imgIn, type, width, height, maxVal);
		allocData(imgOut, type, width, height, maxVal);

		// lê dados do arquivo
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				fscanf(input, "%d", &ch);
				imgIn->data[i][j].r = ch;
				fscanf(input, "%d", &ch);
				imgIn->data[i][j].g = ch;
				fscanf(input, "%d", &ch);
				imgIn->data[i][j].b = ch;
			}
		}
	}
	fclose(input);
}

// cria um arquivo com o resultado guardado na matriz de imagem de saída
void saveImage(PPMImage *img, char *out) {
	FILE *output;
	int i, j;

	// escreve header contendo tipo, comentário, dimensões e escala
	output = fopen(out, "w");
	fprintf(output, "%s\n", img->type);
	fprintf(output, "#imagem com smooth\n");
	fprintf(output, "%d %d\n", img->width, img->height);
	fprintf(output, "%d\n", img->maxVal);

	// escreve dados da imagem
	for (i = 0; i < img->height; i++) {
		for (j = 0; j < img->width; j++) {
			//fprintf(output, "%d %d %d\t", img->data[i][j].r, img->data[i][j].g, img->data[i][j].b);
			fprintf(output, "%d\n", img->data[i][j].r);
			fprintf(output, "%d\n", img->data[i][j].g);
			fprintf(output, "%d\n", img->data[i][j].b);

		}
	}
	fclose(output);
}

// aplica o filtro de smoothing de tamanho 5 pixel a pixel - aplica para a posição (x, y)
void colorFilter (PPMImage *imgIn, PPMImage *imgOut, int x, int y, int sx, int sy) {
	float somaR, somaG, somaB;
	somaR = somaG = somaB = 0;
	int i, j;
	// dado um ponto x, y, pega os 24 pontos no quadrado em sua volta mais ele mesmo (5x5) para fazer a média
	for (i = (x-2); i < (x+3); i++) {
		// verificação dos limites verticais da matriz
		if (i >= 0 && i < imgIn->height) {
			for (j = (y-2); j < (y+3); j++) {
				// verificação dos limites horizontais da matriz
				if (j >= 0 && j < imgIn->width) {
					somaR += imgIn->data[i][j].r;
					somaG += imgIn->data[i][j].g;
					somaB += imgIn->data[i][j].b;
				}
			}
		}
	}
	// divide por 25, achando a média. No caso das bordas, os pixeis inexistentes
	// não foram somados, portanto o valor simbólico deles é zero
	imgOut->data[sx][sy].r = (int)somaR/25;
	imgOut->data[sx][sy].g = (int)somaG/25;
	imgOut->data[sx][sy].b = (int)somaB/25;
}