/**
*	Programação Concorrente - SSC0143 - 2 Semestre de 2015
*	Prof. Dr. Júlio Cezar Estrella
*	Trabalho 2 - Smoothing de imagem utilizando MPI+OMP
*  
*	Alunos:
*		Thiago Ledur Lima		- 8084214
*		Rodrigo Neves Bernardi	- 8066395	
**/

#ifndef _SMOOTHING_H_
#define _SMOOTHING_H_

typedef struct RGBstructure {
	int r;
	int g;
	int b;
} RGB;

typedef struct PPMstructure {
	char type[3];
	int width;
	int height;
	int maxVal;
	RGB **data;
} PPMImage;

void allocData (PPMImage *, char *, int, int, int);
void freeData (PPMImage *);
void readImage (PPMImage *, PPMImage *, char *);
void saveImage (PPMImage *, char *);
void colorFilter (PPMImage *, PPMImage *, int, int, int, int);

#endif