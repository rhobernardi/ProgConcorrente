#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define FILTERDIM 5   //dimensao do filtro. Nesse caso, 5x5.
#define FILTERSIZE 25 //valor da divisao para a media. Nesse caso, 25.

#define INPUT "400x855.ppm"
#define OUTPUT "out.ppm"
#define TIMER "tempo.txt"


typedef struct RGBstructure {
	int r;
	int g;
	int b;
}RGB;


typedef struct PPMstructure {
	char type[2];
	int width;
	int height;
	int maxVal;
	RGB **data;
}PPMImage;


void allocData(PPMImage *, char *, int, int, int);
void freeData(PPMImage *);
void readImage(PPMImage *, PPMImage *, char *);
void saveImage(PPMImage *, char *);
void filter(PPMImage *, PPMImage *);



int main(int argc, char *argv[]) 
{ 
	PPMImage imgIn, imgOut;
	char in[20], out[20], tempo[20];

	// Tempo total
	struct timeval startTimeSeqTotal, endTimeSeqTotal;
	gettimeofday(&startTimeSeqTotal, NULL);

	strcpy(in, INPUT); strcpy(out, OUTPUT); strcpy(tempo, TIMER);

	// Le a imagem de entrada
	readImage(&imgIn, &imgOut, in);

	// Tempo do smoothing
	struct timeval startTimeSeq, endTimeSeq;
	gettimeofday(&startTimeSeq, NULL);

	// Roda o filtro
	filter(&imgIn, &imgOut);

	gettimeofday(&endTimeSeq, NULL);

	// Salva imagem filtrada
	saveImage(&imgOut, out);

	// Libera memoria
	freeData(&imgIn);
	freeData(&imgOut);  

	gettimeofday(&endTimeSeqTotal, NULL); 

	// Calcula o tempo
	double result = endTimeSeq.tv_sec - startTimeSeq.tv_sec + (endTimeSeq.tv_usec - startTimeSeq.tv_usec) / 1000000.0;

	printf("smoothing time: %lfs\n", result);

	FILE *t;
	t = fopen(tempo,"w");
	fprintf(t,"smoothing time: %lfs\n", result);
	fclose(t);


	return 0;
}


/* Aloca memoria */
void allocData(PPMImage *img, char* type, int width, int height, int maxVal) {
	int i;

	strcpy(img->type, type);
	img->width = width;
	img->height = height;
	img->maxVal = maxVal;

	img->data = (RGB **) malloc(height * sizeof(RGB*));
	for (i = 0; i < height; i++) {
		img->data[i] = (RGB *)malloc(width * sizeof(RGB));
	}
}

/* Libera memoria */
void freeData(PPMImage *img) {
	int i;

	for (i = 0; i < img->height; i++) {
		free(img->data[i]);
	}

	free(img->data);
}

/* Cria estrutura PPMImage */
void readImage(PPMImage *imgIn, PPMImage *imgOut, char *in) {
	FILE *input;
	int i, j, ch;
	char type[2];
	int maxVal, width, height;

	input = fopen(in, "r");

	if (input == NULL)
		printf("Can't read image file.\n");
	else {
		/* Le header */
		// Le tipo
		fscanf(input, "%s", type);
		// Pula fim da linha
		while (getc(input) != '\n');

		// Pula comentario
		while (getc(input) == '#') {
			while (getc(input) != '\n');
		}

		// Volta um caracter
		fseek(input, -1, SEEK_CUR);

		// Le tamanho
		fscanf(input, "%d", &width);
		fscanf(input, "%d", &height);
		fscanf(input, "%d", &maxVal);

		allocData(imgIn, type, width, height, maxVal);
		allocData(imgOut, type, width, height, maxVal);
		
		// Le dados
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

/* Cria arquivo com o resultado */
void saveImage(PPMImage *img, char *out) {
	FILE *output;
	int i, j;

	// Escreve header
	output = fopen(out, "w");
	fprintf(output, "%s\n", img->type);
	fprintf(output, "%d %d\n", img->width, img->height);
	fprintf(output, "%d\n", img->maxVal);

	// Escreve dados
	for (i = 0; i < img->height; i++) {
		for (j = 0; j < img->width; j++) {
			fprintf(output, "%d %d %d\t", img->data[i][j].r, img->data[i][j].g, img->data[i][j].b);
		}
		fprintf(output, "\n");
	}

	fclose(output);
}

void filter(PPMImage *imgIn, PPMImage *imgOut) {
	int x, y, s, t, a;
	float filter[FILTERDIM][FILTERDIM];
	// Cria filtro de media
	for (x = 0; x < FILTERDIM; x++) {
		for(y = 0; y < FILTERDIM; y++) {
			filter[x][y] = 1.0/(float)(FILTERSIZE);
		}
	}

	// Valor para controle do indice do filtro
	a = (FILTERDIM-1)/2;
	
	// 2  fors percorrem cada ponto da imagem
	for (x = a; x < imgIn->height - a; x++) {
		for(y = a; y < imgIn->width - a; y++) {
			float newvalueR = 0, newvalueG = 0, newvalueB = 0;
			// 2 fors internos  percorrem a vizinhanca de (x,y)
			for (s = -a; s <= a; s++) {
				for (t = -a ; t <= a; t++) {
					newvalueR += imgIn->data[x+s][y+t].r;
					newvalueG += imgIn->data[x+s][y+t].g;
					newvalueB += imgIn->data[x+s][y+t].b;
				}   
			}
			// armazena na imagem de saida  
			imgOut->data[x][y].r = (int)(newvalueR / FILTERSIZE);
			imgOut->data[x][y].g = (int)(newvalueG / FILTERSIZE);
			imgOut->data[x][y].b = (int)(newvalueB / FILTERSIZE);

		}
	}
} 


