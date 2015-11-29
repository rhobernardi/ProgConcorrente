/**
*   Programação Concorrente - SSC0143 - 2 Semestre de 2015
*   Prof. Dr. Júlio Cezar Estrella
*   Trabalho 3 - Smoothing de imagem utilizando CUDA
*  
*   Alunos:
*       Thiago Ledur Lima       - 8084214
*       Rodrigo Neves Bernardi  - 8066395   
**/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include "Smoothing_CUDA.h"
#include <sys/time.h>

#define FILTERDIM 5
#define FILTERSIZE 25
#define TILEWIDTH 16

typedef unsigned char uchar;

/*typedef struct RGBstructure {
    uchar *r;
    uchar *g;
    uchar *b;
} RGB;*/

typedef struct PPMstructure {
    char type[3];
    int width;
    int height;
    int maxVal;
    //RGB *data
    uchar *r;
    uchar *g;
    uchar *b;
} PPMImage;

// verifica erros
void cudaCheck(cudaError_t error) {
    if(error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

// aloca memória necessária para uma PPMImage, recebendo suas informações
void allocData(PPMImage *img, char* type, int width, int height, int maxVal) {
    strcpy(img->type, type);
    img->width = width;
    img->height = height;
    img->maxVal = maxVal;

    // img->data->r = (uchar *)malloc(width * height * sizeof(uchar));
    // img->data->g = (uchar *)malloc(width * height * sizeof(uchar));
    // img->data->b = (uchar *)malloc(width * height * sizeof(uchar));

    img->r = (uchar *)malloc(width * height * sizeof(uchar));
    img->g = (uchar *)malloc(width * height * sizeof(uchar));
    img->b = (uchar *)malloc(width * height * sizeof(uchar));
}

// libera memória usada por uma PPMImage
void freeData(PPMImage *img) {
    // free(img->data->r);
    // free(img->data->g);
    // free(img->data->b);

    free(img->r);
    free(img->g);
    free(img->b);    
}

// cria uma estrutura PPMImage, incluindo a leitura do arquivo e a chamada da função para alocar a imagem na memória
void readImage(PPMImage *imgIn, PPMImage *imgOut, char *in){
    FILE *input;
    int i;
    char type[3];
    int maxVal, width, height;

    input = fopen(in, "r");

    if (input == NULL)
        printf("Can't read image file.\n");
    else {
        // le tipo
        fscanf(input, "%s", type);

        // compara tipo para ver se é ppm (imagem colorida)
        //if(strcmp(type,"P3\n") == 0)
        //{
            printf("======================TYPE: %s\n", type);
            // pula fim da linha
            while (getc(input) != '\n');

            // pula comentario da linha
            while (getc(input) == '#') {
                while (getc(input) != '\n');
            }

            // volta um caracter
            fseek(input, -1, SEEK_CUR);

            // le dimensões da imagem e a escala das cores
            fscanf(input, "%d", &width);
            fscanf(input, "%d", &height);
            fscanf(input, "%d", &maxVal);

            // aloca as matrizes width x height das imagens de entrada e saída na memória
            allocData(imgIn, type, width, height, maxVal);
            allocData(imgOut, type, width, height, maxVal);
            
            // le dados do arquivo
            for(i = 0; i < width * height ; i++) {
                // fscanf(input, "%hhu %hhu %hhu", &(imgIn->data->r[i]), &(imgIn->data->g[i]), &(imgIn->data->b[i]));
                fscanf(input, "%hhu %hhu %hhu", &(imgIn->r[i]), &(imgIn->g[i]), &(imgIn->b[i]));        
            }
        //}

        // se nao, imagem é pgm (preto e branco)
        /*else*/ if(strcmp(type,"P2\n") == 0)
        {
            // codigo para imagem preto e branco
        }
    }

    fclose(input);
}

// cria um arquivo com o resultado guardado na matriz de imagem de saída
void saveImage(PPMImage *img, char *out) {
    FILE *output;
    int i;

    // escreve header contendo tipo, comentário, dimensões e escala
    output = fopen(out, "w");
    fprintf(output, "%s\n", img->type);
    fprintf(output, "#imagem com smooth\n");
    fprintf(output, "%d %d\n", img->width, img->height);
    fprintf(output, "%d\n", img->maxVal);

    // escreve dados da imagem
    for (i = 0; i < img->height * img->width; i++) {
        //fprintf(output, "%d %d %d\t", img->data[i][j].r, img->data[i][j].g, img->data[i][j].b);
        // fprintf(output, "%hhu %hhu %hhu ", img->data->r[i], img->data->g[i], img->data->b[i]);
        fprintf(output, "%hhu\n", img->r[i]);
        fprintf(output, "%hhu\n", img->g[i]);
        fprintf(output, "%hhu\n", img->b[i]);
    }

    fclose(output);
}

__global__ void filter(uchar *in, uchar *out, int H, int W) {
    int i, j, k;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = y * W + x;
    float Pvalue = 0;

    k = FILTERDIM / 2;

    // restringe as bordas da imagem
    if (x >= W-k|| y >= H-k|| x <= k-1 || y <= k-1) return;

    // aplica convolucao
    for(i = 0; i <= FILTERDIM - 1; i++){
        for(j = 0; j <= FILTERDIM - 1; j++){
           Pvalue += in[(idx - k + i) + W * (j - k)];
        }
    }
    out[idx] = (uchar)(Pvalue / FILTERSIZE);
    printf("\nFILTER FINISHED IN : %d %d\n", x, y);
}


int main(int argc, char const *argv[]) {
    PPMImage imgIn, imgOut;
    uchar *inR, *inG, *inB, *outR, *outG, *outB;
    int size;
    char in[20], out[20], tempo[20];

    // Tempo total
    struct timeval startTimeCudaTotal, endTimeCudaTotal;
    gettimeofday(&startTimeCudaTotal, NULL);

    strcpy(in, "image.ppm");
    strcpy(out, "out_image.ppm");
    strcpy(tempo, "time.txt");

    // Le a imagem de entrada
    readImage(&imgIn, &imgOut, in);

    size = imgIn.width * imgIn.height;

    // Tempo com a alocacao de memoria
    struct timeval startTimeCudaMem, endTimeCudaMem;
    gettimeofday(&startTimeCudaMem, NULL);

    cudaCheck(cudaMalloc((void**)&inR, size * sizeof(uchar)));
    cudaCheck(cudaMalloc((void**)&inG, size * sizeof(uchar)));
    cudaCheck(cudaMalloc((void**)&inB, size * sizeof(uchar)));
    cudaCheck(cudaMalloc((void**)&outR, size * sizeof(uchar)));
    cudaCheck(cudaMalloc((void**)&outG, size * sizeof(uchar)));
    cudaCheck(cudaMalloc((void**)&outB, size * sizeof(uchar)));

    cudaCheck(cudaMemcpy(inR, imgIn.r, size * sizeof(uchar), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(inG, imgIn.g, size * sizeof(uchar), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(inB, imgIn.b, size * sizeof(uchar), cudaMemcpyHostToDevice));

    // define grid e bloco
    dim3 gridDim(imgIn.width / TILEWIDTH + 1, imgIn.height / TILEWIDTH + 1);
    dim3 blockDim(TILEWIDTH, TILEWIDTH);

    // tempo do smoothing
    struct timeval startTimeCuda, endTimeCuda;
    gettimeofday(&startTimeCuda, NULL);

    // aplica o filter como funcao global usando grid e bloco
    filter<<<gridDim, blockDim>>>(inR, outR, imgIn.height, imgIn.width);
    filter<<<gridDim, blockDim>>>(inG, outG, imgIn.height, imgIn.width);
    filter<<<gridDim, blockDim>>>(inB, outB, imgIn.height, imgIn.width);

    gettimeofday(&endTimeCuda, NULL);

    cudaCheck(cudaMemcpy(imgOut.r, outR, size * sizeof(uchar), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(imgOut.g, outG, size * sizeof(uchar), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(imgOut.b, outB, size * sizeof(uchar), cudaMemcpyDeviceToHost));

    gettimeofday(&endTimeCudaMem, NULL);

    // Salva imagem filtrada
    saveImage(&imgOut, out);

    // Libera memoria
    cudaFree(inR);
    cudaFree(inB);
    cudaFree(inG);
    cudaFree(outR);
    cudaFree(outG);
    cudaFree(outB);

    freeData(&imgIn);
    freeData(&imgOut);

    gettimeofday(&endTimeCudaTotal, NULL); 

    // Calcula o tempo
    double result = endTimeCuda.tv_sec - startTimeCuda.tv_sec + (endTimeCuda.tv_usec - startTimeCuda.tv_usec) / 1000000.0;
    //double resultMem = endTimeCudaMem.tv_sec - startTimeCudaMem.tv_sec + (endTimeCudaMem.tv_usec - startTimeCudaMem.tv_usec) / 1000000.0;
    //double resultTotal = endTimeCudaTotal.tv_sec - startTimeCudaTotal.tv_sec + (endTimeCudaTotal.tv_usec - startTimeCudaTotal.tv_usec) / 1000000.0;

    printf("GPU: %lfs\n", result);
    //printf("GPU + memoria + in/out - TOTAL : %lfs\n", resultTotal);
    
    FILE *t;
    t = fopen(tempo,"w");
    fprintf(t,"GPU: %lfs\n", result);
    fclose(t);

    return 0;
}