#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define FILTERDIM 5
#define FILTERSIZE 25
#define TILEWIDTH 16

typedef unsigned char uchar;

struct PPMstructure {
  char type[2];
  int width;
  int height;
  int maxVal;
  uchar *r;
  uchar *g;
  uchar *b;
};

typedef struct PPMstructure PPMImage;

/* Verifica erros */
void cudaCheck(cudaError_t error) {
    if(error != cudaSuccess) {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

/* Aloca memoria */
void allocData(PPMImage *img, char* type, int width, int height, int maxVal) {
    strcpy(img->type, type);
    img->width = width;
    img->height = height;
    img->maxVal = maxVal;

    img->r = (uchar *)malloc(width * height * sizeof(uchar));
    img->g = (uchar *)malloc(width * height * sizeof(uchar));
    img->b = (uchar *)malloc(width * height * sizeof(uchar));
}

/* Libera memoria */
void freeData(PPMImage *img) {
    free(img->r);
    free(img->g);
    free(img->b);
}

/* Cria estrutura PPMImage */
void readImage(PPMImage *imgIn, PPMImage *imgOut, char *in){
    FILE *input;
    int i;
    char type[2];
    int maxVal, width, height;

    input = fopen(in, "r");

    if (input == NULL)
        printf("error\n");
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
        for(i = 0; i < width * height ; i++) {
            fscanf(input, "%hhu %hhu %hhu", &(imgIn->r[i]), &(imgIn->g[i]), &(imgIn->b[i]));
        }
    }

    fclose(input);
}

/* Cria arquivo com o resultado */
void saveImage(PPMImage *img, char *out) {
    FILE *output;
    int i;

    // Escreve header
    output = fopen(out, "w");
    fprintf(output, "%s\n", img->type);
    fprintf(output, "%d %d\n", img->width, img->height);
    fprintf(output, "%d\n", img->maxVal);

    // Escreve dados
    for (i = 0; i < img->height * img->width; i++) {
        fprintf(output, "%hhu %hhu %hhu ", img->r[i], img->g[i], img->b[i]);
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

    // Restringe bordas
    if (x >= W-k|| y >= H-k|| x <= k-1 || y <= k-1) return;

    // Convolucao
    for(i = 0; i <= FILTERDIM - 1; i++){
        for(j = 0; j <= FILTERDIM - 1; j++){
           Pvalue += in[(idx - k + i) + W * (j - k)];
        }
    }
    out[idx] = (uchar)(Pvalue / FILTERSIZE);
}


int main(int argc, char const *argv[]) {
    PPMImage imgIn, imgOut;
    uchar *inR, *inG, *inB, *outR, *outG, *outB;
    int size;
    char in[20], out[20], tempo[20];

    // Tempo total
    struct timeval startTimeCudaTotal, endTimeCudaTotal;
    gettimeofday(&startTimeCudaTotal, NULL);

    strcpy(in, argv[1]);
    strcpy(out, argv[2]);
    strcpy(tempo, argv[3]);

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

    // Define grid e bloco
    dim3 gridDim(imgIn.width / TILEWIDTH + 1, imgIn.height / TILEWIDTH + 1);
    dim3 blockDim(TILEWIDTH, TILEWIDTH);

    // Tempo do smoothing
    struct timeval startTimeCuda, endTimeCuda;
    gettimeofday(&startTimeCuda, NULL);

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