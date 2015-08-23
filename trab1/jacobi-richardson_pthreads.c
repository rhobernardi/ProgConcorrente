#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

//estrutura de dados de cada thread
typedef struct tdata {
	int tid;
	double **matriz;
	double *vetorAux;
	double *vetorKpp;
} tdata;

//printa matriz
void printaMat (double **mat, int ordem) {
	int i, j;
	printf ("\n");
	for (i = 0; i < ordem; i++) {
		for (j = 0; j < ordem; j++) {
			printf("%.1f  ", mat[i][j]);
		}
		printf ("\n");
	}
	printf ("\n");
	return;
}

//printa vetor
void printaVec (double *vec, int size) {
	int i;
	printf ("\n");
	for (i = 0; i < size; i++) {
		printf("x(%d)=%.5f\t", i, vec[i]);
	}
	printf ("\n\n");
	return;
}

int main (int argc, char *argv[]) {
	pthread_t *thread;
	tdata *thread_data;
	int i, j, j_order, j_row_test, j_ite_max, iteracao = 0;
	double j_error, conta, norma1, norma2, erroRel;
	double **matriz, *vetorAux, *vetorKpp, *vetorK;

	//entrada dos dados
	scanf("%d", &j_order);
	scanf("%d", &j_row_test);
	scanf("%lf", &j_error);
	scanf("%d", &j_ite_max);

	//aloca matriz
	matriz = (double**) malloc (j_order * sizeof(double*) );
	for (i = 0; i < j_order; i++) {
		matriz[i] = (double *) malloc (j_order * sizeof(double) );
	}
	//le dados da matriz
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			scanf("%lf", &matriz[i][j]);
		}
	}

	//aloca vetor
	vetorAux = (double*) malloc (j_order * sizeof(double));
	vetorKpp = (double*) malloc (j_order * sizeof(double));
	vetorK = (double*) malloc (j_order * sizeof(double));
	//le dados do vetor
	for (i = 0; i < j_order; i++) {
		scanf("%lf", &vetorAux[i]);
	}

	//primeira parte do método, encontrando B* (modificando a matriz) e x(n)^(0) (modificando o vetor)
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			if (i != j) {
				matriz[i][j] /= matriz[i][i];
			}
		}
		vetorAux[i] /= matriz[i][i];
		//copia pro outro vetor
		vetorKpp[i] = vetorAux[i];
		matriz[i][i] = 0;
	}

	//iterações
	do {
		//para cada linha da matriz/vetor
		for (i = 0; i < j_order; i++) {
			conta = 0;
			for (j = 0; j < j_order; j++) {
				conta -= matriz[i][j]*vetorKpp[j];
			}
			//guarda resultados da iteração anterior
			vetorK[i] = vetorKpp[i];
			//Xi^(k+1) = somatoria ( Bij*Xi^(k) ) + Xi^(0) 
			vetorKpp[i] = conta + vetorAux[i];
		}
		//calcula erro
		norma1 = norma2 = 0;

		for (i = 0; i < j_order; i++) {
			vetorK[i] = vetorKpp[i] - vetorK[i];
			norma1 += pow(vetorK[i], 2);
			norma2 += pow(vetorKpp[i], 2);
		}

		norma1 = sqrt(norma1);
		norma2 = sqrt(norma2);
		erroRel = norma1/norma2;
		iteracao++;

	} while ((iteracao < j_ite_max) && (erroRel > j_error));	//para de iterar quando atinge a precisão desejada
																//ou passa do limite de iterações estabelecido

	//printa o vetor do resultado final do sistema linear
	printaVec(vetorKpp, j_order);

	//libera matriz e vetor
	for (i = 0; i < j_order; i++) {
		free (matriz[i]);
	}
	free (matriz);
	free (vetorAux);
	free (vetorKpp);

	return 0;
}