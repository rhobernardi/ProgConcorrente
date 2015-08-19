#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> //nao usa ainda
#include <time.h>	// nao usa ainda

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
		printf("%.4f  ", vec[i]);
	}
	printf ("\n\n");
	return;
}

int main (int argc, char *argv[]) {
	int i, j, j_order, j_row_test, j_ite_max, iteracao = 0;
	double j_error, conta;
	double **matriz, *vetorAux, *vetorResultado;

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
	vetorResultado = (double*) malloc (j_order * sizeof(double));
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
		vetorResultado[i] = vetorAux[i];
		matriz[i][i] = 0;
	}

	//iterações
	while (iteracao < j_ite_max) {	//TODO: condição de parada é iterações OU erro (por enquanto só faz por iterações)
		//para cada linha da matriz/vetor
		for (i = 0; i < j_order; i++) {
			conta = 0;
			for (j = 0; j < j_order; j++) {
				conta -= matriz[i][j]*vetorResultado[j];
			}
			//Xi^(k+1) = somatoria ( Bij * Xi^(k) ) + Xi^(0) 
			vetorResultado[i] = conta + vetorAux[i];
		}
		iteracao++;
	}

	//printa o vetor do resultado final do sistema linear
	printaVec(vetorResultado, j_order);

	//libera matriz e vetor
	for (i = 0; i < j_order; i++) {
		free (matriz[i]);
	}
	free (matriz);
	free (vetorAux);
	free (vetorResultado);

	

	return 0;
}