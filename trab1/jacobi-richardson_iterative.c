#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//printa matrizA
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

	int i, j, j_order, j_row_test, j_ite_max, iteracao = 0;
	double j_error, conta, norma1, norma2, erroRel;
	double **matrizA, **matrizB, *vetorB, *vetorX, *vetorK;

	//entrada dos dados
	scanf("%d", &j_order);
	scanf("%d", &j_row_test);
	scanf("%lf", &j_error);
	scanf("%d", &j_ite_max);

	//aloca matrizA
	matrizA = (double**) malloc (j_order * sizeof(double*) );
	for (i = 0; i < j_order; i++) {
		matrizA[i] = (double *) malloc (j_order * sizeof(double) );
	}
	//le dados da matrizA
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			scanf("%lf", &matrizA[i][j]);
		}
	}

	//aloca vetor
	vetorB = (double*) malloc (j_order * sizeof(double));
	vetorX = (double*) malloc (j_order * sizeof(double));
	vetorK = (double*) malloc (j_order * sizeof(double));
	//le dados do vetor
	for (i = 0; i < j_order; i++) {
		scanf("%lf", &vetorB[i]);
	}

	printf("Matriz A\t Vetor B\n");
	for (i = 0; i < j_order; i++){
		for (j = 0; j < j_order; j++)
			printf("%.1lf  ", matrizA[i][j]);

		printf("\t  %.1lf\n", vetorB[i]);
	}
	

	//primeira parte do método, encontrando B* (modificando a matrizA) e x(n)^(0) (modificando o vetor)
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			if (i != j) {
				matrizA[i][j] /= matrizA[i][i];
			}
		}
		vetorB[i] /= matrizA[i][i];
		//copia pro outro vetor
		vetorX[i] = vetorB[i];
		matrizA[i][i] = 0;
	}

	matrizB = matrizA;

	printf("\nMatriz B");
	printaMat(matrizB, j_order);

	//iterações
	do {
		//para cada linha da matrizB/vetor
		for (i = 0; i < j_order; i++) {
			conta = 0;
			for (j = 0; j < j_order; j++) {
				conta -= matrizB[i][j]*vetorX[j];
			}
			//guarda resultados da iteração anterior
			vetorK[i] = vetorX[i];
			//Xi^(k+1) = somatoria ( Bij*Xi^(k) ) + Xi^(0) 
			vetorX[i] = conta + vetorB[i];
		}
		//calcula erro
		norma1 = norma2 = 0;

		for (i = 0; i < j_order; i++) {
			vetorK[i] = vetorX[i] - vetorK[i];
			norma1 += pow(vetorK[i], 2);
			norma2 += pow(vetorX[i], 2);
		}

		norma1 = sqrt(norma1);
		norma2 = sqrt(norma2);
		erroRel = norma1/norma2;
		iteracao++;

	} while ((iteracao < j_ite_max) && (erroRel > j_error));	//para de iterar quando atinge a precisão desejada
																//ou passa do limite de iterações estabelecido

	printaVec(vetorB, j_order);

	//printa o vetor do resultado final do sistema linear
	printaVec (vetorX, j_order);


	//RowTest = j_row_test => [argm1] =? argm2
	//argm1 = matrix[j_row_test][0]*vetorX[0] + ... + matrix[j_row_test][n-1]*vetorX[n-1]
	//argm2 = vetorB[j_row_test]
	double argm1 = 0, argm2 = 0;

	for (i = 0; i< j_order; i++){
		argm1 += matrizB[j_row_test][i]*vetorX[i];
	}

	argm2 = vetorB[j_row_test];

	printf("\n\n---------------------------------------------------------\n");
	printf("Iterations: %d\n", iteracao);
	printf("RowTest: %d => [%.6lf] =? %.6lf\n", j_row_test, argm1, argm2);
	printf("---------------------------------------------------------\n");

	//libera matrizA e vetor
	for (i = 0; i < j_order; i++) {
		free (matrizB[i]);
	}
	free (matrizB);
	free (vetorB);
	free (vetorX);

	return 0;
}