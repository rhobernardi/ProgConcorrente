/********************************************************************
*	Trabalho 1 - Programação Concorrente
*	Execução Sequencial do metodo de Jacobi-Richardson
*
*	Rodrigo das Neves Bernardi - 8066395
*	Thiago Ledur Lima - 8084214
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main (int argc, char *argv[]) {
	int i, j, j_order, j_row_test, j_ite_max, iteracao = 0;
	double j_error, somaLinha, norma1, norma2, erroRel, rowtest = 0;
	double **matrizLinha, **matriz, *vetorKpp, *vetorK, *vetorAux, *vetorX0, *vetorB;
	struct timespec clockStart, clockEnd;

	//entrada dos dados
	scanf("%d", &j_order);
	scanf("%d", &j_row_test);
	scanf("%lf", &j_error);
	scanf("%d", &j_ite_max);

	//aloca matrizes
	matrizLinha = (double**) malloc (j_order * sizeof(double*) );
	matriz = (double**) malloc (j_order * sizeof(double*) );
	for (i = 0; i < j_order; i++) {
		matrizLinha[i] = (double *) malloc (j_order * sizeof(double) );
		matriz[i] = (double *) malloc (j_order * sizeof(double) );
	}
	//entrada dos dados da matriz A
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			scanf("%lf", &matriz[i][j]);
		}
	}

	//aloca vetores
	vetorB = (double*) malloc (j_order * sizeof(double));	//vetor B
	vetorK = (double*) malloc (j_order * sizeof(double));	//vetor K
	vetorKpp = (double*) malloc (j_order * sizeof(double));	//vetor Kpp = K+1
	vetorX0 = (double*) malloc (j_order * sizeof(double));	//vetor X0 = X^K para K=0
	vetorAux = (double*) malloc (j_order * sizeof(double));	//vetor auxiliar para cálculo de erro
	//entrada dos dados do vetor B
	for (i = 0; i < j_order; i++) {
		scanf("%lf", &vetorB[i]);
	}

	//com tudo preparado, começa a contar tempo de execução do método
	clock_gettime(CLOCK_MONOTONIC, &clockStart);

	//prepara vetor X^0 e matriz A*, evitando fazer esses cálculos J_ORDER*J_ORDER vezes durante as iterações
	for (i = 0; i < j_order; i++) {
		for (j = 0; j < j_order; j++) {
			if (i != j) {
				matrizLinha[i][j] = matriz[i][j]/matriz[i][i];	//cálculo de Aij*
			}
		}
		vetorX0[i] = vetorKpp[i] = vetorB[i]/matriz[i][i];		//cálculo de Xi^0
		matrizLinha[i][i] = 0;
	}

	//iterações
	do {
		//guarda o vetor X^k, usado para calcular X^(k+1)
		for (i = 0; i < j_order; i++) {
			vetorK[i] = vetorKpp[i];
		}
		//método Jacobi-Richardson
		for (i = 0; i < j_order; i++) {
			somaLinha = 0;
			for (j = 0; j < j_order; j++) {
				if (i != j) {
					somaLinha -= matrizLinha[i][j]*vetorK[j];
				}
			}
			vetorKpp[i] = somaLinha + vetorX0[i];
		}
		//cálculo do erro
		norma1 = norma2 = 0;
		for (i = 0; i < j_order; i++) {
			vetorAux[i] = vetorKpp[i] - vetorK[i];
			norma1 += pow(vetorAux[i], 2);
			norma2 += pow(vetorKpp[i], 2);
		}
		norma1 = sqrt(norma1);
		norma2 = sqrt(norma2);
		erroRel = norma1/norma2;
		iteracao++;
	//itera até atingir o número máximo permitido ou a precisão desejada
	} while ( (iteracao < j_ite_max) && (erroRel > j_error) );

	//termina a contagem do tempo de execução do método
	clock_gettime(CLOCK_MONOTONIC, &clockEnd);

	//faz o row test
	for (i = 0; i < j_order; i++) {
		rowtest += vetorKpp[i]*matriz[j_row_test][i];
	}

	//printa iteracoes e o vetor do resultado final do sistema linear
	printf("\nIterations: %d\nRowTest: %d => [%.6f] =? %.6f\n",
				iteracao, j_row_test, rowtest, vetorX0[j_row_test]*matriz[j_row_test][j_row_test]);
	
	//opcional pra testar: printa tempo
	printf("Time: %.5lfs\n", ( ((double)(clockEnd.tv_nsec - clockStart.tv_nsec)/1000000000) + (clockEnd.tv_sec - clockStart.tv_sec) ) );

	//libera matrizes e vetores
	for (i = 0; i < j_order; i++) {
		free (matrizLinha[i]);
		free (matriz[i]);
	}
	free (matrizLinha);
	free (matriz);
	free (vetorK);
	free (vetorKpp);
	free (vetorX0);
	free (vetorAux);
	free (vetorB);

	return 0;
}
