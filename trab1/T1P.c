#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#define NUMTHREADS 8

//variáveis globais para uso das threads
pthread_barrier_t preparoA, preparoK, calculoErro, verificaParada;
int encerra;

//estrutura de dados de cada thread
typedef struct tdata {
	int tid;
	double **matriz;
	double **matrizLinha;		//todas threads têm os ponteiros pros mesmos dados
	double *vetorK;				//não precisa sincronizar a manipulação deles (mutex lock/unlock)
	double *vetorKpp;			//porque cada thread SEMPRE escreve em pedaços diferente
	double *vetorX0;
	double *vetorAux;
	double *vetorB;
	double norma1;
	double norma2;
	int order;
	int start;
	int end;
} tdata;

void* fazLinha (void *);

int main (int argc, char *argv[]) {
	//por enquanto divide em 8 threads, depois é só manipular o número como quiser
	pthread_t thread[NUMTHREADS];
	tdata thread_data[NUMTHREADS];
	int i, j, j_order, j_row_test, j_ite_max, divisao, iteracao = 0;
	double j_error, rowtest = 0, norma1, norma2, erroRel;
	double **matrizLinha, **matriz, *vetorKpp, *vetorK, *vetorAux, *vetorX0, *vetorB;
	struct timespec clockStart, clockEnd;

	//inicializa barreiras para sincronizar a execução do algoritmo
	pthread_barrier_init (&preparoA, NULL, NUMTHREADS);
	pthread_barrier_init (&preparoK, NULL, NUMTHREADS);
	pthread_barrier_init (&calculoErro, NULL, NUMTHREADS+1);
	pthread_barrier_init (&verificaParada, NULL, NUMTHREADS+1);

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

	encerra = 0;

	//cria threads, seta os dados delas e executa
	divisao = j_order/NUMTHREADS;
	for (i = 0; i < NUMTHREADS; i++) {
		thread_data[i].tid = i;
		thread_data[i].vetorB = vetorB;
		thread_data[i].vetorK = vetorK;
		thread_data[i].vetorKpp = vetorKpp;
		thread_data[i].vetorX0 = vetorX0;
		thread_data[i].vetorAux = vetorAux;
		thread_data[i].matrizLinha = matrizLinha;
		thread_data[i].matriz = matriz;
		thread_data[i].order = j_order;
		//configura o pedaço que cada thread vai mexer, nesse caso dividido em NUMTHREADS threads
		thread_data[i].start = i*divisao;
		//printf("thread_data[%d].start = %d\n", i, thread_data[i].start);	//debug
		if(i != (NUMTHREADS-1) ) {
			thread_data[i].end = (i+1)*divisao;
			//printf("thread_data[%d].end = %d\n", i, thread_data[i].end);	//debug
		}
		else {
			thread_data[i].end = j_order;
			//printf("thread_data[%d].end = %d\n", i, thread_data[i].end);	//debug
		}
		if (pthread_create(&thread[i], NULL, fazLinha, &thread_data[i]) ) {
	    	printf ("ERRRO ao criar uma thread");
	    	return -1;
	    }
	}

	//a main é responsável por sincronizar as iterações entre as threads, calcular o erro e verificar condição de parada
	while (encerra == 0) {
		pthread_barrier_wait(&calculoErro);

		//parte menos complexa do cálculo do erro (menos operações)
		norma1 = norma2 = 0;
		for (i = 0; i < NUMTHREADS; i++) {
			norma1 += thread_data[i].norma1;
			norma2 += thread_data[i].norma2;
		}
		norma1 = sqrt(norma1);
		norma2 = sqrt(norma2);
		erroRel = norma1/norma2;

		iteracao++;
		if ( (iteracao >= j_ite_max) || (erroRel < j_error) ) {
			encerra = 1;
		}
		pthread_barrier_wait (&verificaParada);
	}

	//termina a contagem do tempo de execução do método
	clock_gettime(CLOCK_MONOTONIC, &clockEnd);

	//espera todas threads finalizarem
	for (i = 0; i < NUMTHREADS; i++) {
		pthread_join (thread[i], NULL);
	}

	//faz o row test
	for (i = 0; i < j_order; i++) {
		rowtest += vetorKpp[i]*matriz[j_row_test][i];
	}

	//printa iteracoes e o vetor do resultado final do sistema linear
	printf("\nIterations: %d\nRowTest: %d => [%.6f] =? %.6f\n", iteracao, j_row_test, rowtest, vetorX0[j_row_test]*matriz[j_row_test][j_row_test]);

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

void* fazLinha (void *arg) {
	tdata *data = (tdata *) arg;
	int i, j;
	double somaLinha;


	//prepara vetor X^0 e matriz A*, evitando fazer esses cálculos J_ORDER*J_ORDER vezes durante as iterações
	for (i = data->start; i < data->end; i++) {
		for (j = 0; j < data->order; j++) {
			if (i != j) {
				data->matrizLinha[i][j] = data->matriz[i][j]/data->matriz[i][i];	//cálculo de Aij*
			}
		}
		data->vetorX0[i] = data->vetorKpp[i] = data->vetorB[i]/data->matriz[i][i];		//cálculo de Xi^0
		data->matrizLinha[i][i] = 0;
	}


	pthread_barrier_wait (&preparoA);

	while (encerra == 0) {
		//guarda o vetor X^k, usado para calcular X^(k+1)
		for (i = data->start; i < data->end; i++) {
			data->vetorK[i] = data->vetorKpp[i];
		}
		pthread_barrier_wait (&preparoK);	//aguarda as outras threads salvarem K

		//método Jacobi-Richardson
		for (i = data->start; i < data->end; i++) {
			somaLinha = 0;
			//for (j = data->start; j < data->end; j++) {
			for (j = 0; j < data->order; j++) {
				if (i != j) {
					somaLinha -= data->matrizLinha[i][j] * data->vetorK[j];
				}
			}
			data->vetorKpp[i] = somaLinha + data->vetorX0[i];
		}
		//parte mais complexa do cálculo do erro (mais operações)
		data->norma1 = data->norma2 = 0;
		for (i = data->start; i < data->end; i++) {
			data->vetorAux[i] = data->vetorKpp[i] - data->vetorK[i];
			data->norma1 += pow(data->vetorAux[i], 2);
			data->norma2 += pow(data->vetorKpp[i], 2);
		}
		pthread_barrier_wait (&calculoErro);
		pthread_barrier_wait (&verificaParada);
	}

	pthread_exit(NULL);
}