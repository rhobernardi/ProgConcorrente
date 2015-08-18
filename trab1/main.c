#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


int main(int argc, char* argv[])
{
	int **matrixA;
	int *matrixB;
	int ordem, fila, max_iterations, i, j;
	double error;

	scanf("%d", &ordem);
	scanf("%d", &fila);
	scanf("%lf", &error);
	scanf("%d", &max_iterations);

	matrixB = (int*) malloc(ordem*sizeof(int));
	matrixA = (int **) malloc(ordem*sizeof(int*));
	
	for(i = 0; i < ordem; i++)
	{
		matrixA[i] = (int *) malloc(ordem*sizeof(int));
	}

	for(i = 0; i < ordem; i++)
	{
		for(j = 0; j < ordem; j++)
			scanf("%d", &matrixA[i][j]);
	}

	for(i = 0; i < ordem; i++)
	{
		scanf("%d", &matrixB[i]);
	}

	for(i = 0; i < ordem; i++)
		printf("%d \n", matrixB[i]);

	return 0;
}
