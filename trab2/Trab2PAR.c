/**
*	Programação Concorrente - SSC0143 - 2 Semestre de 2015
*	Prof. Dr. Júlio Cezar Estrella
*	Trabalho 2 - Smoothing de imagem utilizando MPI+OMP
*  
*	Alunos:
*		Thiago Ledur Lima		- 8084214
*		Rodrigo Neves Bernardi	- 8066395	
**/

#include <omp.h>
#include <mpi.h>
#include "Smoothing.c"
#include <time.h>

int main (int argc, char **argv) {
	int i, j, mpiprocs, mpirank, mpinamelen, faixa_height, original_height;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	PPMImage imgIn, imgOut, imgFaixaIn, imgFaixaOut;
	char in[64], out[64];
	struct timespec clockStart, clockEnd;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Get_processor_name(processor_name, &mpinamelen);

	/******************************************************************************************************
	**************************************** nó principal (master) ****************************************
	******************************************************************************************************/
	if (mpirank == 0) {
		strcpy(in, "image.ppm");
		strcpy(out, "out_image.ppm");

		// lê imagem
		readImage(&imgIn, &imgOut, in);

		//com tudo preparado, começa a contar tempo de execução do método
		clock_gettime(CLOCK_MONOTONIC, &clockStart);

		//divide e manda dados e pedaços da matriz da imagem
		for (i = 1; i < mpiprocs; i++) {
			MPI_Send(&imgIn.width, 1, MPI_INT, i, 'w', MPI_COMM_WORLD);	// envia a largura da imagem

			// calcula e envia a altura das faixas pra cada processo nos nós
			faixa_height = imgIn.height/(mpiprocs-1);
			if (imgIn.height%(mpiprocs-1) == 0) {	// caso a divisão da altura pelo número de nós seja exata
				MPI_Send(&faixa_height, 1, MPI_INT, i, 'h', MPI_COMM_WORLD);
			}
			else {
				if (i == mpiprocs-1) {	// caso a divisão não seja exata, calcula um valor diferente para a última faixa
					faixa_height += imgIn.height%(mpiprocs-1);
				}
				MPI_Send(&faixa_height, 1, MPI_INT, i, 'h', MPI_COMM_WORLD);
			}

			// envia as faixas da imagem para os nós
			if (i == 1) {	// primeira faixa
				for (j = 0; j < imgIn.height/(mpiprocs-1); j++) {
					MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				}
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
			}
			else if (i == mpiprocs-1) {	// última faixa
				j-=3;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				for (;j < imgIn.height; j++) {
					MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				}
			}
			else {	// faixas do meio
				j-=3;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				for (;j < (imgIn.height/(mpiprocs-1))*i; j++) {
					MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				}
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
				j++;
				MPI_Send(imgIn.data[j], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, i, 'i', MPI_COMM_WORLD);
			}
		}

		for (j = 1; j < mpiprocs; j++) {
			for (i = ((int)(imgIn.height/(mpiprocs-1)))*(j-1); i < ((int)(imgIn.height/(mpiprocs-1)))*j; i++) {
				//printf("From %d Linha %d\n", j, i);
				MPI_Recv(imgOut.data[i], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, j, 'v', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if (imgIn.height%(mpiprocs-1) != 0) {
				if (j == mpiprocs-1) {
					for (; i < imgIn.height; i++) {
						MPI_Recv(imgOut.data[i], imgIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, j, 'v', MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
				}
			}
		}

		//termina a contagem do tempo de execução do método
		clock_gettime(CLOCK_MONOTONIC, &clockEnd);
		printf("Tempo: %.5lfs\n", ( ((double)(clockEnd.tv_nsec - clockStart.tv_nsec)/1000000000) + (clockEnd.tv_sec - clockStart.tv_sec) ) );

		// salva imagem
		saveImage(&imgOut, out);
		// libera memória
		freeData(&imgIn);
		freeData(&imgOut);
	}
	/******************************************************************************************************
	************************************** nós secundários (workers) **************************************
	******************************************************************************************************/
	else {
		MPI_Recv(&imgFaixaIn.width, 1, MPI_INT, 0, 'w', MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// recebe largura
		MPI_Recv(&original_height, 1, MPI_INT, 0, 'h', MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// recebe altura

		if (mpirank == 1 || mpirank == mpiprocs-1) {	// ajusta o tamanho para as faixas das pontas, com replicação de 2 linhas
			imgFaixaIn.height = original_height + 2;
		}
		else {
			imgFaixaIn.height = original_height + 4;	// ajusta o tamanho para as faixas do meio, com replicação de 4 linhas
		}
		//aloca matrizes na memória para receber os dados da faixa e os manipular
		imgFaixaIn.data = (RGB**) malloc (imgFaixaIn.height * sizeof(RGB*));
		imgFaixaOut.data = (RGB**) malloc (original_height * sizeof(RGB*));
		for (i = 0; i < imgFaixaIn.height; i++) {
			imgFaixaIn.data[i] = (RGB *)malloc(imgFaixaIn.width * sizeof(RGB));
			imgFaixaOut.data[i] = (RGB *)malloc(imgFaixaIn.width * sizeof(RGB));
		}
		imgFaixaOut.width = imgFaixaIn.width;
		imgFaixaOut.height = original_height;

		// recebe todas as linhas da faixa
		for (i = 0; i < imgFaixaIn.height; i++) {
			MPI_Recv(imgFaixaIn.data[i], imgFaixaIn.width * sizeof(RGB), MPI_UNSIGNED_CHAR, 0, 'i', MPI_COMM_WORLD, MPI_STATUS_IGNORE);	// recebe altura real, com as faixas
		}

		omp_set_dynamic(0);	// explicitly disable dynamic teams
		omp_set_num_threads(4);
		//aplica filtro
		if (mpirank == 1) {
			#pragma omp parallel for default(shared) private(i, j)
			for (i = 0; i < original_height; i++) {
				for (j = 0; j < imgFaixaIn.width; j++) {
					colorFilter (&imgFaixaIn, &imgFaixaOut, i, j, i, j);
				}
			}
		}
		else {
			#pragma omp parallel for default(shared) private(i, j)
			for (i = 2; i < original_height+2; i++) {
				for (j = 0; j < imgFaixaIn.width; j++) {
					colorFilter (&imgFaixaIn, &imgFaixaOut, i, j, i-2, j);
				}
			}
		}

		for (i = 0; i < original_height; i++) {
			MPI_Send(imgFaixaOut.data[i], imgFaixaOut.width * sizeof(RGB), MPI_UNSIGNED_CHAR, 0, 'v', MPI_COMM_WORLD);
			//printf("Sending %d\n", i*mpirank);
		}

		// for (i = 0; i < imgFaixaIn.height; i++) {
		// 	free(imgFaixaIn.data[i]);
		// 	free(imgFaixaOut.data[i]);
		// }
		//free(imgFaixaIn.data);
		//free(imgFaixaOut.data);
	}

	MPI_Finalize();
	return 0;
}