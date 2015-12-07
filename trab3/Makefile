all: clean compile

compile:
	gcc ./Trab3SEQ.c -Wall -o ./Trab3SEQ
	mpicc ./Trab3PAR.c -fopenmp -Wall -o ./Trab3PAR
	nvcc ./Trab3CUD.cu -o ./Trab3CUD

clean:
	@find -name "*~" | xargs rm -rf
	rm -rf *.o Trab3SEQ Trab3PAR Trab3CUD out_image.pgm out_image.ppm

zip: compile clean
	@rm -f Trab2_G12A.zip
	@zip -r Trab2_G12A.zip *

#sequencial
seq:
	./Trab3SEQ ./Imagens/RGB_big_13183x4847.ppm out_image.ppm >> ./Resultados/Sequencial_RGBbig.txt

#MPI+OMP cluster
par:
	export GOMP_CPU_AFFINITY="0-7"
	mpirun -np 11 --hostfile cluster.txt ./Trab3PAR ./Imagens/RGB_big_13183x4847.ppm out_image.ppm >> ./Resultados/Paralelo_RGBbig.txt

#CUDA
cud:
	./Trab3CUD ./Imagens/RGB_big_13183x4847.ppm out_image.ppm >> ./Resultados/CUDA_RGBbig.txt
	./Trab3CUD ./Imagens/GS_big_13183x4847.pgm out_image.pgm >> ./Resultados/CUDA_GSbig.txt