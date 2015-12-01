all: clean compile

compile:
	#gcc Trab2SEQ.c -Wall -o Trab2SEQ
	nvcc Trab3CUDA.cu -o Trab3CUDA.out
	mpicc Trab3PAR.c -fopenmp -Wall -o Trab3PAR

clean:
	@find -name "*~" | xargs rm -rf
	rm -rf *.o Trab2SEQ Trab2PAR out_image.ppm

zip: compile clean
	@rm -f Trab3_G12A.zip
	@zip -r Trab3_G12A.zip *

login:
	# password: asd#12a;
	ssh grupo12a@halley.lasdpc.icmc.usp.br -p 22200

upload:
	git commit -m "update"
	git push

run:
	nvcc Trab3CUDA.cu -o Trab3CUDA.out
	#qsub -l nodes=1,walltime=00:10:00 -I
	./Trab3CUDA.out image.ppm out_image.ppm

# CUDA version
run1:
	./Trab3CUDA.out image.ppm out_image.ppm

#MPI+OMP local
run2:
	export GOMP_CPU_AFFINITY="0-3"
	mpirun -np 4 Trab3PAR

#MPI+OMP cluster
run3:
	export GOMP_CPU_AFFINITY="0-3"
	mpirun -np 15 --hostfile cluster.txt Trab3PAR
