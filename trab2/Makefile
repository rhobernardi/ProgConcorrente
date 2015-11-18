all: clean compile

compile:
	gcc Trab2SEQ.c -Wall -o Trab2SEQ
	mpicc Trab2PAR.c -fopenmp -Wall -o Trab2PAR

clean:
	@find -name "*~" | xargs rm -rf
	rm -rf *.o Trab2SEQ Trab2PAR out_image.ppm

zip: compile clean
	@rm -f Trab2_G12A.zip
	@zip -r Trab2_G12A.zip *

#sequencial
run1:
	Trab2SEQ

#MPI+OMP local
run2:
	export GOMP_CPU_AFFINITY="0-3"
	mpirun -np 4 Trab2PAR

#MPI+OMP cluster
run3:
	export GOMP_CPU_AFFINITY="0-3"
	mpirun -np 15 --hostfile cluster.txt Trab2PAR