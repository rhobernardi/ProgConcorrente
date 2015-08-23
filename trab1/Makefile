
CC = gcc -o
FLAGS = -lm -lpthread -Werror

all: iterative parallel

iterative: 
	$(CC) trab1_i jacobi-richardson_iterative.c $(FLAGS)

parallel:
	$(CC) trab1_p jacobi-richardson_pthreads.c $(FLAGS)

clean:
	@rm -rf *~ *.o trab* test

zip:
	@rm -rf *~ *.o
	@rm -rf *.zip
	@zip -r concorrentes_trab1.zip main.c ./matriz/ Makefile

runI:
	./trab1_i < ./matrizes/matriz3.txt

runP:
	./trab1_p < ./matrizes/matriz500.txt