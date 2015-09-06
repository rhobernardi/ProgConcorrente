
CC = gcc -o
FLAGS = -lm -lpthread -Werror

all: iterative parallel

iterative: 
	$(CC) T1S T1S.c -lm

parallel:
	$(CC) T1P T1P.c $(FLAGS)

clean:
	@rm -rf *~ *.o trab* test

zip:
	@rm -rf *~ *.o
	@rm -rf *.zip
	@zip -r concorrentes_trab1.zip main.c ./matriz/ Makefile
