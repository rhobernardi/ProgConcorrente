#SOURCES = lista.c
#HEADERS = graph.h lista.h fila.h
#CC = gcc -std=c99
CC = gcc

all: programa

programa: 
	$(CC) main.c -o trab1

zip:
	@rm -rf *~ *.o
	@rm -rf *.zip
	@zip -r concorrentes_trab1.zip main.c ./matriz/ Makefile

run:
	./trab1 < ./matrizes/matriz500.txt