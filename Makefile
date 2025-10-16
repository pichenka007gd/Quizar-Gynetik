CC=gcc
CFLAGS=-O3 -fPIC -std=c11 -Wall -Wextra -fopenmp
LDFLAGS=-shared -lm -fopenmp
SRC=src/c/main.c
TARGET=main.so

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
