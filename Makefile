CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
TARGET = symnmf
SRC = symnmf.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)