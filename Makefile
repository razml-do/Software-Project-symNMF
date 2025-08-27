CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -lm
TARGET = symnmf
SRC = symnmf.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)