CFLAGS = -g -O0
CFLAGS = -O3 -ffast-math
LDLIBS = -lm

default: test

clean:
	-rm *.o test

test: test.o
