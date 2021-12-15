CFLAGS = -Os -ffast-math
LDLIBS = -lm

default: test

clean:
	-rm *.o test

test: test.o
