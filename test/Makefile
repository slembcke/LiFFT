CFLAGS = -g -O0 -Wno-format
CFLAGS += -O3
LDLIBS = -lm

default: test

clean:
	-rm *.o test

test.o: ../lifft.h ../lifft_dct.h
test: test.o
