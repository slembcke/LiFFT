#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>

// #define LIFFT_FLOAT_TYPE float
// #define LIFFT_STD_COMPLEX
#define LIFFT_IMPLEMENTATION
#include "lifft.h"

void doit(size_t len){
	lifft_complex_t x0[len], out[len], x1[len];
	for(unsigned i = 0; i < len; i++) x0[i] = (lifft_complex_t){cosf(2*_LIFFT_PI*i/len)};
	
	for(unsigned i = 0; i < 1000; i++){
		lifft_forward_complex(x0, out, len);
		lifft_inverse_complex(out, x1, len);
	}
	
	if(len <= 16){
		for(unsigned i = 0; i < len; i++) printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x1[i]), lifft_cimag(x1[i]));
	}
	
	double err = 0;
	for(unsigned i = 0; i < len; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
}

int main(int argc, const char* argv[]){
	doit(16);
	doit(1 << 16);
	
	return EXIT_SUCCESS;
}
