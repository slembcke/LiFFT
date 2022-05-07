#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>

// #define LIFFT_FLOAT_TYPE float
// #define LIFFT_STD_COMPLEX
#define LIFFT_IMPLEMENTATION
#include "lifft.h"

void doit(size_t len){
	lifft_complex_t x0[len], out[len], x1[len];
	for(unsigned i = 0; i < len; i++) x0[i] = (lifft_complex_t){rand()};
	
	for(unsigned i = 0; i < 1000; i++){
		lifft_forward_complex(x0, 0, out, 0, NULL, len);
		lifft_inverse_complex(out, 0, x1, 0, NULL, len);
	}
	
	if(len <= 64){
		for(unsigned i = 0; i < len; i++){
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x0[i]), lifft_cimag(x0[i]));
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x1[i]), lifft_cimag(x1[i]));
			
			lifft_complex_t diff = lifft_csub(x0[i], x1[i]);
			printf("%2d: %e + %ei\n", i, lifft_creal(diff), lifft_cimag(diff));
		}
	}
	
	double err = 0;
	for(unsigned i = 0; i < len; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
}

int main(int argc, const char* argv[]){
	doit(64);
	doit(1 << 16);
	
	return EXIT_SUCCESS;
}
