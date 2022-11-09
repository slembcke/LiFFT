#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>

// #define LIFFT_FLOAT_TYPE float
// #define LIFFT_STD_COMPLEX
#define LIFFT_IMPLEMENTATION
#include "../lifft.h"

void fft_it(size_t n, int iterations){
	lifft_complex_t x0[n], X[n], x1[n];
	for(unsigned i = 0; i < n; i++){
		x0[i] = lifft_complex((float)rand()/(float)RAND_MAX, 0);
		x1[i] = lifft_complex(0, 0);
	}
	
	for(unsigned i = 0; i < iterations; i++){
		// lifft_forward_real(x0, 2, X, 1, n);
		// lifft_inverse_real(X, 1, x1, 2, n);
		lifft_forward_complex(x0, 1, X, 1, n);
		lifft_inverse_complex(X, 1, x1, 1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err on FFT size %9d: %e\n", (int)n, err);
}

void fft_it_2d(size_t n, int iterations){
	lifft_complex_t x0[n*n], X[n*n], x1[n*n];
	for(unsigned i = 0; i < n*n; i++) x0[i] = lifft_complex((float)rand()/(float)RAND_MAX, 0);
	
	for(unsigned i = 0; i < iterations; i++){
		LIFFT_APPLY_2D(lifft_forward_complex, x0, X, n);
		LIFFT_APPLY_2D(lifft_inverse_complex, X, x1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n*n; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err on FFT size %3d x %3d: %e\n", (int)n, (int)n, err);
}

void dct_it(size_t n, int iterations){
	lifft_float_t x0[n], X[n], x1[n];
	for(unsigned i = 0; i < n; i++) x0[i] = (float)rand()/(float)RAND_MAX;
	
	for(unsigned i = 0; i < iterations; i++){
		lifft_forward_dct(x0, 1, X, 1, n);
		lifft_inverse_dct(X, 1, x1, 1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n; i++) err += fabs(x0[i] - x1[i]);
	printf("err on DCT size %9d: %e\n", (int)n, err);
}

void dct_it_2d(size_t n, int iterations){
	lifft_float_t x0[n*n], X[n*n], x1[n*n];
	for(unsigned i = 0; i < n*n; i++) x0[i] = (float)rand()/(float)RAND_MAX;
	
	for(unsigned i = 0; i < iterations; i++){
		LIFFT_APPLY_2D(lifft_forward_dct, x0, X, n);
		LIFFT_APPLY_2D(lifft_inverse_dct, X, x1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n*n; i++) err += fabs(x0[i] - x1[i]);
	printf("err on DCT size %3d x %3d: %e\n", (int)n, (int)n, err);
}

int main(int argc, const char* argv[]){
	int iterations = 1;
	
	fft_it(32, iterations);
	fft_it(1 << 16, iterations);
	fft_it_2d(8, iterations);
	fft_it_2d(1 << 8, iterations);
	
	dct_it(32, iterations);
	dct_it(1 << 16, iterations);
	dct_it_2d(32, iterations);
	dct_it_2d(1 << 8, iterations);
	
	// unsigned n = 8;
	// lifft_complex_t x0[n], X[n], x1[n];
	// for(unsigned i = 0; i < n; i++) x0[i] = lifft_complex((float)rand()/(float)RAND_MAX, 0);
	
	// lifft_forward_real(x0, 2, X, 1, n);
	// lifft_inverse_real(X, 1, x1, 2, n);
	
	// lifft_forward_dct((lifft_float_t*)x0, 2, (lifft_float_t*)X, 2, n);
	// lifft_inverse_dct((lifft_float_t*)X, 2, (lifft_float_t*)x1, 2, n);
	
	// for(unsigned i = 0; i < n; i++){
	// 	printf("% 2d: %.3f %.3f -> %+.3f%+.3f -> %+.3f%+.3f\n", i, x0[i], X[i], x1[i]);
	// }
	
	return EXIT_SUCCESS;
}
