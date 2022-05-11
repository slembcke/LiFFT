#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>

// #define LIFFT_FLOAT_TYPE float
// #define LIFFT_STD_COMPLEX
#define LIFFT_IMPLEMENTATION
#include "../lifft.h"

typedef void lifft_func(lifft_complex_t*, size_t, lifft_complex_t*, size_t, size_t);
void lifft_apply_2d(lifft_func func, lifft_complex_t* x_in, lifft_complex_t* x_out, size_t n){
	lifft_complex_t* tmp = alloca(n*n*sizeof(lifft_complex_t));
	for(int i = 0; i < n; i++) func(x_in + i*n, 1,   tmp + i, n, n);
	for(int i = 0; i < n; i++) func( tmp + i*n, 1, x_out + i, n, n);
}

typedef void lifft_func_(lifft_float_t*, size_t, lifft_float_t*, size_t, size_t);
void lifft_apply_2d_(lifft_func_ func, lifft_float_t* x_in, lifft_float_t* x_out, size_t n){
	lifft_float_t* tmp = alloca(n*n*sizeof(lifft_float_t));
	for(int i = 0; i < n; i++) func(x_in + i*n, 1,   tmp + i, n, n);
	for(int i = 0; i < n; i++) func( tmp + i*n, 1, x_out + i, n, n);
}

// -----------

void fft_it(size_t n, int iterations){
	lifft_complex_t x0[n], X[n], x1[n];
	for(unsigned i = 0; i < n; i++) x0[i] = lifft_complex((float)rand()/(float)RAND_MAX, 0);
	
	for(unsigned i = 0; i < iterations; i++){
		lifft_forward_complex(x0, 1, X, 1, n);
		lifft_inverse_complex(X, 1, x1, 1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
}

void fft_it_2d(size_t n, int iterations){
	lifft_complex_t x0[n*n], X[n*n], x1[n*n];
	for(unsigned i = 0; i < n*n; i++) x0[i] = lifft_complex((float)rand()/(float)RAND_MAX, 0);
	
	for(unsigned i = 0; i < iterations; i++){
		lifft_apply_2d(lifft_forward_complex, x0, X, n);
		lifft_apply_2d(lifft_inverse_complex, X, x1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n*n; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
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
	printf("err: %f\n", err);
}

void dct_it_2d(size_t n, int iterations){
	lifft_float_t x0[n*n], X[n*n], x1[n*n];
	for(unsigned i = 0; i < n*n; i++) x0[i] = (float)rand()/(float)RAND_MAX;
	
	for(unsigned i = 0; i < iterations; i++){
		lifft_apply_2d_(lifft_forward_dct, x0, X, n);
		lifft_apply_2d_(lifft_inverse_dct, X, x1, n);
	}
	
	double err = 0;
	for(unsigned i = 0; i < n*n; i++) err += fabs(x0[i] - x1[i]);
	printf("err: %f\n", err);
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
	
	// lifft_float_t x0[16] = {
	// 	0.70203658, 0.30785784, 0.80697642, 0.2063156 ,
	// 	0.74611309, 0.44949445, 0.58790534, 0.94034123,
	// 	0.86815133, 0.78308922, 0.51704855, 0.58557402,
	// 	0.49798021, 0.43429341, 0.52435585, 0.47455634,
	// }, x1[16];
	// lifft_forward_dct(x0, 1, x1, 1, 16);
	// lifft_inverse_dct(x1, 1, x1, 1, 16);
	
	// for(int i = 0; i < 16; i++){
	// 	// printf("%f\n", x0[i]);
	// 	printf("%f\n", x0[i] - x1[i]);
	// }
	
	return EXIT_SUCCESS;
}
