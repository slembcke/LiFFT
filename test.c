#include <stdint.h>

#include <stdlib.h>
#include <stdio.h>

// #define LIFFT_FLOAT_TYPE float
// #define LIFFT_STD_COMPLEX
#define LIFFT_IMPLEMENTATION
#include "lifft.h"

// void lifft_forward_real(lifft_float_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, lifft_complex_t* tmp, size_t len){
// 	if(stride_in == 0) stride_in = 1;
// 	if(stride_out == 0) stride_out = 1;
// 	if(tmp == NULL) tmp = alloca(len/2*sizeof(lifft_complex_t));
	
// 	unsigned bits = _lifft_bits(len/2);
// 	for(int i = 0; i < len; i += 2) tmp[_lifft_rev_bits24(i/2, bits)] = lifft_complex(x_in[i*stride_in + 0], x_in[i*stride_in + 1]);
// 	_lifft_process(tmp, len/2);
	
// 	lifft_complex_t w = lifft_complex(1, 0), wm = lifft_cispi(-1.0/len);
// 	for(int i0 = 0; i0 < len/2; i0++){
// 		int i1 = -i0 & (len/2 - 1);
// 		lifft_complex_t Xe = lifft_cmul(lifft_cadd(tmp[i0], lifft_conj(tmp[i1])), lifft_complex(0.5, -0.0));
// 		lifft_complex_t Xo = lifft_cmul(lifft_csub(tmp[i0], lifft_conj(tmp[i1])), lifft_complex(0.0, -0.5));
// 		x_out[i0*stride_out] = lifft_cadd(Xe, lifft_cmul(Xo, w));
// 		w = lifft_cmul(w, wm);
// 	}
// }

static void zlog(const char* label, int i, lifft_complex_t x){
	printf("%s[%2d]: % e + % ei\n", label, i, lifft_creal(x), lifft_cimag(x));
}

void doit(size_t len){
	lifft_complex_t x0[len], X[len], x1[len];
	for(unsigned i = 0; i < len; i++) x0[i] = lifft_complex(rand(), rand());
	
	for(unsigned i = 0; i < 1000; i++){
		lifft_forward_complex(x0, 0, X, 0, len);
		lifft_inverse_complex(X, 0, x1, 0, len);
	}
	
	if(len <= 64){
		for(unsigned i = 0; i < len; i++){
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x0[i]), lifft_cimag(x0[i]));
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x1[i]), lifft_cimag(x1[i]));
			
			// lifft_complex_t diff = lifft_csub(x0[i], x1[i]);
			// printf("%2d: % e + % ei\n", i, lifft_creal(diff), lifft_cimag(diff));
		}
	}
	
	double err = 0;
	for(unsigned i = 0; i < len; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
}

// Compute the forward DCT.
void lifft_inverse_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t len){
	if(stride_in == 0) stride_in = 1;
	if(stride_out == 0) stride_out = 1;
	lifft_complex_t* tmp = alloca(len*sizeof(lifft_complex_t));
	lifft_complex_t* Z = alloca(len*sizeof(lifft_complex_t));
	unsigned bits = _lifft_bits(len);
	
	lifft_complex_t w = lifft_complex(0.5/len, 0), wm = lifft_cispi(0.25/len);
	for(int i = 0; i < len; i++){
		tmp[i] = lifft_cmul(lifft_complex(x_in[i], 0), w);
		w = lifft_cmul(w, wm);
	}
	
	w = lifft_complex(1, 0), wm = lifft_cispi(0.5/len);
	for(int i = 0; i < len; i++){
		lifft_complex_t Xe = lifft_cadd(tmp[i], lifft_conj(tmp[-i & (len - 1)]));
		lifft_complex_t Xo = lifft_cmul(lifft_csub(tmp[i], lifft_conj(tmp[-i & (len - 1)])), w);
		Z[_lifft_rev_bits24(i, bits)] = lifft_cadd(Xe, lifft_cmul(Xo, lifft_complex(0, 1)));
		w = lifft_cmul(w, wm);
	}
	Z[0] = lifft_cmul(Z[0], lifft_complex(0.5, 0.5));
	
	
	for(int i = 0; i < len; i++) Z[i] = lifft_conj(Z[i]);
	_lifft_process(Z, len);
	
	for(int i = 0; i < len/2; i++){
		x_out[2*i + 0] = +lifft_creal(Z[i]);
		x_out[2*i + 1] = -lifft_cimag(Z[i]);
	}
}

void dctit(size_t len){
	lifft_float_t x0[len], X[len], x1[len];
	for(unsigned i = 0; i < len; i++) x0[i] = rand();
	
	for(unsigned i = 0; i < 1000; i++){
		lifft_forward_dct(x0, 0, X, 0, len);
		lifft_inverse_dct(X, 0, x1, 0, len);
	}
	
	if(len <= 64){
		for(unsigned i = 0; i < len; i++){
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x0[i]), lifft_cimag(x0[i]));
			// printf("%2d: % .2f + % .2fi\n", i, lifft_creal(x1[i]), lifft_cimag(x1[i]));
			
			// lifft_complex_t diff = lifft_csub(x0[i], x1[i]);
			// printf("%2d: % e + % ei\n", i, lifft_creal(diff), lifft_cimag(diff));
		}
	}
	
	double err = 0;
	for(unsigned i = 0; i < len; i++) err += x0[i] - x1[i];
	printf("err: %f\n", err);
}

int main(int argc, const char* argv[]){
	// doit(32);
	// doit(1 << 16);
	
	dctit(32);
	dctit(1 << 16);
	
	// lifft_float_t x0[16] = {
	// 	0.70203658, 0.30785784, 0.80697642, 0.2063156 ,
	// 	0.74611309, 0.44949445, 0.58790534, 0.94034123,
	// 	0.86815133, 0.78308922, 0.51704855, 0.58557402,
	// 	0.49798021, 0.43429341, 0.52435585, 0.47455634,
	// }, X[16], x1[16];
	// lifft_forward_dct(x0, 0, X, 0, 16);
	// lifft_inverse_dct(X, 0, x1, 0, 16);
	
	// for(int i = 0; i < 16; i++){
	// 	printf("%f\n", x0[i] - x1[i]);
	// }
	
	return EXIT_SUCCESS;
}
