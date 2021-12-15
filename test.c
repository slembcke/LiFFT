#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define _LIFFT_PI 3.14159265358979323846

// #define LIFFT_STD_COMPLEX
#if defined(LIFFT_STD_COMPLEX)
	#include <complex.h>
	typedef complex float lifft_complex_t;
	
	static inline lifft_complex_t lift_complex(float real, float imag){return real + imag*I;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return x + y;}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return x - y;}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return x*y;}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return x/y;}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return conjf(x);}
	static inline float lifft_cabs(lifft_complex_t x){return cabsf(x);}
	static inline float lifft_creal(lifft_complex_t x){return crealf(x);}
	static inline float lifft_cimag(lifft_complex_t x){return cimagf(x);}
	static inline lifft_complex_t LIFFT_CISPI(lifft_complex_t x){return cexpf((2*_LIFFT_PI*I)*x);}
#elif !defined(lifft_complex_t)
	#include <complex.h>
	typedef struct {float real, imag;} lifft_complex_t;
	static inline lifft_complex_t lift_complex(float real, float imag){lifft_complex_t res = {real, imag}; return res;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real + y.real, x.imag + y.imag);}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real - y.real, x.imag - y.imag);}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real);}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return lift_complex((x.real*y.real + x.imag*y.imag)/(y.real*y.real + y.imag*y.imag), (x.imag*y.real - x.real*y.imag)/(y.real*y.real + y.imag*y.imag));}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return lift_complex(x.real, -x.imag);}
	static inline float lifft_cabs(lifft_complex_t x){return hypotf(x.real, x.imag);}
	static inline float lifft_creal(lifft_complex_t x){return x.real;}
	static inline float lifft_cimag(lifft_complex_t x){return x.imag;}
	static inline lifft_complex_t lifft_cispi(float x){return lift_complex(cosf(2*_LIFFT_PI*x), sinf(2*_LIFFT_PI*x));}
#else
	#error LiFFT: No types defined.
#endif

static void _lifft_process(lifft_complex_t* out, size_t len){
	for(int stride = 1; stride < len; stride *= 2){
		lifft_complex_t wm = lifft_cispi(-0.5/stride);
		for(int i = 0; i < len; i += 2*stride){
			lifft_complex_t w = lift_complex(1, 0);
			for(int j = 0; j < stride; j++){
				size_t idx0 = i + j, idx1 = idx0 + stride;
				lifft_complex_t p = out[idx0], q = lifft_cmul(w, out[idx1]);
				out[idx0] = lifft_cadd(p, q);
				out[idx1] = lifft_csub(p, q);
				w = lifft_cmul(w, wm);
			}
		}
	}
}

static unsigned _lifft_bits(size_t len){
	unsigned bits = log2(len);
	assert(len == 1 << bits);
	assert(bits <= 16);
	return bits;
}

static int _lifft_reverse_bits16(unsigned n, unsigned bits){
	static const unsigned REV4[] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
	
	int rev = 0;
	for(int i = 0; i < 4; i++) rev = (rev << 4) | REV4[n >> (4*i) & 0xF];
	return rev >> (16 - bits);
}

void lifft_forward_complex(lifft_complex_t* x, lifft_complex_t* out, size_t len){
	unsigned bits = _lifft_bits(len);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits16(i, bits)] = x[i];
	_lifft_process(out, len);
}

void lifft_inverse_complex(lifft_complex_t* x, lifft_complex_t* out, size_t len){
	unsigned bits = _lifft_bits(len);
	lifft_complex_t scale = lift_complex(1.0/len, 0);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits16(i, bits)] = lifft_cmul(x[i], scale);
	for(int i = 0; i < len; i++) out[i] = lifft_conj(out[i]);
	_lifft_process(out, len);
	for(int i = 0; i < len; i++) out[i] = lifft_conj(out[i]);
}

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
	
	float err = 0;
	for(unsigned i = 0; i < len; i++) err += lifft_cabs(lifft_csub(x0[i], x1[i]));
	printf("err: %f\n", err);
}

int main(int argc, const char* argv[]){
	doit(16);
	doit(1 << 16);
	
	return EXIT_SUCCESS;
}
