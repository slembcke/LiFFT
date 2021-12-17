#include <stdint.h>
#include <stddef.h>

#ifndef LIFFT_NO_STDLIB
	#include <math.h>
	#include <assert.h>
#endif

#ifndef LIFFT_FLOAT_TYPE
	#define LIFFT_FLOAT_TYPE double
#endif

typedef LIFFT_FLOAT_TYPE lifft_float_t;
#define _LIFFT_PI 3.14159265358979323846

#if defined(LIFFT_STD_COMPLEX)
	#include <complex.h>
	typedef complex LIFFT_FLOAT_TYPE lifft_complex_t;
	
	static inline lifft_complex_t lift_complex(lifft_float_t real, lifft_float_t imag){return real + imag*I;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return x + y;}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return x - y;}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return x*y;}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return x/y;}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return conj(x);}
	static inline lifft_float_t lifft_cabs(lifft_complex_t x){return cabs(x);}
	static inline lifft_float_t lifft_creal(lifft_complex_t x){return creal(x);}
	static inline lifft_float_t lifft_cimag(lifft_complex_t x){return cimag(x);}
	static inline lifft_complex_t lifft_cispi(lifft_complex_t x){return cexp((2*_LIFFT_PI*I)*x);}
#elif !defined(LIFFT_COMPLEX_TYPE)
	typedef struct {lifft_float_t real, imag;} lifft_complex_t;
	static inline lifft_complex_t lift_complex(lifft_float_t real, lifft_float_t imag){lifft_complex_t res = {real, imag}; return res;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real + y.real, x.imag + y.imag);}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real - y.real, x.imag - y.imag);}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return lift_complex(x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real);}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return lift_complex((x.real*y.real + x.imag*y.imag)/(y.real*y.real + y.imag*y.imag), (x.imag*y.real - x.real*y.imag)/(y.real*y.real + y.imag*y.imag));}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return lift_complex(x.real, -x.imag);}
	static inline lifft_float_t lifft_cabs(lifft_complex_t x){return sqrt(x.real*x.real + x.imag*x.imag);}
	static inline lifft_float_t lifft_creal(lifft_complex_t x){return x.real;}
	static inline lifft_float_t lifft_cimag(lifft_complex_t x){return x.imag;}
	static inline lifft_complex_t lifft_cispi(lifft_float_t x){return lift_complex(cos(2*_LIFFT_PI*x), sin(2*_LIFFT_PI*x));}
#endif

#ifdef LIFFT_IMPLEMENTATION

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
	assert(bits <= 32);
	return bits;
}

static inline size_t _lifft_reverse_bits24(size_t n, unsigned bits){
	static const uint8_t REV[] = {
		0x00, 0x20, 0x10, 0x30, 0x08, 0x28, 0x18, 0x38, 0x04, 0x24, 0x14, 0x34, 0x0C, 0x2C, 0x1C, 0x3C,
		0x02, 0x22, 0x12, 0x32, 0x0A, 0x2A, 0x1A, 0x3A, 0x06, 0x26, 0x16, 0x36, 0x0E, 0x2E, 0x1E, 0x3E,
		0x01, 0x21, 0x11, 0x31, 0x09, 0x29, 0x19, 0x39, 0x05, 0x25, 0x15, 0x35, 0x0D, 0x2D, 0x1D, 0x3D,
		0x03, 0x23, 0x13, 0x33, 0x0B, 0x2B, 0x1B, 0x3B, 0x07, 0x27, 0x17, 0x37, 0x0F, 0x2F, 0x1F, 0x3F,
	};
	
	size_t rev = 0;
	rev <<= 6; rev |= REV[n & 0x3F]; n >>= 6;
	rev <<= 6; rev |= REV[n & 0x3F]; n >>= 6;
	rev <<= 6; rev |= REV[n & 0x3F]; n >>= 6;
	rev <<= 6; rev |= REV[n & 0x3F]; n >>= 6;
	return rev >> (24 - bits);
}

void lifft_forward_complex(lifft_complex_t* x, lifft_complex_t* out, size_t len){
	unsigned bits = _lifft_bits(len);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits24(i, bits)] = x[i];
	_lifft_process(out, len);
}

void lifft_inverse_complex(lifft_complex_t* x, lifft_complex_t* out, size_t len){
	unsigned bits = _lifft_bits(len);
	lifft_complex_t scale = lift_complex(1.0/len, 0);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits24(i, bits)] = lifft_conj(lifft_cmul(x[i], scale));
	_lifft_process(out, len);
	for(int i = 0; i < len; i++) out[i] = lifft_conj(out[i]);
}

#endif
