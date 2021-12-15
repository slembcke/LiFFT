#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define _LIFFT_PI 3.14159265358979323846

#define LIFFT_STD_COMPLEX
#if defined(LIFFT_STD_COMPLEX)
	#include <complex.h>
	typedef complex float lifft_complex;
	
	#define LIFFT_COMPLEX(_real_, _imag_) (_real_ + _imag_*I)
	#define LIFFT_CADD(_x_, _y_) ((_x_) + (_y_))
	#define LIFFT_CSUB(_x_, _y_) ((_x_) - (_y_))
	#define LIFFT_CMUL(_x_, _y_) ((_x_) * (_y_))
	#define LIFFT_CDIV(_x_, _y_) ((_x_) / (_y_))
	#define LIFFT_CONJ(_x_) conjf(_x_)
	#define LIFFT_CABS(_x_) cabsf(_x_)
	#define LIFFT_CREAL(_x_) crealf(_x_)
	#define LIFFT_CIMAG(_x_) cimagf(_x_)
	#define LIFFT_CISPI(_x_) cexpf(2*_LIFFT_PI*I*(_x_))
#elif !defined(lifft_complex)
	#include <complex.h>
	typedef struct {
		float real, imag;
	} lifft_complex;
	
	static inline lifft_complex LIFFT_COMPLEX(float real, float imag){lifft_complex res = {real, imag}; return res;}
	static inline lifft_complex LIFFT_CADD(lifft_complex x, lifft_complex y){return LIFFT_COMPLEX(x.real + y.real, x.imag + y.imag);}
	static inline lifft_complex LIFFT_CSUB(lifft_complex x, lifft_complex y){return LIFFT_COMPLEX(x.real - y.real, x.imag - y.imag);}
	static inline lifft_complex LIFFT_CMUL(lifft_complex x, lifft_complex y){return LIFFT_COMPLEX(x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real);}
	
	static inline lifft_complex LIFFT_CDIV(lifft_complex x, float y){
		return LIFFT_COMPLEX(x.real/y, x.imag/y);
		// float denom = y.real*y.real + y.imag*y.imag;
		// return LIFFT_COMPLEX((x.real*y.real + x.imag*y.imag)/denom, (x.imag*y.real - x.real*y.imag)/denom);
	}
	
	static inline lifft_complex LIFFT_CONJ(lifft_complex x){return LIFFT_COMPLEX(x.real, -x.imag);}
	static inline float LIFFT_CABS(lifft_complex x){return hypotf(x.real, x.imag);}
	static inline float LIFFT_CREAL(lifft_complex x){return x.real;}
	static inline float LIFFT_CIMAG(lifft_complex x){return x.imag;}
	static inline lifft_complex LIFFT_CISPI(float x){return LIFFT_COMPLEX(cosf(2*_LIFFT_PI*x), sinf(2*_LIFFT_PI*x));}
#else
	#error LiFFT: No types defined.
#endif

static void _lifft_process(lifft_complex* out, size_t len){
	for(int stride = 1; stride < len; stride *= 2){
		lifft_complex wm = LIFFT_CISPI(-stride/2);
		for(int i = 0; i < len; i += 2*stride){
			lifft_complex w = {1}; // TODO
			for(int j = 0; j < stride; j++){
				size_t idx0 = i + j, idx1 = idx0 + stride;
				lifft_complex p = out[idx0], q = LIFFT_CMUL(w, out[idx1]);
				out[idx0] = LIFFT_CADD(p, q);
				out[idx1] = LIFFT_CSUB(p, q);
				w = LIFFT_CMUL(w, wm);
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

void lifft_forward_complex(lifft_complex* x, lifft_complex* out, size_t len){
	unsigned bits = _lifft_bits(len);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits16(i, bits)] = x[i];
	_lifft_process(out, len);
}

void lifft_inverse_complex(lifft_complex* x, lifft_complex* out, size_t len){
	unsigned bits = _lifft_bits(len);
	for(int i = 0; i < len; i++) out[_lifft_reverse_bits16(i, bits)] = LIFFT_CDIV(x[i], len);
	for(int i = 0; i < len; i++) out[i] = LIFFT_CONJ(out[i]);
	_lifft_process(out, len);
	for(int i = 0; i < len; i++) out[i] = LIFFT_CONJ(out[i]);
}

void doit(size_t len){
	lifft_complex x0[len], out[len], x1[len];
	for(unsigned i = 0; i < len; i++) x0[i] = (lifft_complex){cosf(2*_LIFFT_PI*i/len)};
	
	for(unsigned i = 0; i < 1000; i++){
		lifft_forward_complex(x0, out, len);
		lifft_inverse_complex(out, x1, len);
	}
	
	if(len <= 16){
		for(unsigned i = 0; i < len; i++) printf("%2d: % .2f + % .2fi\n", i, LIFFT_CREAL(x1[i]), LIFFT_CIMAG(x1[i]));
	}
	
	float err = 0;
	for(unsigned i = 0; i < len; i++) err += LIFFT_CABS(LIFFT_CSUB(x0[i], x1[i]));
	printf("err: %f\n", err);
}

int main(int argc, const char* argv[]){
	doit(16);
	doit(1 << 16);
	
	return EXIT_SUCCESS;
}
