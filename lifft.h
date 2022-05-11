#include <stdint.h>
#include <stddef.h>
#include <alloca.h>

#ifndef LIFFT_NO_STDLIB
	#include <math.h>
	#include <assert.h>
#endif

#ifndef LIFFT_FLOAT_TYPE
	#define LIFFT_FLOAT_TYPE double
#endif

typedef LIFFT_FLOAT_TYPE lifft_float_t;
#define _LIFFT_PI 3.14159265358979323846
#define _LIFFT_SQRT_2 1.4142135623730951

#if defined(LIFFT_STD_COMPLEX)
	#include <complex.h>
	typedef complex LIFFT_FLOAT_TYPE lifft_complex_t;
	
	static inline lifft_complex_t lifft_complex(lifft_float_t real, lifft_float_t imag){return real + imag*I;}
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
	static inline lifft_complex_t lifft_complex(lifft_float_t real, lifft_float_t imag){lifft_complex_t res = {real, imag}; return res;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real + y.real, x.imag + y.imag);}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real - y.real, x.imag - y.imag);}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real);}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return lifft_complex((x.real*y.real + x.imag*y.imag)/(y.real*y.real + y.imag*y.imag), (x.imag*y.real - x.real*y.imag)/(y.real*y.real + y.imag*y.imag));}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return lifft_complex(x.real, -x.imag);}
	static inline lifft_float_t lifft_cabs(lifft_complex_t x){return sqrt(x.real*x.real + x.imag*x.imag);}
	static inline lifft_float_t lifft_creal(lifft_complex_t x){return x.real;}
	static inline lifft_float_t lifft_cimag(lifft_complex_t x){return x.imag;}
	static inline lifft_complex_t lifft_cispi(lifft_float_t x){return lifft_complex(cos(2*_LIFFT_PI*x), sin(2*_LIFFT_PI*x));}
#endif

// Compute the forward FFT on complex data.
// 'n' must be a power of two.
void lifft_forward_complex(lifft_complex_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n);

// Compute the inverse FFT on complex data.
// 'n' must be a power of two.
void lifft_inverse_complex(lifft_complex_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n);

// Compute the forward DCT2 via a real valued FFT
// 'n' must be a power of two.
void lifft_forward_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n);

// Compute the inverse DCT2 via real valued iFFT.
// 'n' must be a power of two.
void lifft_inverse_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n);

#define LIFFT_APPLY_2D(func, x_in, x_out, n) { \
	typeof(*x_in)* _tmp_ = alloca(n*n*sizeof(*x_in)); \
	for(int i = 0; i < n; i++) func( x_in + i*n, 1, _tmp_ + i, n, n); \
	for(int i = 0; i < n; i++) func(_tmp_ + i*n, 1, x_out + i, n, n); \
}

#ifdef LIFFT_IMPLEMENTATION

static unsigned _lifft_setup(size_t n, size_t stride_in, size_t stride_out){
	unsigned bits = log2(n);
	// Check size.
	assert(n == 1 << bits && bits <= 32);
	// Check valid strides.
	assert(stride_in && stride_out);
	return bits;
}

// Reverse bits in an integer of up to 24 bits.
static inline size_t _lifft_rev_bits24(size_t n, unsigned bits){
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

// Cooley Tukey FFT algorithm that processes complex signal 'x' in place.
// 'x' is expected in byte-reversed index order, but shuffled back to linear order after calling.
static void _lifft_process(lifft_complex_t* x, size_t n){
	for(int stride = 1; stride < n; stride *= 2){
		lifft_complex_t wm = lifft_cispi(-0.5/stride);
		for(int i = 0; i < n; i += 2*stride){
			lifft_complex_t w = lifft_complex(1, 0);
			for(int j = 0; j < stride; j++){
				size_t idx0 = i + j, idx1 = idx0 + stride;
				lifft_complex_t p = x[idx0], q = lifft_cmul(w, x[idx1]);
				x[idx0] = lifft_cadd(p, q);
				x[idx1] = lifft_csub(p, q);
				w = lifft_cmul(w, wm);
			}
		}
	}
}

void lifft_forward_complex(lifft_complex_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out);
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n*sizeof(lifft_complex_t));
	
	// Copy to tmp[] in shufled order, apply the FFT, then copy to the output.
	for(int i = 0; i < n; i++) tmp[_lifft_rev_bits24(i, bits)] = x_in[i*stride_in];
	_lifft_process(tmp, n);
	for(int i = 0; i < n; i++) x_out[i*stride_out] = tmp[i];
}

void lifft_inverse_complex(lifft_complex_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out);
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n*sizeof(lifft_complex_t));
	
	// Compute iFFT via iFFT(x) = conj(FFT(conj(x)))/n
	lifft_complex_t scale = lifft_complex(1.0/n, 0);
	for(int i = 0; i < n; i++) tmp[_lifft_rev_bits24(i, bits)] = lifft_conj(lifft_cmul(x_in[i*stride_in], scale));
	_lifft_process(tmp, n);
	for(int i = 0; i < n; i++) x_out[i*stride_out] = lifft_conj(tmp[i]);
}

void lifft_forward_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out);
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n*sizeof(lifft_complex_t));
	
	// Pack data for DCT2 as even/odd fields.
	for(int i = 0; i < n/2; i++){
		int idx = _lifft_rev_bits24(i, bits);
		lifft_float_t xe = x_in[stride_in*(2*i + 0)], xo = x_in[stride_in*(2*i + 1)];
		tmp[idx] = lifft_complex(xe, xo), tmp[n - idx - 1] = lifft_complex(xo, xe);
	}
	
	_lifft_process(tmp, n);
	
	// Unpack the DCT2 results using the even/odd symmetry property of the FFT.
	lifft_complex_t w = lifft_complex(1, 0), wm = lifft_cispi(-0.25/n);
	for(int i = 0; i < n; i++){
		lifft_complex_t X0 = tmp[i], X1 = lifft_conj(tmp[-i & (n - 1)]);
		lifft_complex_t Xe = lifft_cadd(X0, X1), Xo = lifft_cmul(lifft_csub(X0, X1), lifft_complex(0, -1));
		x_out[i*stride_out] = lifft_creal(lifft_cmul(lifft_cadd(Xe, lifft_cmul(Xo, lifft_cmul(w, w))), w))/2;
		w = lifft_cmul(w, wm);
	}
}

void lifft_inverse_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out);
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n*sizeof(lifft_complex_t));
	
	// Pack the DCT2 data using... lots of math.
	lifft_complex_t wm = lifft_cispi(0.25/n), w = wm;
	for(int i = 1; i < n/2; i++){
		lifft_complex_t w3 = lifft_cmul(lifft_cmul(w, w), w);
		lifft_complex_t X0 = lifft_complex(x_in[stride_in*i], 0), X1 = lifft_complex(0, -x_in[stride_in*(-i & (n - 1))]);
		lifft_complex_t Xe = lifft_cmul(lifft_cadd(X0, X1), w), Xo = lifft_cmul(lifft_csub(X0, X1), w3);
		tmp[_lifft_rev_bits24( i, bits)] = lifft_complex(lifft_creal(Xe) - lifft_cimag(Xo), -lifft_cimag(Xe) - lifft_creal(Xo));
		tmp[_lifft_rev_bits24(-i, bits)] = lifft_complex(lifft_creal(Xe) + lifft_cimag(Xo), +lifft_cimag(Xe) - lifft_creal(Xo));
		w = lifft_cmul(w, wm);
	}
	
	// Fill in the special cases.
	float x0 = x_in[0], x1 = x_in[stride_in*n/2]*_LIFFT_SQRT_2;
	tmp[0] = lifft_complex(x0, -x0);
	tmp[1] = lifft_complex(x1,  x1);
	
	_lifft_process(tmp, n);
	
	// Unpack results.
	for(int i = 0; i < n/2; i++){
		x_out[stride_out*(2*i + 0)] = +lifft_creal(tmp[i])*(0.5/n);
		x_out[stride_out*(2*i + 1)] = -lifft_cimag(tmp[i])*(0.5/n);
	}
}

#endif
