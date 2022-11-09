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
#define _LIFFT_PI ((lifft_float_t)3.14159265358979323846)
#define _LIFFT_SQRT_2 ((lifft_float_t)1.4142135623730951)

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
	static inline lifft_complex_t lifft_cispi(lifft_complex_t x){return cexp((_LIFFT_PI*I)*x);}
#elif !defined(LIFFT_COMPLEX_TYPE)
	typedef struct {lifft_float_t real, imag;} lifft_complex_t;
	static inline lifft_complex_t lifft_complex(lifft_float_t real, lifft_float_t imag){lifft_complex_t res = {real, imag}; return res;}
	static inline lifft_complex_t lifft_cadd(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real + y.real, x.imag + y.imag);}
	static inline lifft_complex_t lifft_csub(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real - y.real, x.imag - y.imag);}
	static inline lifft_complex_t lifft_cmul(lifft_complex_t x, lifft_complex_t y){return lifft_complex(x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real);}
	static inline lifft_complex_t lifft_cdiv(lifft_complex_t x, lifft_complex_t y){return lifft_complex((x.real*y.real + x.imag*y.imag)/(y.real*y.real + y.imag*y.imag), (x.imag*y.real - x.real*y.imag)/(y.real*y.real + y.imag*y.imag));}
	static inline lifft_complex_t lifft_conj(lifft_complex_t x){return lifft_complex(x.real, -x.imag);}
	static inline lifft_float_t lifft_cabs(lifft_complex_t x){return (lifft_float_t)sqrt(x.real*x.real + x.imag*x.imag);}
	static inline lifft_float_t lifft_creal(lifft_complex_t x){return x.real;}
	static inline lifft_float_t lifft_cimag(lifft_complex_t x){return x.imag;}
	static inline lifft_complex_t lifft_cispi(lifft_float_t x){return lifft_complex((lifft_float_t)cos(_LIFFT_PI*x), (lifft_float_t)sin(_LIFFT_PI*x));}
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
	unsigned bits = (unsigned)log2(n);
	// Check size.
	assert(n == 1u << bits && bits <= 32u);
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
	for(size_t stride = 1; stride < n; stride *= 2){
		lifft_complex_t wm = lifft_cispi(-1/(lifft_float_t)stride);
		for(size_t i = 0; i < n; i += 2*stride){
			lifft_complex_t w = lifft_complex(1, 0);
			for(size_t j = 0; j < stride; j++){
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
	for(size_t i = 0; i < n; i++) tmp[_lifft_rev_bits24(i, bits)] = x_in[i*stride_in];
	_lifft_process(tmp, n);
	for(size_t i = 0; i < n; i++) x_out[i*stride_out] = tmp[i];
}

void lifft_inverse_complex(lifft_complex_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out);
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n*sizeof(lifft_complex_t));
	
	// Compute iFFT via iFFT(x) = conj(FFT(conj(x)))/n
	lifft_complex_t scale = lifft_complex((lifft_float_t)1.0/n, 0);
	for(size_t i = 0; i < n; i++) tmp[_lifft_rev_bits24(i, bits)] = lifft_conj(lifft_cmul(x_in[i*stride_in], scale));
	_lifft_process(tmp, n);
	for(size_t i = 0; i < n; i++) x_out[i*stride_out] = lifft_conj(tmp[i]);
}

void _lifft_extract_real(lifft_complex_t* x_in, lifft_complex_t* x_out, size_t stride_out, size_t n){
	lifft_complex_t w = lifft_complex(0, -1), wm = lifft_cispi((lifft_float_t)-2.0/n);
	for(size_t i = 0; i <= n/4; i++){
		// Unpack using even/odd fft symmetry
		lifft_complex_t p = x_in[i], q = lifft_conj(x_in[-i&(n/2 - 1)]);
		lifft_complex_t xe = lifft_cadd(p, q), xo = lifft_cmul(lifft_csub(p, q), w);
		w = lifft_cmul(w, wm);
		
		// Apply final stage of Cooley Tukey
		x_out[i*stride_out] = lifft_cadd(xe, xo);
		x_out[(n/2 - i)*stride_out] = lifft_conj(lifft_csub(xe, xo));
	}
}

void lifft_forward_real(lifft_float_t* x_in, size_t stride_in, lifft_complex_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n/2*sizeof(lifft_complex_t));
	
	// Copy as [evens + odds*im]
	for(size_t i = 0; i < n/2; i++) tmp[_lifft_rev_bits24(i, bits)] = lifft_complex(x_in[(2*i + 0)*stride_in]/2, x_in[(2*i + 1)*stride_in]/2);
	_lifft_process(tmp, n/2);
	_lifft_extract_real(tmp, x_out, stride_out, n);
}

void lifft_inverse_real(lifft_complex_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n/2*sizeof(lifft_complex_t));
	
	lifft_complex_t w = lifft_complex(0, 1), wm = lifft_cispi((lifft_float_t)2.0/n);
	for(size_t i = 0; i <= n/4; i++){
		// Calculate evens/odds from real fft symmetry
		lifft_complex_t p = x_in[i*stride_in], q = lifft_conj(x_in[(n/2 - i)*stride_in]);
		lifft_complex_t xe = lifft_cadd(p, q), xo = lifft_cmul(lifft_csub(p, q), w);
		w = lifft_cmul(w, wm);
		
		// Pack using even/odd symetry
		tmp[_lifft_rev_bits24(i, bits)] = lifft_conj(lifft_cadd(xe, xo));
		tmp[_lifft_rev_bits24(-i & (n/2 - 1), bits)] = lifft_csub(xe, xo);
	}
	
	_lifft_process(tmp, n/2);
	
	// Extract evens from real and odd from imag
	for(size_t i = 0; i < n/2; i++){
		x_out[(2*i + 0)*stride_out] = +lifft_creal(tmp[i])/n;
		x_out[(2*i + 1)*stride_out] = -lifft_cimag(tmp[i])/n;
	}
}

void lifft_forward_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	lifft_complex_t* tmp = (lifft_complex_t*)alloca(n/2*sizeof(lifft_complex_t));
	
	// To calculate the DCT II, you need to double and mirror x_in, BUT!
	// That means evens and odds are mirrored, so you can compute the other via symmetry.
	for(size_t i = 0; i < n/4; i++){
		tmp[_lifft_rev_bits24(i, bits)] = lifft_complex(x_in[(4*i + 0)*stride_in], x_in[(4*i + 2)*stride_in]);
		tmp[_lifft_rev_bits24(n/2 - i - 1, bits)] = lifft_complex(x_in[(4*i + 3)*stride_in], x_in[(4*i + 1)*stride_in]);
	}
	
	// Compute the real fft
	_lifft_process(tmp, n/2);
	_lifft_extract_real(tmp, tmp, 1, n);
	
	// Compute the DCT II from the FFT
	lifft_complex_t w = lifft_complex(1, 0), wm = lifft_cispi((lifft_float_t)-0.5/n);
	for(size_t i = 0; i <= n/2; i++){
		lifft_complex_t p = lifft_cmul(tmp[i], w);
		w = lifft_cmul(w, wm);
		
		x_out[(-i&(n - 1))*stride_out] = -lifft_cimag(p);
		x_out[i*stride_out] = lifft_creal(p);
	}
}

// TODO this is quite the mess. See if I can compute this from the DCT III directly instead.
void lifft_inverse_dct(lifft_float_t* x_in, size_t stride_in, lifft_float_t* x_out, size_t stride_out, size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	lifft_complex_t* tmp = (lifft_complex_t*)alloca((n/2 + 1)*sizeof(lifft_complex_t));
	lifft_complex_t* tmp2 = (lifft_complex_t*)alloca((n/2 + 1)*sizeof(lifft_complex_t));
	
	lifft_complex_t wm = lifft_cispi((lifft_float_t)0.5/n), w = lifft_complex(0.5, 0);
	tmp[0] = lifft_cmul(lifft_complex(x_in[0], 0), w);
	w = lifft_cmul(w, wm);
	for(size_t i = 1; i <= n/2; i++){
		tmp[i] = lifft_cmul(lifft_complex(x_in[i*stride_in], -x_in[(n - i)*stride_in]), w);
		w = lifft_cmul(w, wm);
	}
	
	lifft_float_t xe[n];
	lifft_inverse_real(tmp, 1, xe, 1, n);
	
	for(size_t i = 0; i < n/4; i++){
		x_out[(4*i + 0)*stride_out] = xe[2*i + 0];
		x_out[(4*i + 1)*stride_out] = xe[n - 2*i - 1];
		x_out[(4*i + 2)*stride_out] = xe[2*i + 1];
		x_out[(4*i + 3)*stride_out] = xe[n - 2*i - 2];
	}
}

#endif
