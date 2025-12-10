// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Scott Lembcke and Howling Moon Software

// Compute a DCT II.
// 'x_in' and 'x_out' must be length 'n'.
// 'scratch' must be length 'n/2'.
// 'n' must be a power of two.
void lifft_dct2(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n);

// Compute a DCT III.
// Scaled to be an exact inverse to lifft_dct2().
// 'x_in' and 'x_out' must be length 'n'.
// 'scratch' must be length 'n/2'.
// 'n' must be a power of two.
void lifft_dct3(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n);

#ifdef LIFFT_IMPLEMENTATION

void lifft_dct2(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	
	for(size_t i = 0; i < n/4; i++){
		scratch[_lifft_rev_bits18(i, bits)] = lifft_complex(x_in[(4*i + 0)*stride_in], x_in[(4*i + 2)*stride_in]);
		scratch[_lifft_rev_bits18(n/2 - i - 1, bits)] = lifft_complex(x_in[(4*i + 3)*stride_in], x_in[(4*i + 1)*stride_in]);
	}
	
	_lifft_process(scratch, n/2);
	lifft_complex_t w0 = lifft_complex(0, -1), wm0 = lifft_cispi((lifft_float_t)-2.0/n);
	lifft_complex_t w1 = lifft_complex(1,  0), wm1 = lifft_cispi((lifft_float_t)-0.5/n);
	for(size_t i0 = 0; i0 < n/2; i0++){
		size_t i1 = -i0 & (n/2 - 1);
		lifft_complex_t p = lifft_cmul(scratch[i0], w1), q = lifft_cmul(lifft_conj(scratch[i1]), w1);
		lifft_complex_t s = lifft_cadd(p, q), t = lifft_cmul(lifft_csub(p, q), w0);
		w0 = lifft_cmul(w0, wm0);
		
		x_out[(i0      )*stride_out] = lifft_creal(lifft_cadd(s, t));
		x_out[(i0 + n/2)*stride_out] = lifft_creal(lifft_cmul(lifft_csub(s, t), lifft_complex(_LIFFT_SQRT_2_2, -_LIFFT_SQRT_2_2)));
		w1 = lifft_cmul(w1, wm1);
	}
}

void lifft_dct3(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	
	lifft_float_t p = x_in[0*stride_in]*(lifft_float_t)0.5/n, q = x_in[(n/2)*stride_in]*_LIFFT_SQRT_2_2/n;
	scratch[0] = lifft_complex(p - q, p + q);
	
	lifft_complex_t wm0 = lifft_cispi((lifft_float_t)-2.0/n), w0 = wm0;
	lifft_complex_t wm1 = lifft_cispi((lifft_float_t)-0.5/n), w1 = lifft_cmul(wm1, lifft_complex((lifft_float_t)0.5/n, 0));
	for(size_t i0 = 1; i0 < n/2; i0++){
		size_t i1 = -i0 & (n - 1);
		lifft_complex_t p = lifft_complex(x_in[(i0      )*stride_in], +x_in[(i1      )*stride_in]);
		lifft_complex_t q = lifft_complex(x_in[(i1 - n/2)*stride_in], -x_in[(i0 + n/2)*stride_in]);
		q = lifft_cmul(q, lifft_complex(_LIFFT_SQRT_2_2, _LIFFT_SQRT_2_2));
		
		lifft_complex_t s = lifft_cmul(lifft_cadd(p, q), lifft_complex(0, 1));
		lifft_complex_t t = lifft_cmul(lifft_csub(p, q), w0);
		w0 = lifft_cmul(w0, wm0);
		
		scratch[_lifft_rev_bits18(i0, bits)] = lifft_cmul(lifft_cadd(s,t), w1);
		w1 = lifft_cmul(w1, wm1);
	}
	
	_lifft_process(scratch, n/2);
	for(size_t i = 0; i < n/4; i++){
		lifft_complex_t p = scratch[i];
		lifft_complex_t q = scratch[n/2 - 1 - i];
		x_out[(4*i + 0)*stride_out] = lifft_cimag(p);
		x_out[(4*i + 1)*stride_out] = lifft_creal(q);
		x_out[(4*i + 2)*stride_out] = lifft_creal(p);
		x_out[(4*i + 3)*stride_out] = lifft_cimag(q);
	}
}

#endif
