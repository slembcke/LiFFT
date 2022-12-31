// Compute the forward DCT II.
// 'x_in' and 'x_out' must be length 'n'.
// 'scratch' must be length 'n/2'.
// 'n' must be a power of two.
void lifft_forward_dct(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n);

// Compute the inverse DCT II (DCT III).
// 'x_in' and 'x_out' must be length 'n'.
// 'scratch' must be length 'n/2 + 1'.
// 'n' must be a power of two.
void lifft_inverse_dct(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n);

#ifdef LIFFT_IMPLEMENTATION

void lifft_forward_dct(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	
	// To calculate the DCT II, you need to double and mirror x_in, BUT!
	// That means evens and odds are mirrored, so you can compute the other via symmetry.
	for(size_t i = 0; i < n/4; i++){
		scratch[_lifft_rev_bits24(i, bits)] = lifft_complex(x_in[(4*i + 0)*stride_in], x_in[(4*i + 2)*stride_in]);
		scratch[_lifft_rev_bits24(n/2 - i - 1, bits)] = lifft_complex(x_in[(4*i + 3)*stride_in], x_in[(4*i + 1)*stride_in]);
	}
	
	// Compute real valued FFT
	_lifft_process(scratch, n/2);
	lifft_complex_t w0 = lifft_complex(0, -1), wm0 = lifft_cispi((lifft_float_t)-2.0/n);
	for(size_t i = 0; i <= n/4; i++){
		// Unpack using even/odd fft symmetry
		lifft_complex_t p = scratch[i], q = lifft_conj(scratch[-i&(n/2 - 1)]);
		lifft_complex_t xe = lifft_cadd(p, q), xo = lifft_cmul(lifft_csub(p, q), w0);
		w0 = lifft_cmul(w0, wm0);
		
		// Apply final stage of Cooley Tukey
		scratch[i] = lifft_cadd(xe, xo);
		scratch[(n/2 - i)] = lifft_conj(lifft_csub(xe, xo));
	}
	
	// TODO can these loops be fused?
	// Compute the DCT II using the even/odd symmetry.
	lifft_complex_t w1 = lifft_complex(1, 0), wm1 = lifft_cispi((lifft_float_t)-0.5/n);
	for(size_t i = 0; i <= n/2; i++){
		lifft_complex_t p = lifft_cmul(scratch[i], w1);
		w1 = lifft_cmul(w1, wm1);
		
		x_out[(-i&(n - 1))*stride_out] = -lifft_cimag(p);
		x_out[i*stride_out] = lifft_creal(p);
	}
}

// TODO this is quite the mess. See if I can compute this from the DCT III directly instead.
void lifft_inverse_dct(lifft_float_t x_in[], size_t stride_in, lifft_float_t x_out[], size_t stride_out, lifft_complex_t scratch[], size_t n){
	unsigned bits = _lifft_setup(n, stride_in, stride_out) - 1;
	
	lifft_complex_t wm = lifft_cispi((lifft_float_t)0.5/n), w = lifft_complex(0.5, 0);
	scratch[0] = lifft_cmul(lifft_complex(x_in[0], 0), w);
	w = lifft_cmul(w, wm);
	
	for(size_t i = 1; i <= n/2; i++){
		scratch[i] = lifft_cmul(lifft_complex(x_in[i*stride_in], -x_in[(n - i)*stride_in]), w);
		w = lifft_cmul(w, wm);
	}
	
	// TODO why does this need so many temporary array. Aurghrgrh!
	lifft_float_t scratch_real[n];
	lifft_complex_t scratch2[n];
	lifft_inverse_real(scratch, 1, scratch_real, 1, scratch2, n);
	
	for(size_t i = 0; i < n/4; i++){
		x_out[(4*i + 0)*stride_out] = scratch_real[2*i + 0];
		x_out[(4*i + 1)*stride_out] = scratch_real[n - 2*i - 1];
		x_out[(4*i + 2)*stride_out] = scratch_real[2*i + 1];
		x_out[(4*i + 3)*stride_out] = scratch_real[n - 2*i - 2];
	}
}

#endif
