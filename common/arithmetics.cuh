/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

/*
A part of this file has been migrated from https://github.com/ncw/iprime
Copyright (C) 2012 by Nick Craig-Wood http://www.craig-wood.com/nick/
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>

#pragma once

///
/// This is our modulo p = 2^64 - 2^32 + 1
///
const uint64_t MODULO = 0xFFFFFFFF00000001;

///
/// Arithmetic modulo p
///
 

///
/// Addition x + y (mod p)
///   Assuming that the input elements are < p, we calculate y - (p - x) because there is one single 'if' to keep the result within the range 0..p-1
///
__device__ __inline__ uint64_t add_Mod(uint64_t x, uint64_t y)
{
	x = MODULO - x;
	uint64_t d = y - x;
	if (y < x)
		d += MODULO;
	return d;
}

__device__ __inline__ uint64_t sub_Mod(uint64_t x, uint64_t y)
{
	uint64_t d = x - y;
	if (x < y)
		d += MODULO;
	return d;
}

///
/// Negate -x (mod p)
///
__device__ __inline__ uint64_t neg_Mod(uint64_t x)
{
	return MODULO - x;
}

///
/// Multiplication x * y (mod p)
///
///  
__device__ __inline__ uint64_t mul_Mod(uint64_t x, uint64_t y)
{
	// The first step is to compute a 128-bit result by a regular multiplication
	// The result is stored in two 64-bit digits (reslo and reshi)
	uint64_t reslo = x * y;
	uint64_t reshi = __umul64hi(x, y);

	// The second step is to reduce this result modulo p:
	//   Given a 128-bit number broken into 4 x 32-bit chunks : (x3, x2, x1, x0) 
	//   then (x3, x2, x1, x0) mod p = (0, x2, x1, x0-x3) because 2^96 mod p = -1
	//                               = (0, 0, x1+x2, x0-x3-x2) because 2^64 mod p = 2^32 - 1
	if (reslo >= MODULO)
		reslo -= MODULO;
	uint64_t a = (uint32_t)reshi + (reshi >> 32); // x2 + x3
	uint64_t d = a - (reshi << 32);
	if (a < reshi << 32)
		d += MODULO;
	a = reslo - d;
	if (reslo < d)
		a += MODULO;
	return a;
}

///
/// Multiplication x * y (mod p)
/// This version does not use __umul64hi to get highest part of the 64-bit by 64-bit multiplication
/// Not used, no performance improvement
///
__device__ __inline__ uint64_t mul_Mod_no_intrinsec(uint64_t x, uint64_t y)
{
	uint64_t xl = x & 0xffffffff;
	uint64_t yl = y & 0xffffffff;
	uint64_t p = xl * yl;
	uint64_t pl = p & 0xffffffff;
	uint64_t ph = p >> 32;

	x >>= 32;
	y >>= 32;

	p = (x * yl) + ph;
	uint64_t wl = p & 0xffffffff;
	uint64_t wh = p >> 32;

	p = (xl * y) + wl;

	// Now we can build the 128-bit result
	uint64_t reslo = (p << 32) + pl;
	uint64_t reshi = (x * y) + wh + (p >> 32);

	// The second step is to reduce this result modulo p:
	//   Given a 128-bit number broken into 4 x 32-bit chunks : (x3, x2, x1, x0) 
	//   then (x3, x2, x1, x0) mod p = (0, x2, x1, x0-x3) because 2^96 mod p = -1
	//                               = (0, 0, x1+x2, x0-x3-x2) because 2^64 mod p = 2^32 - 1

	if (reslo >= MODULO)
		reslo -= MODULO;
	uint64_t a = (uint32_t)reshi + (reshi >> 32); // x2 + x3
	uint64_t d = a - (reshi << 32);
	if (a < reshi << 32)
		d += MODULO;
	a = reslo - d;
	if (reslo < d)
		a += MODULO;
	return a;
}

///
/// Multiplication x * y (mod p) where one element is not larger that 32 bits
///
__device__ __inline__ uint64_t mul32_Mod(uint64_t x, uint64_t y)
{
	// The first step is to compute a 128-bit result by a regular multiplication
	// The result is stored in two 64-bit digits (reslo and reshi)
	// but the result does not exceed 96 bits
	uint64_t reslo = x * y;
	uint64_t reshi = __umul64hi(x, y); // 32 bit max

	//   Given a 96-bit number broken into 4 x 32-bit chunks : (0, x2, x1, x0) 
	//   then (0, x2, x1, x0) mod p = (0, 0, x1+x2, x0-x2) because 2^64 mod p = 2^32 - 1
	uint64_t a = ((uint64_t(0xFFFFFFFF) - reshi) << 32) + reshi + 1;
	uint64_t r = reslo - a;
	// carry
	if (reslo < a) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t sqr_Mod(uint64_t u) {
	return mul_Mod(u, u);
}

///
/// Power a ^ exp (mod p)
///
__device__ __inline__ uint64_t pow_Mod(uint64_t a, uint64_t exp)
{
	uint64_t res = 1;
	while (exp) {
		if (exp & 1)
			res = mul_Mod(res, a);
		exp >>= 1;
		a = mul_Mod(a, a);
	}
	return res;
}

///
/// Inverse 1/a (mod p)
///
/// The inverse of a is a^(p-2) mod p
/// since
///   a * a^(p-2) = a^(p-1)
///   a^(p-1) = 1 mod p (Little Fermat theorem)
///   thus a * a^(p-2) = 1 mod p
///
__device__ __inline__ uint64_t inv_Mod(uint64_t a)
{
	return pow_Mod(a, MODULO - 2);
}

//
// Shift arithmetics, the code below until the end of this file, has been migrated from https://github.com/ncw/iprime
// Copyright (C) 2012 by Nick Craig-Wood http://www.craig-wood.com/nick/
//
__device__ __inline__ uint64_t shift_0_to_31(uint64_t x, int shift) {
	uint64_t reslo = x << shift;
	uint64_t reshi = x >> (64 - shift);
	uint64_t t = ((uint64_t(0xFFFFFFFF) - reshi) << 32) + reshi + 1;
	uint64_t r = reslo - t;
	// carry
	if (reslo < t) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t shift_32_to_63(uint64_t x, int shift) {
	uint64_t xhigh = uint32_t(x >> (96 - shift));
	uint64_t xmid = uint32_t(x >> (64 - shift));
	uint64_t xlow = uint32_t(x << (shift - 32));
	uint64_t t0 = uint64_t(xmid) << 32; // (xmid, 0)
	uint64_t t1 = uint64_t(xmid);       // (0, xmid)

	t0 -= t1;             // (xmid, -xmid) no carry and must be in range 0..p-1
	t1 = uint64_t(xhigh); // (0, xhigh)
	uint64_t r = t0 - t1; // (xmid, - xhigh - xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	t0 = r;

	// add (xlow, 0) by subtacting p - (xlow, 0) = (2^32 - 1 - xlow, 1)
	t1 = ((uint64_t(0xFFFFFFFF) - xlow) << 32) + 1; // -(xlow, 0)
	r = t0 - t1;                                // (xlow + xmid, - xhigh - xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t shift_64_to_95(uint64_t x, int shift) {
	uint64_t xhigh = uint32_t(x >> (128 - shift));
	uint64_t xmid = uint32_t(x >> (96 - shift));
	uint64_t xlow = uint32_t(x << (shift - 64));

	uint64_t t0 = uint64_t(xlow) << 32; // (xlow, 0)
	uint64_t t1 = uint64_t(xlow);       // (0, xlow)

	t0 -= t1; // (xlow, -xlow) - no carry possible
	t1 = (uint64_t(xhigh) << 32) + uint64_t(xmid); // (xhigh, xmid)
	uint64_t r = t0 - t1;                        // (xlow, -xlow) - (xhigh, xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t shift(uint64_t x, int shift) {
	// Generic shifts may lead to warp divergent code, with poor performance
	switch (shift >> 5) {
	case 0:return shift_0_to_31(x, shift);
	case 1: return shift_32_to_63(x, shift);
	case 2: return shift_64_to_95(x, shift);
	default:
		return sub_Mod(0, x); // shift of 96 is negate (other cases > 96 should not happen)
	}
}

__device__ __inline__ uint64_t shift24(uint64_t x) {
	uint64_t xmid_xlow = x << 24;
	uint64_t xhigh = uint32_t(x >> (64 - 24));
	uint64_t t = ((uint64_t(0xFFFFFFFF) - xhigh) << 32) + uint64_t(xhigh + 1);
	uint64_t r = xmid_xlow - t;
	// carry
	if (xmid_xlow < t) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t shift48(uint64_t x) {
	uint64_t xhigh = uint32_t(x >> (96 - 48));
	uint64_t xmid = uint32_t(x >> (64 - 48));
	uint64_t xlow = uint32_t(x << (48 - 32));
	uint64_t t0 = uint64_t(xmid) << 32; // (xmid, 0)
	uint64_t t1 = uint64_t(xmid);       // (0, xmid)

	t0 -= t1;             // (xmid, -xmid) no carry and must be in range 0..p-1
	t1 = uint64_t(xhigh); // (0, xhigh)
	uint64_t r = t0 - t1; // (xmid, - xhigh - xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	t0 = r;

	// add (xlow, 0) by subtacting p - (xlow, 0) = (2^32 - 1 - xlow, 1)
	t1 = ((uint64_t(0xFFFFFFFF) - xlow) << 32) + 1; // -(xlow, 0)
	r = t0 - t1;                                // (xlow + xmid, - xhigh - xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	return r;
}

__device__ __inline__ uint64_t shift72(uint64_t x) {
	uint64_t xhigh = uint32_t(x >> (128 - 72));
	uint64_t xmid = uint32_t(x >> (96 - 72));
	uint64_t xlow = uint32_t(x << (72 - 64));

	uint64_t t0 = uint64_t(xlow) << 32; // (xlow, 0)
	uint64_t t1 = uint64_t(xlow);       // (0, xlow)

	t0 -= t1; // (xlow, -xlow) - no carry possible
	t1 = (uint64_t(xhigh) << 32) + uint64_t(xmid); // (xhigh, xmid)
	uint64_t r = t0 - t1;                        // (xlow, -xlow) - (xhigh, xmid)
	// carry?
	if (t0 < t1) {
		r += MODULO;
	}
	return r;
}

