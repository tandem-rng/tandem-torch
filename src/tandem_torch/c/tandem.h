/* Tandem8x32: a noncryptographic pseudorandom number generator, fast on CPUs and GPUs alike.
 *
 * Reference C implementation of https://github.com/tandem-rng/spec.
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#ifndef TANDEM_H
#define TANDEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TANDEM_DEFAULT_K 32u

/* A generator is its transport form (key, bit position, chunk length K) plus a cache of
 * the eight chunk states that make up the current 1024-bit row. Copy it freely: the cache
 * is a pure function of the transport form, stored word-major (o[word][lane]). Treat every
 * field as private. */
typedef struct {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;
    uint32_t cached;
    uint64_t row;
    uint32_t o[4][8];
    uint32_t h[4][8];
} tandem_rng;

/* Constructors. K is a power of two in [1, 65536]; pass 0 for the default of 32. */
tandem_rng tandem_from_key(const uint32_t key[4], uint64_t pos, uint32_t K);
tandem_rng tandem_seed(uint64_t seed_lo, uint64_t seed_hi, uint32_t K);

/* Transport form. */
void tandem_key(const tandem_rng *rng, uint32_t key[4]);
uint64_t tandem_position(const tandem_rng *rng);
uint32_t tandem_chunk_length(const tandem_rng *rng);

/* 128-bit words as two 64-bit halves, little-endian. */
typedef struct {
    uint64_t lo, hi;
} tandem_u128;

/* Scalar draws: align to the width, read, advance. */
bool tandem_next_bool(tandem_rng *rng);
uint8_t tandem_next_u8(tandem_rng *rng);
uint16_t tandem_next_u16(tandem_rng *rng);
uint32_t tandem_next_u32(tandem_rng *rng);
uint64_t tandem_next_u64(tandem_rng *rng);
tandem_u128 tandem_next_u128(tandem_rng *rng);
float tandem_next_f32(tandem_rng *rng);
double tandem_next_f64(tandem_rng *rng);
/* IEEE binary16 bit pattern of a uniform draw in [0, 1), the spec's Float16 mapping. */
uint16_t tandem_next_f16_bits(tandem_rng *rng);
/* A uniform Unicode scalar value, 64 stream bits per draw. */
uint32_t tandem_next_char(tandem_rng *rng);
/* Complex draws: out[0] is the real part, out[1] the imaginary part. */
void tandem_next_c32(tandem_rng *rng, float out[2]);
void tandem_next_c64(tandem_rng *rng, double out[2]);

/* Fills: n aligned elements, the same values as n scalar draws. */
void tandem_fill_bool(tandem_rng *rng, bool *out, size_t n);
void tandem_fill_u8(tandem_rng *rng, uint8_t *out, size_t n);
void tandem_fill_u16(tandem_rng *rng, uint16_t *out, size_t n);
void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n);
void tandem_fill_u64(tandem_rng *rng, uint64_t *out, size_t n);
void tandem_fill_u128(tandem_rng *rng, tandem_u128 *out, size_t n);
void tandem_fill_f32(tandem_rng *rng, float *out, size_t n);
void tandem_fill_f64(tandem_rng *rng, double *out, size_t n);
void tandem_fill_f16_bits(tandem_rng *rng, uint16_t *out, size_t n);
void tandem_fill_char(tandem_rng *rng, uint32_t *out, size_t n);
/* n complex values as 2n interleaved components. */
void tandem_fill_c32(tandem_rng *rng, float *out, size_t n);
void tandem_fill_c64(tandem_rng *rng, double *out, size_t n);

/* Random access: element i of the fill that would start here, without advancing. */
uint32_t tandem_at_u32(const tandem_rng *rng, uint64_t i);
uint64_t tandem_at_u64(const tandem_rng *rng, uint64_t i);
float tandem_at_f32(const tandem_rng *rng, uint64_t i);
double tandem_at_f64(const tandem_rng *rng, uint64_t i);

/* Derived generators. Children start at position 0 with the parent's K. */
tandem_rng tandem_split(const tandem_rng *rng, uint64_t index);
tandem_rng tandem_sub(const tandem_rng *rng, uint64_t purpose);
/* Fork n children from the parent's current block and move the parent past it. */
void tandem_fork(tandem_rng *parent, tandem_rng *children, uint64_t n);

/* Building blocks, exposed for conformance tests and ports. */
void tandem_T(uint32_t o[4], uint32_t h[4]);
void tandem_F(uint32_t o[4], uint32_t h[4]);
void tandem_F_keyed(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                    uint32_t o[4], uint32_t h[4]);
/* Block B(c, j): the exposed half of chunk c after j + 1 steps. */
void tandem_block(const uint32_t key[4], uint64_t c, uint32_t j, uint32_t out[4]);

#ifdef __cplusplus
}
#endif

#endif /* TANDEM_H */
