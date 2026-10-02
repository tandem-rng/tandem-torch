/* Tandem8x32 reference implementation. See tandem.h and the specification. */
#include "tandem.h"

#include <string.h>
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#define CLOCK_WEYL 0x9e3779b9u
#define DOMAIN_STREAM 0x9e3779b9u
#define DOMAIN_SPLIT 0xbb67ae85u
#define DOMAIN_FORK 0xd2511f53u
#define DOMAIN_FOLD 0xcd9e8d57u
#define DOMAIN_SEED 0xa54ff53au
#define AUX_STREAM 0x94d049bbu

static const uint32_t RC[8] = {0xd17cc1b7u, 0xa7220a94u, 0xfe13abe8u, 0xfa9a6ee0u,
                               0xedb14accu, 0x9e21c820u, 0xff28b1d5u, 0xef5de2b0u};

static inline uint32_t rotl(uint32_t x, unsigned r) { return (x << r) | (x >> (32u - r)); }

void tandem_T(uint32_t o[4], uint32_t h[4]) {
    uint64_t p0 = (uint64_t)o[0] * (h[0] | 1u);
    uint64_t p1 = (uint64_t)o[2] * (h[1] | 1u);
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    uint32_t n0 = o[1] ^ hi1 ^ lo1;
    uint32_t n1 = rotl(lo1, 16) ^ h[2];
    uint32_t n2 = o[3] ^ hi0 ^ lo0;
    uint32_t n3 = rotl(lo0, 16) ^ h[3];

    h[0] ^= rotl(h[1], 7);
    h[1] ^= rotl(h[2], 13);
    h[2] ^= rotl(h[3], 22);
    h[3] ^= rotl(h[0], 3);
    h[0] = (h[0] + CLOCK_WEYL) ^ n0;

    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
    o[3] = n3;
}

void tandem_F(uint32_t o[4], uint32_t h[4]) {
    for (int r = 0; r < 8; r++) {
        uint32_t t[4];
        tandem_T(o, h);
        o[0] ^= RC[r];
        memcpy(t, o, sizeof t);
        memcpy(o, h, sizeof t);
        memcpy(h, t, sizeof t);
    }
}

void tandem_F_keyed(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                    uint32_t o[4], uint32_t h[4]) {
    o[0] = (uint32_t)counter;
    o[1] = (uint32_t)(counter >> 32);
    o[2] = domain;
    o[3] = aux;
    memcpy(h, key, 16);
    tandem_F(o, h);
}

void tandem_block(const uint32_t key[4], uint64_t c, uint32_t j, uint32_t out[4]) {
    uint32_t h[4];
    tandem_F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, out, h);
    for (uint32_t s = 0; s <= j; s++) tandem_T(out, h);
}

/* ---- Eight lanes ---------------------------------------------------------------------
 *
 * A row is the eight chunks of a group at one step, so the cache is word-major, o[word][lane],
 * and one T over a row is four element-wise operations per word. The `lanes` type holds that
 * state in registers inside the row loop. With GCC 12+ or clang it is built from vector
 * extensions and compiles to NEON or SSE/AVX; the scalar version below it is the same
 * loop written out, for other compilers or -DTANDEM_NO_SIMD. */

/* What the row loop writes: the raw words, or the words mapped to floats on the way out,
 * while the row is still in registers. A second pass over the buffer costs about as much as
 * the generation itself. */
typedef enum { STORE_RAW, STORE_F32, STORE_F64 } store_mode;

#if (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)) && !defined(TANDEM_NO_SIMD)
typedef uint32_t u32x4 __attribute__((vector_size(16)));
typedef uint64_t u64x4 __attribute__((vector_size(32)));

typedef struct {
    u32x4 o[4], h[4]; /* four lanes of each word */
} quad;

typedef struct {
    quad q[2];
} lanes;

static inline u32x4 vrotl(u32x4 x, unsigned r) { return (x << r) | (x >> (32u - r)); }

static inline void quad_T(quad *q) {
    u32x4 *o = q->o, *h = q->h;
    u64x4 p0 = __builtin_convertvector(o[0], u64x4) * __builtin_convertvector(h[0] | 1u, u64x4);
    u64x4 p1 = __builtin_convertvector(o[2], u64x4) * __builtin_convertvector(h[1] | 1u, u64x4);
    u32x4 lo0 = __builtin_convertvector(p0, u32x4), hi0 = __builtin_convertvector(p0 >> 32, u32x4);
    u32x4 lo1 = __builtin_convertvector(p1, u32x4), hi1 = __builtin_convertvector(p1 >> 32, u32x4);
    u32x4 n0 = o[1] ^ hi1 ^ lo1;
    u32x4 n1 = vrotl(lo1, 16) ^ h[2];
    u32x4 n2 = o[3] ^ hi0 ^ lo0;
    u32x4 n3 = vrotl(lo0, 16) ^ h[3];

    h[0] ^= vrotl(h[1], 7);
    h[1] ^= vrotl(h[2], 13);
    h[2] ^= vrotl(h[3], 22);
    h[3] ^= vrotl(h[0], 3);
    h[0] = (h[0] + CLOCK_WEYL) ^ n0;

    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
    o[3] = n3;
}

/* F on chunks c0 .. c0+3 at once. */
static void quad_seed(quad *q, const uint32_t key[4], uint64_t c0) {
    uint32_t lo = (uint32_t)c0;
    u32x4 counter = {lo, lo + 1u, lo + 2u, lo + 3u}, zero = {0, 0, 0, 0};
    q->o[0] = counter;
    q->o[1] = zero + (uint32_t)(c0 >> 32);
    q->o[2] = zero + DOMAIN_STREAM;
    q->o[3] = zero + AUX_STREAM;
    for (unsigned w = 0; w < 4; w++) q->h[w] = zero + key[w];
    for (int r = 0; r < 8; r++) {
        u32x4 t[4];
        quad_T(q);
        q->o[0] ^= RC[r];
        memcpy(t, q->o, sizeof t);
        memcpy(q->o, q->h, sizeof t);
        memcpy(q->h, t, sizeof t);
    }
}

typedef uint64_t u64x2 __attribute__((vector_size(16)));
typedef float f32x4 __attribute__((vector_size(16)));
typedef double f64x2 __attribute__((vector_size(16)));

static inline void store_block(u32x4 b, char *dst, store_mode mode) {
    if (mode == STORE_F32) {
#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t f = vcvtq_n_f32_u32(vshrq_n_u32((uint32x4_t)b, 8), 24);
#else
        f32x4 f = __builtin_convertvector(b >> 8, f32x4) * 0x1p-24f;
#endif
        memcpy(dst, &f, 16);
    } else if (mode == STORE_F64) {
        u64x2 w;
        memcpy(&w, &b, 16);
#if defined(__ARM_NEON) && defined(__aarch64__)
        float64x2_t f = vcvtq_n_f64_u64(vshrq_n_u64((uint64x2_t)w, 11), 53);
#else
        f64x2 f = __builtin_convertvector(w >> 11, f64x2) * 0x1p-53;
#endif
        memcpy(dst, &f, 16);
    } else {
        memcpy(dst, &b, 16);
    }
}

/* The four blocks of a quad in stream order: a 4x4 word transpose. */
static inline void quad_store(const quad *q, char *dst, store_mode mode) {
    u32x4 t0 = __builtin_shufflevector(q->o[0], q->o[1], 0, 4, 1, 5);
    u32x4 t1 = __builtin_shufflevector(q->o[2], q->o[3], 0, 4, 1, 5);
    u32x4 t2 = __builtin_shufflevector(q->o[0], q->o[1], 2, 6, 3, 7);
    u32x4 t3 = __builtin_shufflevector(q->o[2], q->o[3], 2, 6, 3, 7);
    u32x4 b0 = __builtin_shufflevector(t0, t1, 0, 1, 4, 5);
    u32x4 b1 = __builtin_shufflevector(t0, t1, 2, 3, 6, 7);
    u32x4 b2 = __builtin_shufflevector(t2, t3, 0, 1, 4, 5);
    u32x4 b3 = __builtin_shufflevector(t2, t3, 2, 3, 6, 7);
    store_block(b0, dst, mode);
    store_block(b1, dst + 16, mode);
    store_block(b2, dst + 32, mode);
    store_block(b3, dst + 48, mode);
}

static inline void lanes_load(lanes *L, const tandem_rng *rng) {
    for (unsigned w = 0; w < 4; w++)
        for (unsigned k = 0; k < 2; k++) {
            memcpy(&L->q[k].o[w], &rng->o[w][4u * k], 16);
            memcpy(&L->q[k].h[w], &rng->h[w][4u * k], 16);
        }
}

static inline void lanes_save(const lanes *L, tandem_rng *rng) {
    for (unsigned w = 0; w < 4; w++)
        for (unsigned k = 0; k < 2; k++) {
            memcpy(&rng->o[w][4u * k], &L->q[k].o[w], 16);
            memcpy(&rng->h[w][4u * k], &L->q[k].h[w], 16);
        }
}

static inline void lanes_T(lanes *L) {
    quad_T(&L->q[0]);
    quad_T(&L->q[1]);
}

static inline void lanes_seed(lanes *L, const uint32_t key[4], uint64_t g) {
    quad_seed(&L->q[0], key, 8u * g);
    quad_seed(&L->q[1], key, 8u * g + 4u);
}

static inline void lanes_store(const lanes *L, char *dst, store_mode mode) {
    quad_store(&L->q[0], dst, mode);
    quad_store(&L->q[1], dst + 64, mode);
}
#else
typedef struct {
    uint32_t o[4][8], h[4][8];
} lanes;

static inline void lanes_load(lanes *L, const tandem_rng *rng) {
    memcpy(L->o, rng->o, sizeof L->o);
    memcpy(L->h, rng->h, sizeof L->h);
}

static inline void lanes_save(const lanes *L, tandem_rng *rng) {
    memcpy(rng->o, L->o, sizeof L->o);
    memcpy(rng->h, L->h, sizeof L->h);
}

static inline void lanes_T(lanes *L) {
    for (unsigned l = 0; l < 8; l++) {
        uint32_t o[4] = {L->o[0][l], L->o[1][l], L->o[2][l], L->o[3][l]};
        uint32_t h[4] = {L->h[0][l], L->h[1][l], L->h[2][l], L->h[3][l]};
        tandem_T(o, h);
        for (unsigned w = 0; w < 4; w++) {
            L->o[w][l] = o[w];
            L->h[w][l] = h[w];
        }
    }
}

static inline void lanes_seed(lanes *L, const uint32_t key[4], uint64_t g) {
    for (unsigned l = 0; l < 8; l++) {
        uint32_t o[4], h[4];
        tandem_F_keyed(key, 8u * g + l, DOMAIN_STREAM, AUX_STREAM, o, h);
        for (unsigned w = 0; w < 4; w++) {
            L->o[w][l] = o[w];
            L->h[w][l] = h[w];
        }
    }
}

static inline void lanes_store(const lanes *L, char *dst, store_mode mode) {
    uint32_t row[32];
    for (unsigned l = 0; l < 8; l++)
        for (unsigned w = 0; w < 4; w++) row[4u * l + w] = L->o[w][l];
    if (mode == STORE_F32) {
        float f[32];
        for (unsigned i = 0; i < 32; i++) f[i] = (float)(row[i] >> 8) * 0x1p-24f;
        memcpy(dst, f, sizeof f);
    } else if (mode == STORE_F64) {
        uint64_t raw[16];
        double f[16];
        memcpy(raw, row, sizeof raw);
        for (unsigned i = 0; i < 16; i++) f[i] = (double)(raw[i] >> 11) * 0x1p-53;
        memcpy(dst, f, sizeof f);
    } else {
        memcpy(dst, row, sizeof row);
    }
}
#endif

/* ---- Rows ------------------------------------------------------------------------------ */

static inline unsigned log2k(uint32_t K) {
    unsigned s = 0;
    while ((K >> s) > 1u) s++;
    return s;
}

/* Produce rows [row, row + nrows) in stream order, 128 bytes each, into `out`, or only move
 * the cache when out is NULL. Stepping forward inside the cached group costs one T per row;
 * any other jump reseeds the group. Afterwards the cache holds the last row produced. */
static void run_rows(tandem_rng *rng, uint64_t row, size_t nrows, char *out, store_mode mode) {
    unsigned shift = log2k(rng->K);
    uint64_t mask = rng->K - 1u, at = rng->row;
    int live = rng->cached != 0;
    lanes L;

    if (nrows == 0) return;
    lanes_load(&L, rng);
    while (nrows) {
        size_t run = rng->K - (size_t)(row & mask); /* rows left in this group */
        if (!live || at != row) {
            if (live && row > at && (row >> shift) == (at >> shift)) {
                for (uint64_t r = at; r < row; r++) lanes_T(&L);
            } else {
                lanes_seed(&L, rng->key, row >> shift);
                for (uint64_t s = 0; s <= (row & mask); s++) lanes_T(&L);
            }
            live = 1;
        }
        if (run > nrows) run = nrows;
        for (size_t r = 0; r < run; r++) {
            if (r) lanes_T(&L);
            if (out) {
                lanes_store(&L, out, mode);
                out += 128;
            }
        }
        at = row + run - 1u;
        row += run;
        nrows -= run;
    }
    lanes_save(&L, rng);
    rng->row = at;
    rng->cached = 1u;
}

static inline void load_row(tandem_rng *rng, uint64_t row) {
    if (!rng->cached || rng->row != row) run_rows(rng, row, 1, NULL, STORE_RAW);
}

/* ---- Reads ----------------------------------------------------------------------------- */

static inline uint64_t align_pos(uint64_t pos, unsigned w) {
    return (pos + w - 1u) & ~((uint64_t)w - 1u);
}

static inline uint32_t word_at(const tandem_rng *rng, uint64_t p) {
    return rng->o[(p >> 5) & 3u][(p >> 7) & 7u];
}

/* w bits (1 <= w <= 64, a power of two) at an aligned position. */
static inline uint64_t read(tandem_rng *rng, uint64_t p, unsigned w) {
    load_row(rng, p >> 10);
    if (w == 64u) return word_at(rng, p) | ((uint64_t)word_at(rng, p + 32u) << 32);
    return (word_at(rng, p) >> (p & 31u)) & (0xffffffffu >> (32u - w));
}

/* Draw w bits: align, advance, read. */
static inline uint64_t next(tandem_rng *rng, unsigned w) {
    uint64_t p = align_pos(rng->pos, w);
    rng->pos = p + w;
    return read(rng, p, w);
}

static inline double to_f64(uint64_t raw) { return (double)(raw >> 11) * 0x1p-53; }
static inline float to_f32(uint32_t raw) { return (float)(raw >> 8) * 0x1p-24f; }

/* (raw >> 5) * 2^-11 as binary16 bits. Every such value is zero or a normal half whose
 * significand is the 11-bit integer k = raw >> 5, so the encoding is exact. */
static uint16_t to_f16_bits(uint16_t raw) {
    unsigned k = raw >> 5, m = 0;
    if (k == 0) return 0;
    while ((k >> m) > 1u) m++;
    return (uint16_t)(((m + 4u) << 10) | ((k << (10u - m)) & 0x3ffu));
}

/* floor(raw * 1112064 / 2^64), then skip the surrogate range. The partial products
 * a * m and b * m are below 2^53, so the high word is (a * m + (b * m >> 32)) >> 32. */
static uint32_t to_char(uint64_t raw) {
    uint64_t a = raw >> 32, b = raw & 0xffffffffu, m = 1112064u;
    uint64_t hi = (a * m + ((b * m) >> 32)) >> 32;
    return (uint32_t)(hi < 0xd800u ? hi : hi + 0x800u);
}

/* ---- Public: construction and transport ------------------------------------------------ */

tandem_rng tandem_from_key(const uint32_t key[4], uint64_t pos, uint32_t K) {
    tandem_rng rng;
    memset(&rng, 0, sizeof rng);
    memcpy(rng.key, key, 16);
    rng.pos = pos;
    rng.K = K ? K : TANDEM_DEFAULT_K;
    return rng;
}

tandem_rng tandem_seed(uint64_t seed_lo, uint64_t seed_hi, uint32_t K) {
    uint32_t o[4] = {0, 0, DOMAIN_SEED, 0};
    uint32_t h[4] = {(uint32_t)seed_lo, (uint32_t)(seed_lo >> 32), (uint32_t)seed_hi,
                     (uint32_t)(seed_hi >> 32)};
    tandem_F(o, h);
    return tandem_from_key(o, 0, K);
}

void tandem_key(const tandem_rng *rng, uint32_t key[4]) { memcpy(key, rng->key, 16); }
uint64_t tandem_position(const tandem_rng *rng) { return rng->pos; }
uint32_t tandem_chunk_length(const tandem_rng *rng) { return rng->K; }

/* ---- Public: scalar draws --------------------------------------------------------------- */

bool tandem_next_bool(tandem_rng *rng) { return next(rng, 1) != 0; }
uint8_t tandem_next_u8(tandem_rng *rng) { return (uint8_t)next(rng, 8); }
uint16_t tandem_next_u16(tandem_rng *rng) { return (uint16_t)next(rng, 16); }
uint32_t tandem_next_u32(tandem_rng *rng) { return (uint32_t)next(rng, 32); }
uint64_t tandem_next_u64(tandem_rng *rng) { return next(rng, 64); }
float tandem_next_f32(tandem_rng *rng) { return to_f32((uint32_t)next(rng, 32)); }
double tandem_next_f64(tandem_rng *rng) { return to_f64(next(rng, 64)); }
uint16_t tandem_next_f16_bits(tandem_rng *rng) { return to_f16_bits((uint16_t)next(rng, 16)); }
uint32_t tandem_next_char(tandem_rng *rng) { return to_char(next(rng, 64)); }

tandem_u128 tandem_next_u128(tandem_rng *rng) {
    uint64_t p = align_pos(rng->pos, 128);
    tandem_u128 v;
    v.lo = read(rng, p, 64);
    v.hi = read(rng, p + 64u, 64);
    rng->pos = p + 128u;
    return v;
}

void tandem_next_c32(tandem_rng *rng, float out[2]) {
    out[0] = tandem_next_f32(rng);
    out[1] = tandem_next_f32(rng);
}

void tandem_next_c64(tandem_rng *rng, double out[2]) {
    out[0] = tandem_next_f64(rng);
    out[1] = tandem_next_f64(rng);
}

/* ---- Public: fills ---------------------------------------------------------------------- */

/* After alignment to its width, every integer fill is the same little-endian byte stream:
 * single bytes up to the first row boundary, whole rows from run_rows, single bytes after. */
static void fill_raw(tandem_rng *rng, void *out, size_t n, unsigned w) {
    uint64_t p = align_pos(rng->pos, w);
    size_t nbytes = n * (w / 8u);
    char *dst = out;

    rng->pos = p + n * (uint64_t)w;
    for (; nbytes && (p & 1023u); nbytes--, p += 8u) *dst++ = (char)read(rng, p, 8);
    if (nbytes >= 128u) {
        size_t nrows = nbytes / 128u;
        run_rows(rng, p >> 10, nrows, dst, STORE_RAW);
        dst += nrows * 128u;
        p += nrows * 1024u;
        nbytes -= nrows * 128u;
    }
    for (; nbytes; nbytes--, p += 8u) *dst++ = (char)read(rng, p, 8);
}

void tandem_fill_u8(tandem_rng *rng, uint8_t *out, size_t n) { fill_raw(rng, out, n, 8); }
void tandem_fill_u16(tandem_rng *rng, uint16_t *out, size_t n) { fill_raw(rng, out, n, 16); }
void tandem_fill_u32(tandem_rng *rng, uint32_t *out, size_t n) { fill_raw(rng, out, n, 32); }
void tandem_fill_u64(tandem_rng *rng, uint64_t *out, size_t n) { fill_raw(rng, out, n, 64); }
void tandem_fill_u128(tandem_rng *rng, tandem_u128 *out, size_t n) { fill_raw(rng, out, n, 128); }

void tandem_fill_bool(tandem_rng *rng, bool *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tandem_next_bool(rng);
}

/* Whole rows of a float fill, mapped on the way out; returns how many elements remain. */
static size_t fill_rows(tandem_rng *rng, void *out, size_t n, unsigned w, store_mode mode) {
    uint64_t p = rng->pos; /* at a row boundary, or n is 0 */
    size_t per_row = 1024u / w, nrows = n / per_row;
    if (nrows == 0) return n;
    run_rows(rng, p >> 10, nrows, out, mode);
    rng->pos = p + nrows * 1024u;
    return n - nrows * per_row;
}

void tandem_fill_f32(tandem_rng *rng, float *out, size_t n) {
    rng->pos = align_pos(rng->pos, 32);
    for (; n && (rng->pos & 1023u); n--) *out++ = tandem_next_f32(rng);
    size_t rest = fill_rows(rng, out, n, 32, STORE_F32);
    out += n - rest;
    for (; rest; rest--) *out++ = tandem_next_f32(rng);
}

void tandem_fill_f64(tandem_rng *rng, double *out, size_t n) {
    rng->pos = align_pos(rng->pos, 64);
    for (; n && (rng->pos & 1023u); n--) *out++ = tandem_next_f64(rng);
    size_t rest = fill_rows(rng, out, n, 64, STORE_F64);
    out += n - rest;
    for (; rest; rest--) *out++ = tandem_next_f64(rng);
}

void tandem_fill_f16_bits(tandem_rng *rng, uint16_t *out, size_t n) {
    fill_raw(rng, out, n, 16);
    for (size_t i = 0; i < n; i++) out[i] = to_f16_bits(out[i]);
}

void tandem_fill_char(tandem_rng *rng, uint32_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = tandem_next_char(rng);
}

void tandem_fill_c32(tandem_rng *rng, float *out, size_t n) { tandem_fill_f32(rng, out, 2u * n); }
void tandem_fill_c64(tandem_rng *rng, double *out, size_t n) { tandem_fill_f64(rng, out, 2u * n); }

/* ---- Public: random access and derived generators --------------------------------------- */

/* Element i of the fill that would start here. Works on a copy, so the cache stays put. */
static uint64_t at(const tandem_rng *rng, uint64_t i, unsigned w) {
    tandem_rng tmp = *rng;
    return read(&tmp, align_pos(rng->pos, w) + i * w, w);
}

uint32_t tandem_at_u32(const tandem_rng *rng, uint64_t i) { return (uint32_t)at(rng, i, 32); }
uint64_t tandem_at_u64(const tandem_rng *rng, uint64_t i) { return at(rng, i, 64); }
float tandem_at_f32(const tandem_rng *rng, uint64_t i) { return to_f32((uint32_t)at(rng, i, 32)); }
double tandem_at_f64(const tandem_rng *rng, uint64_t i) { return to_f64(at(rng, i, 64)); }

static tandem_rng child(const tandem_rng *rng, uint64_t counter, uint32_t domain, uint32_t aux,
                        unsigned half) {
    uint32_t o[4], h[4];
    tandem_F_keyed(rng->key, counter, domain, aux, o, h);
    return tandem_from_key(half ? h : o, 0, rng->K);
}

tandem_rng tandem_split(const tandem_rng *rng, uint64_t index) {
    return child(rng, index >> 1, DOMAIN_SPLIT, 0, (unsigned)(index & 1u));
}

tandem_rng tandem_sub(const tandem_rng *rng, uint64_t purpose) {
    return child(rng, purpose, DOMAIN_FOLD, 0, 0);
}

void tandem_fork(tandem_rng *parent, tandem_rng *children, uint64_t n) {
    uint64_t b = parent->pos >> 7;
    for (uint64_t i = 0; i < n; i++)
        children[i] = child(parent, b, DOMAIN_FORK, (uint32_t)(i >> 1), (unsigned)(i & 1u));
    parent->pos = (b + 1u) << 7;
}
