#pragma once

#include "base_types.h"
#include "math.h"
#include "sse_mathfun.h"
#include <assert.h>

#define SIMD_LANES 4

#if SIMD_LANES == 1

typedef float w_float;
typedef u32 w_u32;

static inline w_u32 w_u32_write(const u32 vals[SIMD_LANES])
{
    return vals[0];
}

static inline void w_u32_read(w_u32 v, u32 out[SIMD_LANES])
{
    out[0] = v;
}

static inline w_u32 w_u32_broadcast(u32 base)
{
    return base;
}

static inline w_float w_float_broadcast(float val)
{
    return val;
}

static inline w_float w_float_write(const float vals[SIMD_LANES])
{
    return vals[0];
}

static inline void w_float_read(w_float val, float out[SIMD_LANES])
{
    out[0] = val;
}

static inline w_u32 w_u32_or(w_u32 lhs, w_u32 rhs)
{
    return lhs | rhs;
}

static inline w_u32 w_u32_xor(w_u32 lhs, w_u32 rhs)
{
    return lhs ^ rhs;
}

static inline w_u32 w_u32_and(w_u32 lhs, w_u32 rhs)
{
    return lhs & rhs;
}

static inline w_u32 w_u32_andnot(w_u32 lhs, w_u32 rhs)
{
    return lhs & ~rhs;
}

static inline w_u32 w_u32_shr(w_u32 lhs, u32 imm)
{
    return lhs >> imm;
}

static inline w_u32 w_u32_shl(w_u32 lhs, u32 imm)
{
    return lhs << imm;
}

static inline w_u32 w_u32_add(w_u32 lhs, w_u32 rhs)
{
    return lhs + rhs;
}

static inline w_float w_float_add(w_float lhs, w_float rhs)
{
    return lhs + rhs;
}

static inline w_float w_float_sub(w_float lhs, w_float rhs)
{
    return lhs - rhs;
}

static inline w_float w_float_mul(w_float lhs, w_float rhs)
{
    return lhs * rhs;
}

static inline w_float w_float_div(w_float lhs, w_float rhs)
{
    return lhs / rhs;
}

static inline w_float w_float_sqrt(w_float u)
{
    return sqrtf(u);
}

static inline w_float w_float_sin(w_float u)
{
    return sinf(u);
}

static inline w_float w_float_cos(w_float u)
{
    return cosf(u);
}

static inline w_float w_u32_pun_float(w_u32 u);
static inline w_u32 w_float_pun_u32(w_float f);

static inline w_float w_u32_cast_float(w_u32 u)
{
    return u;
}

static inline w_u32 w_float_cast_u32(w_float f)
{
    return f;
}

static inline u32 w_u32_horizontal_add(w_u32 v)
{
    return v;
}

static inline float w_float_horizontal_add(w_float v)
{
    return v;
}

static inline b32 w_u32_horizontal_and(w_u32 mask)
{
    return mask;
}

static inline w_u32 w_float_ge(w_float u, w_float v)
{
    return u >= v ? 0xFFFFFFFF : 0;
}

static inline w_u32 w_float_gt(w_float u, w_float v)
{
    return u > v ? 0xFFFFFFFF : 0;
}

static inline w_u32 w_float_le(w_float u, w_float v)
{
    return u <= v ? 0xFFFFFFFF : 0;
}

static inline w_u32 w_float_lt(w_float u, w_float v)
{
    return u < v ? 0xFFFFFFFF : 0;
}

#elif SIMD_LANES == 4

#include <emmintrin.h>

typedef struct {
    __m128i v;
} w_u32;

typedef struct {
    __m128 v;
} w_float;

static inline w_u32 w_u32_broadcast(u32 base)
{
    w_u32 res;
    res.v = _mm_set_epi32(base, base, base, base);

    return res;
}

static inline w_float w_float_broadcast(float val)
{
    w_float res;
    res.v = _mm_set_ps(val, val, val, val);

    return res;
}

static inline w_float w_float_write(const float vals[SIMD_LANES])
{
    w_float res;
    res.v = _mm_set_ps(vals[3], vals[2], vals[1], vals[0]);

    return res;
}

static inline w_u32 w_u32_or(w_u32 lhs, w_u32 rhs)
{
    w_u32 res;
    res.v = _mm_or_si128(lhs.v, rhs.v);

    return res;
}

static inline w_u32 w_u32_xor(w_u32 lhs, w_u32 rhs)
{
    w_u32 res;
    res.v = _mm_xor_si128(lhs.v, rhs.v);

    return res;
}

static inline w_u32 w_u32_and(w_u32 lhs, w_u32 rhs)
{
    w_u32 res;
    res.v = _mm_and_si128(lhs.v, rhs.v);

    return res;
}

// NOTE(octave) : compute lhs & ~rhs, whereas sse switches the arguments around
static inline w_u32 w_u32_andnot(w_u32 lhs, w_u32 rhs)
{
    w_u32 res;
    res.v = _mm_andnot_si128(rhs.v, lhs.v);

    return res;
}

static inline w_u32 w_u32_shr(w_u32 lhs, u32 imm)
{
    w_u32 res;
    res.v = _mm_srli_epi32(lhs.v, imm);

    return res;
}

static inline w_u32 w_u32_shl(w_u32 lhs, u32 imm)
{
    w_u32 res;
    res.v = _mm_slli_epi32(lhs.v, imm);

    return res;
}

static inline w_u32 w_u32_add(w_u32 lhs, w_u32 rhs)
{
    w_u32 res;
    res.v = _mm_add_epi32(lhs.v, rhs.v);

    return res;
}

static inline w_float w_float_add(w_float lhs, w_float rhs)
{
    w_float res;
    res.v = _mm_add_ps(lhs.v, rhs.v);

    return res;
}

static inline w_float w_float_sub(w_float lhs, w_float rhs)
{
    w_float res;
    res.v = _mm_sub_ps(lhs.v, rhs.v);

    return res;
}

static inline w_float w_float_mul(w_float lhs, w_float rhs)
{
    w_float res;
    res.v = _mm_mul_ps(lhs.v, rhs.v);

    return res;
}

static inline w_float w_float_div(w_float lhs, w_float rhs)
{
    w_float res;
    res.v = _mm_div_ps(lhs.v, rhs.v);

    return res;
}

static inline w_float w_float_sqrt(w_float u)
{
    w_float res;
    res.v = _mm_sqrt_ps(u.v);

    return res;
}

static inline w_float w_float_sin(w_float u)
{
    w_float res;
    res.v = sin_ps(u.v);

    return res;
}

static inline w_float w_float_cos(w_float u)
{
    w_float res;
    res.v = cos_ps(u.v);

    return res;
}

static inline w_float w_u32_pun_float(w_u32 u);
static inline w_u32 w_float_pun_u32(w_float f);

static inline w_float w_u32_cast_float(w_u32 u)
{
    w_float res;
    res.v = _mm_cvtepi32_ps(u.v);

    return res;
}

static inline w_u32 w_float_cast_u32(w_float f)
{
    w_u32 res;
    res.v = _mm_cvtps_epi32(f.v);

    return res;
}

static inline u32 w_u32_horizontal_add(w_u32 v)
{
    __m128i vshifted = _mm_srli_si128(v.v, 8);
    __m128i sum1 = _mm_add_epi32(v.v, vshifted);
    __m128i sum1shifted = _mm_srli_si128(sum1, 4);
    __m128i sum2 = _mm_add_epi32(sum1, sum1shifted);

    return _mm_cvtsi128_si32(sum2);
}

static inline b32 w_u32_horizontal_and(w_u32 mask)
{
    __m128i vshifted = _mm_srli_si128(mask.v, 8);
    __m128i and1 = _mm_and_si128(mask.v, vshifted);
    __m128i and1shifted = _mm_srli_si128(and1, 4);
    __m128i and2 = _mm_and_si128(and1, and1shifted);

    return _mm_cvtsi128_si32(and2);
}

static inline float w_float_horizontal_add(w_float v)
{
    __m128 swap_two_by_two = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sum1 = _mm_add_ps(v.v, swap_two_by_two);

    __m128 sum1shifted = _mm_movehl_ps(sum1, sum1);
    __m128 sum2 = _mm_add_ss(sum1, sum1shifted);

    return _mm_cvtss_f32(sum2);
}

static inline w_u32 w_float_ge(w_float u, w_float v)
{
    w_u32 res;
    res.v = (__m128i)_mm_cmpge_ps(u.v, v.v);

    return res;
}

static inline w_u32 w_float_gt(w_float u, w_float v)
{
    w_u32 res;
    res.v = (__m128i)_mm_cmpgt_ps(u.v, v.v);

    return res;
}

static inline w_u32 w_float_le(w_float u, w_float v)
{
    w_u32 res;
    res.v = (__m128i)_mm_cmple_ps(u.v, v.v);

    return res;
}

static inline w_u32 w_float_lt(w_float u, w_float v)
{
    w_u32 res;
    res.v = (__m128i)_mm_cmplt_ps(u.v, v.v);

    return res;
}

static inline w_u32 w_u32_write(const u32 vals[SIMD_LANES])
{
    w_u32 res;
    res.v = _mm_set_epi32(vals[3], vals[2], vals[1], vals[0]);

    return res;
}

static inline void w_u32_read(w_u32 v, u32 out[SIMD_LANES])
{
    _mm_store_si128((__m128i*)out, v.v);
}


#else
#error Only SIMD_LANES==1 or SIMD_LANES==4 supported
#endif

static inline void w_u32_conditional_assign(w_u32 mask, w_u32* lhs, w_u32 rhs)
{
    *lhs = w_u32_or(w_u32_andnot(*lhs, mask), w_u32_and(rhs, mask));
}

static inline w_float w_u32_pun_float(w_u32 u)
{
    union _ {
        w_u32 u;
        w_float f;
    };

    union _ pun;
    pun.u = u;
    return pun.f;
}

static inline w_u32 w_float_pun_u32(w_float f)
{
    union _ {
        w_u32 u;
        w_float f;
    };

    union _ pun;
    pun.f = f;
    return pun.u;
}

static inline void w_float_conditional_assign(w_u32 mask, w_float* lhs, w_float rhs)
{
    w_u32 rhs_u = w_float_pun_u32(rhs);
    w_u32 lhs_u = w_float_pun_u32(*lhs);
    w_u32_conditional_assign(mask, &lhs_u, rhs_u);
    *lhs = w_u32_pun_float(lhs_u);
}

static inline w_float w_float_masked(w_u32 mask, w_float rhs)
{
    w_u32 rhs_u = w_float_pun_u32(rhs);
    rhs_u = w_u32_and(mask, rhs_u);

    return w_u32_pun_float(rhs_u);
}

typedef union {
    struct {
        w_float x;
        w_float y;
        w_float z;
    };
    w_float co[3];
} w_v3;

static inline w_v3 w_v3_broadcast(v3 val)
{
    return (w_v3){
        w_float_broadcast(val.x),
        w_float_broadcast(val.y),
        w_float_broadcast(val.z),
    };
}

static inline w_v3 w_v3_write(const float vals[3][SIMD_LANES])
{
    return (w_v3){
        w_float_write(vals[0]),
        w_float_write(vals[1]),
        w_float_write(vals[2]),
    };
}

static inline void w_v3_conditional_assign(w_u32 mask, w_v3* lhs, w_v3 rhs)
{
    w_float_conditional_assign(mask, &lhs->x, rhs.x);
    w_float_conditional_assign(mask, &lhs->y, rhs.y);
    w_float_conditional_assign(mask, &lhs->z, rhs.z);
}

static inline w_v3 w_v3_add(w_v3 lhs, w_v3 rhs)
{
    w_v3 result = {
        .x = w_float_add(lhs.x, rhs.x),
        .y = w_float_add(lhs.y, rhs.y),
        .z = w_float_add(lhs.z, rhs.z),
    };

    return result;
}

static inline w_v3 w_v3_sub(w_v3 lhs, w_v3 rhs)
{
    w_v3 result = {
        .x = w_float_sub(lhs.x, rhs.x),
        .y = w_float_sub(lhs.y, rhs.y),
        .z = w_float_sub(lhs.z, rhs.z),
    };

    return result;
}

static inline w_v3 w_v3_hadamard(w_v3 lhs, w_v3 rhs)
{
    w_v3 result = {
        .x = w_float_mul(lhs.x, rhs.x),
        .y = w_float_mul(lhs.y, rhs.y),
        .z = w_float_mul(lhs.z, rhs.z),
    };

    return result;
}

static inline w_float w_v3_dot(w_v3 lhs, w_v3 rhs)
{
    // TODO(octave) : use madds here
    return w_float_add(w_float_add(w_float_mul(lhs.x, rhs.x), w_float_mul(lhs.y, rhs.y)), w_float_mul(lhs.z, rhs.z));
}

static inline w_v3 w_v3_scale(w_float lhs, w_v3 rhs)
{
    w_v3 result = {
        .x = w_float_mul(lhs, rhs.x),
        .y = w_float_mul(lhs, rhs.y),
        .z = w_float_mul(lhs, rhs.z),
    };

    return result;
}

static inline w_v3 w_v3_div(w_v3 lhs, w_float rhs)
{
    w_v3 result = {
        .x = w_float_div(lhs.x, rhs),
        .y = w_float_div(lhs.y, rhs),
        .z = w_float_div(lhs.z, rhs),
    };

    return result;
}

static inline w_v3 w_v3_in_basis(w_v3 coords, w_v3 x, w_v3 y, w_v3 z)
{
    return w_v3_add(w_v3_add(w_v3_scale(coords.x, x), w_v3_scale(coords.y, y)), w_v3_scale(coords.z, z));
}

static inline w_v3 w_v3_normalized(w_v3 v)
{
    w_float squared_norm = w_v3_dot(v, v);
    w_float norm = w_float_sqrt(squared_norm);

    return w_v3_div(v, norm);
}

static inline w_v3 w_v3_cross(w_v3 lhs, w_v3 rhs)
{
    w_v3 result;
    result.x = w_float_sub(w_float_mul(lhs.y, rhs.z), w_float_mul(lhs.z, rhs.y));
    result.y = w_float_sub(w_float_mul(lhs.z, rhs.x), w_float_mul(lhs.x, rhs.z));
    result.z = w_float_sub(w_float_mul(lhs.x, rhs.y), w_float_mul(lhs.y, rhs.x));

    return result;
}

static inline v3 w_v3_horizontal_add(w_v3 v)
{
    v3 result = {
        w_float_horizontal_add(v.x),
        w_float_horizontal_add(v.y),
        w_float_horizontal_add(v.z),
    };

    return result;
}

static inline w_v3 w_v3_masked(w_u32 mask, w_v3 v)
{
    w_v3 result = {
        w_float_masked(mask, v.x),
        w_float_masked(mask, v.y),
        w_float_masked(mask, v.z),
    };

    return result;
}
