#pragma once

#include <stdint.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint8_t u8;
typedef u32 b32;

typedef union {
    struct {
        float x;
        float y;
        float z;
    };
    float co[3];
} v3;
