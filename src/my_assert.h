#pragma once

#if 1

#define ASSERT(expr)                                                                                                   \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "%s:%d: Assertion %s failed\n", __FILE__, __LINE__, #expr);                                \
            raise(SIGTRAP);                                                                                            \
        }                                                                                                              \
    } while (0)

#else

#define ASSERT(expr)

#endif

#define ASSERT_ZERO(a, tol) ASSERT(fabs(a) < tol)

#define ASSERT_EQ(a, b, tol) ASSERT(fabs(a - b) < tol * fabs(b))

#define ASSERT_V3_NORMALIZED(v, tol) ASSERT_EQ(v3_norm(v), 1.0f, tol)
