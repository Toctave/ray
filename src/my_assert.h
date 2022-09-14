#pragma once

#define ASSERT_ZERO(a, tol) ASSERT(fabs(a) < tol)

#define ASSERT_EQ(a, b, tol) ASSERT(fabs(a - b) < tol * fabs(b))

#define ASSERT_V3_NORMALIZED(v, tol) ASSERT_EQ(v3_norm(v), 1.0f, tol)
