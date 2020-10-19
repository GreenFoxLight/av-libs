/*
    This file is in the public domain. Where
    a public domain declaration is not recognized, you are granted
    a license to freely use, modify, and redistribute it in
    any way you choose.
 */

#ifndef AV_MATH_H
#define AV_MATH_H

/* Small math library, should compile as C89.
 *
 * I intended to use this for a personal project of mine, that never came to
 * pass; so here it is for everyone who might need something like this.
 * I don't claim that this is particularily optimized, so you are probably
 * better of using something like glm.
 *
 * ~ Kevin Trogant
 */

/* TODO:
 * Quaternions, generating rotation and projection matrices etc.
 */

/* You can some functionality by defining the appropriate macros */

/* common definitions */
#ifndef AV_INT8
    #define AV_INT8 signed char
#endif
#ifndef AV_INT16
    #define AV_INT16 signed short
#endif
#ifndef AV_INT32
    #define AV_INT32 signed int
#endif
#ifndef AV_INT64
    #define AV_INT64 signed long long
#endif
#ifndef AV_UINT8
    #define AV_UINT8 unsigned char
#endif
#ifndef AV_UINT16
    #define AV_UINT16 unsigned short
#endif
#ifndef AV_UINT32
    #define AV_UINT32 unsigned int
#endif
#ifndef AV_UINT64
    #define AV_UINT64 unsigned long long
#endif
typedef AV_INT8   av_int8;
typedef AV_INT16  av_int16;
typedef AV_INT32  av_int32;
typedef AV_INT64  av_int64;
typedef AV_UINT8  av_uint8;
typedef AV_UINT16 av_uint16;
typedef AV_UINT32 av_uint32;
typedef AV_UINT64 av_uint64;
typedef float     av_float32;
typedef double    av_float64;

#ifndef AV_FLOAT_TYPE
    #define AV_FLOAT_TYPE av_float32 
#endif
#ifndef AV_INT_TYPE
    #define AV_INT_TYPE av_int32 
#endif

/* default int and float types.
 * these are used for all avmath.h
 * types and functions.
 */
typedef AV_FLOAT_TYPE av_float;
typedef AV_INT_TYPE av_int;

#ifndef AV_API
    #ifdef __cplusplus
        #define AV_API extern "C"
    #else
        #define AV_API extern
    #endif
#endif
#ifdef __cplusplus
    #ifndef AV_CPPI_API
        #define AV_CPP_API static inline
    #endif
#endif
#ifndef AV_PRIVATE
    #define AV_PRIVATE static
#endif

#ifndef AV_UNUSED
   /* The traditional (void)x is not suficient,
    * because it evaluates x.
    * sizeof(x) does not
    */ 
    #define AV_UNUSED(X) ((void)sizeof(X))
#endif

/* Enables or disables check for 0 determinant
 * in the matrix inverse functions
 */
#ifndef AV_CHECK_DETERMINANT
    #define AV_CHECK_DETERMINANT 1
#endif

/* Used for floating point comparisons */
#ifndef AV_EPSILON
    #define AV_EPSILON 1e-4f 
#endif

/* Floating point types */
typedef struct av_vec2f {
    av_float x;
    av_float y;
} av_vec2f;

/** 3d vector of floats */
typedef struct av_vec3f {
    av_float x;
    av_float y;
    av_float z;
} av_vec3f;

/** 4d vector of floats */
typedef struct av_vec4f {
    av_float x;
    av_float y;
    av_float z;
    av_float w;
} av_vec4f;

/* Note that these matrices are in column-major order.
 * If you want to use these in OpenGL shaders, you can
 * pre-multiply them to vectors.
 **/

/** 3x3 matrix of floats */
typedef struct av_mat3x3f {
    av_float v00, v10, v20;
    av_float v01, v11, v21;
    av_float v02, v12, v22;
} av_mat3x3f;

/** 4x4 matrix of floats */
typedef struct av_mat4x4f {
    av_float v00, v10, v20, v30;
    av_float v01, v11, v21, v31;
    av_float v02, v12, v22, v32;
    av_float v03, v13, v23, v33;
} av_mat4x4f;

/* Integer types */
typedef struct av_vec2i {
    av_int x;
    av_int y;
} av_vec2i;

/** 3d vector of integers */
typedef struct av_vec3i {
    av_int x;
    av_int y;
    av_int z;
} av_vec3i;

/** 4d vector of integers */
typedef struct av_vec4i {
    av_int x;
    av_int y;
    av_int z;
    av_int w;
} av_vec4i;

/** 3x3 matrix of integers */
typedef struct av_mat3x3i {
    av_int v00, v10, v20;
    av_int v01, v11, v21;
    av_int v02, v12, v22;
} av_mat3x3i;

/** 4x4 matrix of integers */
typedef struct av_mat4x4i {
    av_int v00, v10, v20, v30;
    av_int v01, v11, v21, v31;
    av_int v02, v12, v22, v32;
    av_int v03, v13, v23, v33;
} av_mat4x4i;

/* Vector creation and conversion */
AV_API av_vec2f av_make_vec2f(av_float x, av_float y);
AV_API av_vec3f av_make_vec3f(av_float x, av_float y, av_float z); 
AV_API av_vec4f av_make_vec4f(av_float x, av_float y, av_float z, av_float w);
AV_API av_vec2i av_make_vec2i(av_int x, av_int y);
AV_API av_vec3i av_make_vec3i(av_int x, av_int y, av_int z);
AV_API av_vec4i av_make_vec4i(av_int x, av_int y, av_int z, av_int w);
AV_API av_vec2f av_vec2i2f(av_vec2i);
AV_API av_vec3f av_vec3i2f(av_vec3i);
AV_API av_vec4f av_vec4i2f(av_vec4i);
AV_API av_vec2i av_vec2f2i(av_vec2f);
AV_API av_vec3i av_vec3f2i(av_vec3f);
AV_API av_vec4i av_vec4f2i(av_vec4f);

/* Matrix creation and conversion */
AV_API av_mat3x3f av_make_mat3x3f_identity(void);
AV_API av_mat4x4f av_make_mat4x4f_identity(void);
AV_API av_mat3x3f av_make_mat3x3f_zero(void);
AV_API av_mat4x4f av_make_mat4x4f_zero(void);
AV_API av_mat3x3f av_make_mat3x3f(av_float v00, av_float v10, av_float v20,
                                   av_float v01, av_float v11, av_float v21,
                                   av_float v02, av_float v12, av_float v22);
AV_API av_mat4x4f av_make_mat4x4f(
                av_float v00, av_float v10, av_float v20, av_float v30,
                av_float v01, av_float v11, av_float v21, av_float v31,
                av_float v02, av_float v12, av_float v22, av_float v32,
                av_float v03, av_float v13, av_float v23, av_float v33);
AV_API av_mat3x3f av_make_mat3x3fv(const av_float *v);
AV_API av_mat4x4f av_make_mat4x4fv(const av_float *v);
AV_API av_mat3x3i av_make_mat3x3i_identity(void);
AV_API av_mat4x4i av_make_mat4x4i_identity(void);
AV_API av_mat3x3i av_make_mat3x3i_zero(void);
AV_API av_mat4x4i av_make_mat4x4i_zero(void);
AV_API av_mat3x3i av_make_mat3x3i(av_int v00, av_int v10, av_int v20,
                                   av_int v01, av_int v11, av_int v21,
                                   av_int v02, av_int v12, av_int v22);
AV_API av_mat4x4i av_make_mat4x4i(
                        av_int v00, av_int v10, av_int v20, av_int v30,
                        av_int v01, av_int v11, av_int v21, av_int v31,
                        av_int v02, av_int v12, av_int v22, av_int v32,
                        av_int v03, av_int v13, av_int v23, av_int v33);
AV_API av_mat3x3i av_make_mat3x3iv(const av_int *v);
AV_API av_mat4x4i av_make_mat4x4iv(const av_int *v);
AV_API av_mat3x3f av_mat3x3i2f(const av_mat3x3i *);
AV_API av_mat4x4f av_mat4x4i2f(const av_mat4x4i *);
AV_API av_mat3x3i av_mat3x3f2i(const av_mat3x3f *);
AV_API av_mat4x4i av_mat4x4f2i(const av_mat4x4f *);

AV_API av_vec3f av_mat3x3f_get_column(const av_mat3x3f *m, unsigned int i);
AV_API av_vec4f av_mat4x4f_get_column(const av_mat4x4f *m, unsigned int i);
AV_API av_vec3i av_mat3x3i_get_column(const av_mat3x3i *m, unsigned int i);
AV_API av_vec4i av_mat4x4i_get_column(const av_mat4x4i *m, unsigned int i);

/* Vector operations */
AV_API av_vec3f av_vec3f_add(const av_vec3f *lhs, const av_vec3f *rhs);
AV_API av_vec4f av_vec4f_add(const av_vec4f *lhs, const av_vec4f *rhs);
AV_API av_vec3f av_vec3f_sub(const av_vec3f *lhs, const av_vec3f *rhs);
AV_API av_vec4f av_vec4f_sub(const av_vec4f *lhs, const av_vec4f *rhs);
AV_API av_vec3f av_vec3f_mul(const av_vec3f *lhs, av_float s);
AV_API av_vec4f av_vec4f_mul(const av_vec4f *lhs, av_float s);
AV_API av_vec3f av_vec3f_div(const av_vec3f *lhs, av_float s);
AV_API av_vec4f av_vec4f_div(const av_vec4f *lhs, av_float s);
AV_API av_vec3i av_vec3i_add(const av_vec3i *lhs, const av_vec3i *rhs);
AV_API av_vec4i av_vec4i_add(const av_vec4i *lhs, const av_vec4i *rhs);
AV_API av_vec3i av_vec3i_sub(const av_vec3i *lhs, const av_vec3i *rhs);
AV_API av_vec4i av_vec4i_sub(const av_vec4i *lhs, const av_vec4i *rhs);
AV_API av_vec3i av_vec3i_mul(const av_vec3i *lhs, av_int s);
AV_API av_vec4i av_vec4i_mul(const av_vec4i *lhs, av_int s);
AV_API av_vec3i av_vec3i_div(const av_vec3i *lhs, av_int s);
AV_API av_vec4i av_vec4i_div(const av_vec4i *lhs, av_int s);

/* Normalize, Dot (Scalar), Length, Inner Product, Outer Product, Cross Product */
AV_API av_vec3f av_vec3f_normalize(const av_vec3f *v);
AV_API av_vec4f av_vec4f_normalize(const av_vec4f *v);
AV_API av_float av_vec3f_dot(const av_vec3f *lhs, const av_vec3f *rhs);
AV_API av_float av_vec4f_dot(const av_vec4f *lhs, const av_vec4f *rhs);
AV_API av_int av_vec3i_dot(const av_vec3i *lhs, const av_vec3i *rhs);
AV_API av_int av_vec4i_dot(const av_vec4i *lhs, const av_vec4i *rhs);
AV_API av_float av_vec3f_length(const av_vec3f *v);
AV_API av_float av_vec4f_length(const av_vec4f *v);
AV_API av_float av_vec3f_inner(const av_vec3f *v);
AV_API av_float av_vec4f_inner(const av_vec4f *v);
AV_API av_int av_vec3i_inner(const av_vec3i *v);
AV_API av_int av_vec4i_inner(const av_vec4i *v);
AV_API av_mat3x3f av_vec3f_outer(const av_vec3f *lhs, const av_vec3f *rhs);
AV_API av_mat4x4f av_vec4f_outer(const av_vec4f *lhs, const av_vec4f *rhs);
AV_API av_mat3x3i av_vec3i_outer(const av_vec3i *lhs, const av_vec3i *rhs);
AV_API av_mat4x4i av_vec4i_outer(const av_vec4i *lhs, const av_vec4i *rhs);
AV_API av_vec3f av_vec3f_cross(const av_vec3f *lhs, const av_vec3f *rhs);
AV_API av_vec3i av_vec3i_cross(const av_vec3i *lhs, const av_vec3i *rhs);

/* C++ operator overloadings for vector operations */
#ifdef __cplusplus
AV_CPP_API av_vec3f
operator+(const av_vec3f &lhs, const av_vec3f &rhs)
{
    return av_vec3f_add(&lhs, &rhs); 
}
AV_CPP_API av_vec4f
operator+(const av_vec4f &lhs, const av_vec4f &rhs)
{
    return av_vec4f_add(&lhs, &rhs);
}
AV_CPP_API av_vec3f
operator-(const av_vec3f &lhs, const av_vec3f &rhs)
{
    return av_vec3f_sub(&lhs, &rhs);
}
AV_CPP_API av_vec4f
operator-(const av_vec4f &lhs, const av_vec4f &rhs)
{
    return av_vec4f_sub(&lhs, &rhs);
}
AV_CPP_API av_vec3f
operator*(const av_vec3f &lhs, av_float s)
{
    return av_vec3f_mul(&lhs, s);
}
AV_CPP_API av_vec4f
operator*(const av_vec4f &lhs, av_float s)
{
    return av_vec4f_mul(&lhs, s);
}
AV_CPP_API av_vec3f
operator/(const av_vec3f &lhs, av_float s)
{
    return av_vec3f_div(&lhs, s);
}
AV_CPP_API av_vec4f
operator/(const av_vec4f &lhs, av_float s)
{
    return av_vec4f_div(&lhs, s);
}
AV_CPP_API av_vec3i
operator+(const av_vec3i &lhs, const av_vec3i &rhs)
{
    return av_vec3i_add(&lhs, &rhs); 
}
AV_CPP_API av_vec4i
operator+(const av_vec4i &lhs, const av_vec4i &rhs)
{
    return av_vec4i_add(&lhs, &rhs);
}
AV_CPP_API av_vec3i
operator-(const av_vec3i &lhs, const av_vec3i &rhs)
{
    return av_vec3i_sub(&lhs, &rhs);
}
AV_CPP_API av_vec4i
operator-(const av_vec4i &lhs, const av_vec4i &rhs)
{
    return av_vec4i_sub(&lhs, &rhs);
}
AV_CPP_API av_vec3i
operator*(const av_vec3i &lhs, av_int s)
{
    return av_vec3i_mul(&lhs, s);
}
AV_CPP_API av_vec4i
operator*(const av_vec4i &lhs, av_int s)
{
    return av_vec4i_mul(&lhs, s);
}
AV_CPP_API av_vec3i
operator/(const av_vec3i &lhs, av_int s)
{
    return av_vec3i_div(&lhs, s);
}
AV_CPP_API av_vec4i
operator/(const av_vec4i &lhs, av_int s)
{
    return av_vec4i_div(&lhs, s);
}
#endif

/* Matrix addition, subtraction, multiplication */
AV_API av_mat3x3f av_mat3x3f_add(const av_mat3x3f *lhs, const av_mat3x3f *rhs);
AV_API av_mat4x4f av_mat4x4f_add(const av_mat4x4f *lhs, const av_mat4x4f *rhs);
AV_API av_mat3x3i av_mat3x3i_add(const av_mat3x3i *lhs, const av_mat3x3i *rhs);
AV_API av_mat4x4i av_mat4x4i_add(const av_mat4x4i *lhs, const av_mat4x4i *rhs);
AV_API av_mat3x3f av_mat3x3f_sub(const av_mat3x3f *lhs, const av_mat3x3f *rhs);
AV_API av_mat4x4f av_mat4x4f_sub(const av_mat4x4f *lhs, const av_mat4x4f *rhs);
AV_API av_mat3x3i av_mat3x3i_sub(const av_mat3x3i *lhs, const av_mat3x3i *rhs);
AV_API av_mat4x4i av_mat4x4i_sub(const av_mat4x4i *lhs, const av_mat4x4i *rhs);
AV_API av_mat3x3f av_mat3x3f_mul(const av_mat3x3f *lhs, const av_mat3x3f *rhs);
AV_API av_mat4x4f av_mat4x4f_mul(const av_mat4x4f *lhs, const av_mat4x4f *rhs);
AV_API av_mat3x3i av_mat3x3i_mul(const av_mat3x3i *lhs, const av_mat3x3i *rhs);
AV_API av_mat4x4i av_mat4x4i_mul(const av_mat4x4i *lhs, const av_mat4x4i *rhs);
AV_API av_vec3f av_mat3x3f_vec3f_mul(const av_mat3x3f *lhs, const av_vec3f *rhs);
AV_API av_vec4f av_mat4x4f_vec4f_mul(const av_mat4x4f *lhs, const av_vec4f *rhs);
AV_API av_vec3i av_mat3x3i_vec3i_mul(const av_mat3x3i *lhs, const av_vec3i *rhs);
AV_API av_vec4i av_mat4x4i_vec4i_mul(const av_mat4x4i *lhs, const av_vec4i *rhs);
AV_API av_mat3x3f av_mat3x3f_f_mul(av_float s, const av_mat3x3f* rhs);
AV_API av_mat4x4f av_mat4x4f_f_mul(av_float s, const av_mat4x4f* rhs);
AV_API av_mat3x3i av_mat3x3i_i_mul(av_int s, const av_mat3x3i* rhs);
AV_API av_mat4x4i av_mat4x4i_i_mul(av_int s, const av_mat4x4i* rhs);

#ifdef __cplusplus
/* C++ operator overloadings for matrix addition, subtraction and multiplication */
AV_CPP_API av_mat3x3f
operator+(const av_mat3x3f &lhs, const av_mat3x3f &rhs)
{
    return av_mat3x3f_add(&lhs, &rhs);
}
AV_CPP_API av_mat4x4f
operator+(const av_mat4x4f &lhs, const av_mat4x4f &rhs)
{
    return av_mat4x4f_add(&lhs, &rhs);
}
AV_CPP_API av_mat3x3i
operator+(const av_mat3x3i &lhs, const av_mat3x3i &rhs)
{
    return av_mat3x3i_add(&lhs, &rhs);
}
AV_CPP_API av_mat4x4i
operator+(const av_mat4x4i &lhs, const av_mat4x4i &rhs)
{
    return av_mat4x4i_add(&lhs, &rhs);
}
AV_CPP_API av_mat3x3f
operator-(const av_mat3x3f &lhs, const av_mat3x3f &rhs)
{
    return av_mat3x3f_sub(&lhs, &rhs);
}
AV_CPP_API av_mat4x4f
operator-(const av_mat4x4f &lhs, const av_mat4x4f &rhs)
{
    return av_mat4x4f_sub(&lhs, &rhs);
}
AV_CPP_API av_mat3x3i
operator-(const av_mat3x3i &lhs, const av_mat3x3i &rhs)
{
    return av_mat3x3i_sub(&lhs, &rhs);
}
AV_CPP_API av_mat4x4i
operator-(const av_mat4x4i &lhs, const av_mat4x4i &rhs)
{
    return av_mat4x4i_sub(&lhs, &rhs);
}
AV_CPP_API av_mat3x3f
operator*(const av_mat3x3f &lhs, const av_mat3x3f &rhs)
{
    return av_mat3x3f_mul(&lhs, &rhs);
}
AV_CPP_API av_mat4x4f
operator*(const av_mat4x4f &lhs, const av_mat4x4f &rhs)
{
    return av_mat4x4f_mul(&lhs, &rhs);
}
AV_CPP_API av_mat3x3i
operator*(const av_mat3x3i &lhs, const av_mat3x3i &rhs)
{
    return av_mat3x3i_mul(&lhs, &rhs);
}
AV_CPP_API av_mat4x4i
operator*(const av_mat4x4i &lhs, const av_mat4x4i &rhs)
{
    return av_mat4x4i_mul(&lhs, &rhs);
}
AV_CPP_API av_vec3f 
operator*(const av_mat3x3f &lhs, const av_vec3f &rhs)
{
    return av_mat3x3f_vec3f_mul(&lhs, &rhs);
}
AV_CPP_API av_vec4f 
operator*(const av_mat4x4f &lhs, const av_vec4f &rhs)
{
    return av_mat4x4f_vec4f_mul(&lhs, &rhs);
}
AV_CPP_API av_vec3i 
operator*(const av_mat3x3i &lhs, const av_vec3i &rhs)
{
    return av_mat3x3i_vec3i_mul(&lhs, &rhs);
}
AV_CPP_API av_vec4i 
operator*(const av_mat4x4i &lhs, const av_vec4i &rhs)
{
    return av_mat4x4i_vec4i_mul(&lhs, &rhs);
}
#endif
/* Matrix transpose, determinant, inverse */
AV_API av_mat3x3f av_mat3x3f_transpose(const av_mat3x3f *m);
AV_API av_mat4x4f av_mat4x4f_transpose(const av_mat4x4f *m);
AV_API av_mat3x3i av_mat3x3i_transpose(const av_mat3x3i *m);
AV_API av_mat4x4i av_mat4x4i_transpose(const av_mat4x4i *m);
AV_API av_float av_mat3x3f_determinant(const av_mat3x3f *m);
AV_API av_float av_mat4x4f_determinant(const av_mat4x4f *m);
/* This function only works for 4 by 4 matrices where the last
 * row is exactly 0 0 0 1; which is quite common in game engines.
 */ 
AV_API av_float av_mat4x4f_fast_determinant(const av_mat4x4f *m);
AV_API av_int av_mat3x3i_determinant(const av_mat3x3i *m);
AV_API av_int av_mat4x4i_determinant(const av_mat4x4i *m);
/* This function only works for 4 by 4 matrices where the last
 * row is exactly 0 0 0 1; which is quite common in game engines.
 */ 
AV_API av_int av_mat4x4i_fast_determinant(const av_mat4x4i *m);
AV_API av_mat3x3f av_mat3x3f_inverse(const av_mat3x3f *m); 
AV_API av_mat4x4f av_mat4x4f_inverse(const av_mat4x4f *m); 
/* Inverse matrices for int matrices don't make much sense, i think */

#endif /* _AVORIS_MATH_H */


#ifdef AV_IMPLEMENTATION
#undef AV_IMPLEMENTATION

/* Helper functions that can be replaced by defining the appropriate macro */
#ifndef AV_SQRTF
/* Square root.
 * Uses the C89 sqrt function, which 
 * result in floats getting promoted to double.
 * This is kinda bad, but this way we don't require C99.
 *
 * The "correct" way to do this would be to provide our own
 * sqrtf function, which may be a wrapper around the machines
 * sqrtf assembly instruction.
 */
#include <math.h>
#define AV_SQRTF sqrt 
#endif

#ifndef AV_ASSERT
/* TODO(Kevin): We can be better than this! */
#include <assert.h>
#define AV_ASSERT assert 
#endif

/* Vector creation and conversion */
AV_API av_vec2f
av_make_vec2f(av_float x, av_float y)
{
    av_vec2f v;
    v.x = x; v.y = y;
    return v;
}
AV_API av_vec3f
av_make_vec3f(av_float x, av_float y, av_float z)
{
    av_vec3f v;
    v.x = x; v.y = y; v.z = z;
    return v;
} 
AV_API av_vec4f
av_make_vec4f(av_float x, av_float y, av_float z, av_float w)
{
    av_vec4f v;
    v.x = x; v.y = y; v.z = z; v.w = w;
    return v;
}
AV_API av_vec2i
av_make_vec2i(av_int x, av_int y)
{
    av_vec2i v;
    v.x = x; v.y = y;
    return v;
}
AV_API av_vec3i
av_make_vec3i(av_int x, av_int y, av_int z)
{
    av_vec3i v;
    v.x = x; v.y = y; v.z = z;
    return v;
}
AV_API av_vec4i
av_make_vec4i(av_int x, av_int y, av_int z, av_int w)
{
    av_vec4i v;
    v.x = x; v.y = y; v.z = z; v.w = w;
    return v;
}
AV_API av_vec2f
av_vec2i2f(av_vec2i vi)
{
    av_vec2f v;
    v.x = (av_float)vi.x;
    v.y = (av_float)vi.y;
    return v;
}
AV_API av_vec3f
av_vec3i2f(av_vec3i vi)
{
    av_vec3f v;
    v.x = (av_float)vi.x;
    v.y = (av_float)vi.y;
    v.z = (av_float)vi.z;
    return v;
}
AV_API av_vec4f
av_vec4i2f(av_vec4i vi)
{
    av_vec4f v;
    v.x = (av_float)vi.x;
    v.y = (av_float)vi.y;
    v.z = (av_float)vi.z;
    v.w = (av_float)vi.w;
    return v;
}
AV_API av_vec2i
av_vec2f2i(av_vec2f vi)
{
    av_vec2i v;
    v.x = (av_int)vi.x;
    v.y = (av_int)vi.y;
    return v;
}
AV_API av_vec3i
av_vec3f2i(av_vec3f vi)
{
    av_vec3i v;
    v.x = (av_int)vi.x;
    v.y = (av_int)vi.y;
    v.z = (av_int)vi.z;
    return v;
}
AV_API av_vec4i
av_vec4f2i(av_vec4f vi)
{
    av_vec4i v;
    v.x = (av_int)vi.x;
    v.y = (av_int)vi.y;
    v.z = (av_int)vi.z;
    return v;
}
/* Matrix creation and conversion */
AV_API av_mat3x3f
av_make_mat3x3f_identity(void)
{
    av_mat3x3f m;
    m = av_make_mat3x3f_zero();
    m.v00 = (av_float)1.f;
    m.v11 = (av_float)1.f;
    m.v22 = (av_float)1.f;
    return m;
}
AV_API av_mat4x4f
av_make_mat4x4f_identity(void)
{
    av_mat4x4f m;
    m = av_make_mat4x4f_zero();
    m.v00 = (av_float)1.f;
    m.v11 = (av_float)1.f;
    m.v22 = (av_float)1.f;
    m.v33 = (av_float)1.f;
    return m;
}
AV_API av_mat3x3f
av_make_mat3x3f_zero(void)
{
    av_mat3x3f m;
    m.v00 = (av_float)0.f;
    m.v10 = (av_float)0.f;
    m.v20 = (av_float)0.f;
    m.v01 = (av_float)0.f;
    m.v11 = (av_float)0.f;
    m.v21 = (av_float)0.f;
    m.v02 = (av_float)0.f;
    m.v12 = (av_float)0.f;
    m.v22 = (av_float)0.f;
    return m;
}
AV_API av_mat4x4f
av_make_mat4x4f_zero(void)
{
    av_mat4x4f m;
    m.v00 = (av_float)0.f;
    m.v10 = (av_float)0.f;
    m.v20 = (av_float)0.f;
    m.v30 = (av_float)0.f;
    m.v01 = (av_float)0.f;
    m.v11 = (av_float)0.f;
    m.v21 = (av_float)0.f;
    m.v31 = (av_float)0.f;
    m.v02 = (av_float)0.f;
    m.v12 = (av_float)0.f;
    m.v22 = (av_float)0.f;
    m.v32 = (av_float)0.f;
    m.v03 = (av_float)0.f;
    m.v13 = (av_float)0.f;
    m.v23 = (av_float)0.f;
    m.v33 = (av_float)0.f;
    return m;
}
AV_API av_mat3x3f
av_make_mat3x3f(av_float v00, av_float v10, av_float v20,
                av_float v01, av_float v11, av_float v21,
                av_float v02, av_float v12, av_float v22)
{
    av_mat3x3f m;
    m.v00 = v00;
    m.v01 = v01;
    m.v02 = v02;
    m.v10 = v10;
    m.v11 = v11;
    m.v12 = v12;
    m.v20 = v20;
    m.v21 = v21;
    m.v22 = v22;
    return m;
}
AV_API av_mat4x4f
av_make_mat4x4f(av_float v00, av_float v10, av_float v20, av_float v30,
                av_float v01, av_float v11, av_float v21, av_float v31,
                av_float v02, av_float v12, av_float v22, av_float v32,
                av_float v03, av_float v13, av_float v23, av_float v33)
{
    av_mat4x4f m;
    m.v00 = v00;
    m.v01 = v01;
    m.v02 = v02;
    m.v03 = v03;
    m.v10 = v10;
    m.v11 = v11;
    m.v12 = v12;
    m.v13 = v13;
    m.v20 = v20;
    m.v21 = v21;
    m.v22 = v22;
    m.v23 = v23;
    m.v30 = v30;
    m.v31 = v31;
    m.v32 = v32;
    m.v33 = v33;
    return m;
}
AV_API av_mat3x3f
av_make_mat3x3fv(const av_float *v)
{
    av_mat3x3f m;
    unsigned int i;
    av_float *c;
    c = &m.v00;
    for (i = 0; i < 9; ++i)
        c[i] = v[i];
    return m;
}
AV_API av_mat4x4f
av_make_mat4x4fv(const av_float *v)
{
    av_mat4x4f m;
    unsigned int i;
    av_float *c;
    c = &m.v00;
    for (i = 0; i < 16; ++i)
        c[i] = v[i];
    return m;
}
AV_API av_mat3x3i
av_make_mat3x3i_identity(void)
{
    av_mat3x3i m;
    m = av_make_mat3x3i_zero();
    m.v00 = 1;
    m.v11 = 1;
    m.v22 = 1;
    return m;
}
AV_API av_mat4x4i
av_make_mat4x4i_identity(void)
{
    av_mat4x4i m;
    m = av_make_mat4x4i_zero();
    m.v00 = 1;
    m.v11 = 1;
    m.v22 = 1;
    m.v33 = 1;
    return m;
}
AV_API av_mat3x3i
av_make_mat3x3i_zero(void)
{
    av_mat3x3i m;
    m.v00 = 0;
    m.v10 = 0;
    m.v20 = 0;
    m.v01 = 0;
    m.v11 = 0;
    m.v21 = 0;
    m.v02 = 0;
    m.v12 = 0;
    m.v22 = 0;
    return m;
}
AV_API av_mat4x4i
av_make_mat4x4i_zero(void)
{
    av_mat4x4i m;
    m.v00 = 0;
    m.v10 = 0;
    m.v20 = 0;
    m.v30 = 0;
    m.v01 = 0;
    m.v11 = 0;
    m.v21 = 0;
    m.v31 = 0;
    m.v02 = 0;
    m.v12 = 0;
    m.v22 = 0;
    m.v32 = 0;
    m.v03 = 0;
    m.v13 = 0;
    m.v23 = 0;
    m.v33 = 0;
    return m;
}
AV_API av_mat3x3i
av_make_mat3x3i(av_int v00, av_int v10, av_int v20,
                av_int v01, av_int v11, av_int v21,
                av_int v02, av_int v12, av_int v22)
{
    av_mat3x3i m;
    m.v00 = v00;
    m.v01 = v01;
    m.v02 = v02;
    m.v10 = v10;
    m.v11 = v11;
    m.v12 = v12;
    m.v20 = v20;
    m.v21 = v21;
    m.v22 = v22;
    return m;
}
AV_API av_mat4x4i
av_make_mat4x4i(av_int v00, av_int v10, av_int v20, av_int v30,
                av_int v01, av_int v11, av_int v21, av_int v31,
                av_int v02, av_int v12, av_int v22, av_int v32,
                av_int v03, av_int v13, av_int v23, av_int v33)
{
    av_mat4x4i m;
    m.v00 = v00;
    m.v01 = v01;
    m.v02 = v02;
    m.v03 = v03;
    m.v10 = v10;
    m.v11 = v11;
    m.v12 = v12;
    m.v13 = v13;
    m.v20 = v20;
    m.v21 = v21;
    m.v22 = v22;
    m.v23 = v23;
    m.v30 = v30;
    m.v31 = v31;
    m.v32 = v32;
    m.v33 = v33;
    return m;
}
AV_API av_mat3x3i
av_make_mat3x3iv(const av_int *v)
{
    av_mat3x3i m;
    unsigned int i;
    av_int *c;
    c = &m.v00;
    for (i = 0; i < 9; ++i)
        c[i] = v[i];
    return m;
}
AV_API av_mat4x4i
av_make_mat4x4iv(const av_int *v)
{
    av_mat4x4i m;
    unsigned int i;
    av_int *c;
    c = &m.v00;
    for (i = 0; i < 16; ++i)
        c[i] = v[i];
    return m;
}
AV_API av_mat3x3f
av_mat3x3i2f(const av_mat3x3i *mi)
{
    av_mat3x3f m;
    m.v00 = (av_float)mi->v00;
    m.v10 = (av_float)mi->v10;
    m.v20 = (av_float)mi->v20;
    m.v01 = (av_float)mi->v01;
    m.v11 = (av_float)mi->v11;
    m.v21 = (av_float)mi->v21;
    m.v02 = (av_float)mi->v02;
    m.v12 = (av_float)mi->v12;
    m.v22 = (av_float)mi->v22;
    return m;
}
AV_API av_mat4x4f 
av_mat4x4i2f(const av_mat4x4i *mi)
{
    av_mat4x4f m;
    m.v00 = (av_float)mi->v00;
    m.v10 = (av_float)mi->v10;
    m.v20 = (av_float)mi->v20;
    m.v30 = (av_float)mi->v30;
    m.v01 = (av_float)mi->v01;
    m.v11 = (av_float)mi->v11;
    m.v21 = (av_float)mi->v21;
    m.v31 = (av_float)mi->v31;
    m.v02 = (av_float)mi->v02;
    m.v12 = (av_float)mi->v12;
    m.v22 = (av_float)mi->v22;
    m.v32 = (av_float)mi->v32;
    m.v03 = (av_float)mi->v03;
    m.v13 = (av_float)mi->v13;
    m.v23 = (av_float)mi->v23;
    m.v33 = (av_float)mi->v33;
    return m;
}
AV_API av_mat3x3i 
av_mat3x3f2i(const av_mat3x3f *mi)
{
    av_mat3x3i m;
    m.v00 = (av_int)mi->v00;
    m.v10 = (av_int)mi->v10;
    m.v20 = (av_int)mi->v20;
    m.v01 = (av_int)mi->v01;
    m.v11 = (av_int)mi->v11;
    m.v21 = (av_int)mi->v21;
    m.v02 = (av_int)mi->v02;
    m.v12 = (av_int)mi->v12;
    m.v22 = (av_int)mi->v22;
    return m;
}
AV_API av_mat4x4i
av_mat4x4f2i(const av_mat4x4f *mi)
{
    av_mat4x4i m;
    m.v00 = (av_int)mi->v00;
    m.v10 = (av_int)mi->v10;
    m.v20 = (av_int)mi->v20;
    m.v30 = (av_int)mi->v30;
    m.v01 = (av_int)mi->v01;
    m.v11 = (av_int)mi->v11;
    m.v21 = (av_int)mi->v21;
    m.v31 = (av_int)mi->v31;
    m.v02 = (av_int)mi->v02;
    m.v12 = (av_int)mi->v12;
    m.v22 = (av_int)mi->v22;
    m.v32 = (av_int)mi->v32;
    m.v03 = (av_int)mi->v03;
    m.v13 = (av_int)mi->v13;
    m.v23 = (av_int)mi->v23;
    m.v33 = (av_int)mi->v33;
    return m;
}
AV_API av_vec3f
av_mat3x3f_get_column(const av_mat3x3f *m, unsigned int i)
{
    const av_float *c;
    av_vec3f v;
    AV_ASSERT(i < 3); 
    c = &m->v00 + i * 3;
    v.x = c[0];
    v.y = c[1];
    v.z = c[2];
    return v;
}
AV_API av_vec4f
av_mat4x4f_get_column(const av_mat4x4f *m, unsigned int i)
{
    const av_float *c;
    av_vec4f v;
    AV_ASSERT(i < 4); 
    c = &m->v00 + i * 4;
    v.x = c[0];
    v.y = c[1];
    v.z = c[2];
    v.w = c[3];
    return v;
}
AV_API av_vec3i
av_mat3x3i_get_column(const av_mat3x3i *m, unsigned int i)
{
    const av_int *c;
    av_vec3i v;
    AV_ASSERT(i < 3); 
    c = &m->v00 + i * 3;
    v.x = c[0];
    v.y = c[1];
    v.z = c[2];
    return v;
}
AV_API av_vec4i
av_mat4x4i_get_column(const av_mat4x4i *m, unsigned int i)
{
    const av_int *c;
    av_vec4i v;
    AV_ASSERT(i < 4); 
    c = &m->v00 + i * 4;
    v.x = c[0];
    v.y = c[1];
    v.z = c[2];
    v.w = c[3];
    return v;
}
/* Vector operations */
AV_API av_vec3f
av_vec3f_add(const av_vec3f *lhs, const av_vec3f *rhs)
{
    av_vec3f v;
    v.x = lhs->x + rhs->x;
    v.y = lhs->y + rhs->y;
    v.z = lhs->z + rhs->z;
    return v;
}
AV_API av_vec4f
av_vec4f_add(const av_vec4f *lhs, const av_vec4f *rhs)
{
    av_vec4f v;
    v.x = lhs->x + rhs->x;
    v.y = lhs->y + rhs->y;
    v.z = lhs->z + rhs->z;
    v.w = lhs->w + rhs->w;
    return v;
}
AV_API av_vec3f
av_vec3f_sub(const av_vec3f *lhs, const av_vec3f *rhs)
{
    av_vec3f v;
    v.x = lhs->x - rhs->x;
    v.y = lhs->y - rhs->y;
    v.z = lhs->z - rhs->z;
    return v;
}
AV_API av_vec4f
av_vec4f_sub(const av_vec4f *lhs, const av_vec4f *rhs)
{
    av_vec4f v;
    v.x = lhs->x - rhs->x;
    v.y = lhs->y - rhs->y;
    v.z = lhs->z - rhs->z;
    v.w = lhs->w - rhs->w;
    return v;
}
AV_API av_vec3f
av_vec3f_mul(const av_vec3f *lhs, av_float s)
{
    av_vec3f v;
    v.x = lhs->x * s;
    v.y = lhs->y * s;
    v.z = lhs->z * s;
    return v;
}
AV_API av_vec4f
av_vec4f_mul(const av_vec4f *lhs, av_float s)
{
    av_vec4f v;
    v.x = lhs->x * s;
    v.y = lhs->y * s;
    v.z = lhs->z * s;
    v.w = lhs->w * s;
    return v;
}
AV_API av_vec3f
av_vec3f_div(const av_vec3f *lhs, av_float s)
{
    av_vec3f v;
    v.x = lhs->x / s;
    v.y = lhs->y / s;
    v.z = lhs->z / s;
    return v;
}
AV_API av_vec4f
av_vec4f_div(const av_vec4f *lhs, av_float s)
{
    av_vec4f v;
    v.x = lhs->x / s;
    v.y = lhs->y / s;
    v.z = lhs->z / s;
    v.w = lhs->w / s;
    return v;
}
AV_API av_vec3i
av_vec3i_add(const av_vec3i *lhs, const av_vec3i *rhs)
{
    av_vec3i v;
    v.x = lhs->x + rhs->x;
    v.y = lhs->y + rhs->y;
    v.z = lhs->z + rhs->z;
    return v;
}
AV_API av_vec4i
av_vec4i_add(const av_vec4i *lhs, const av_vec4i *rhs)
{
    av_vec4i v;
    v.x = lhs->x + rhs->x;
    v.y = lhs->y + rhs->y;
    v.z = lhs->z + rhs->z;
    v.w = lhs->w + rhs->w;
    return v;
}
AV_API av_vec3i
av_vec3i_sub(const av_vec3i *lhs, const av_vec3i *rhs)
{
    av_vec3i v;
    v.x = lhs->x - rhs->x;
    v.y = lhs->y - rhs->y;
    v.z = lhs->z - rhs->z;
    return v;
}
AV_API av_vec4i
av_vec4i_sub(const av_vec4i *lhs, const av_vec4i *rhs)
{
    av_vec4i v;
    v.x = lhs->x - rhs->x;
    v.y = lhs->y - rhs->y;
    v.z = lhs->z - rhs->z;
    v.w = lhs->w - rhs->w;
    return v;
}
AV_API av_vec3i
av_vec3i_mul(const av_vec3i *lhs, av_int s)
{
    av_vec3i v;
    v.x = lhs->x * s;
    v.y = lhs->y * s;
    v.z = lhs->z * s;
    return v;
}
AV_API av_vec4i
av_vec4i_mul(const av_vec4i *lhs, av_int s)
{
    av_vec4i v;
    v.x = lhs->x * s;
    v.y = lhs->y * s;
    v.z = lhs->z * s;
    v.w = lhs->w * s;
    return v;
}
AV_API av_vec3i
av_vec3i_div(const av_vec3i *lhs, av_int s)
{
    av_vec3i v;
    v.x = lhs->x / s;
    v.y = lhs->y / s;
    v.z = lhs->z / s;
    return v;
}
AV_API av_vec4i
av_vec4i_div(const av_vec4i *lhs, av_int s)
{
    av_vec4i v;
    v.x = lhs->x / s;
    v.y = lhs->y / s;
    v.z = lhs->z / s;
    v.w = lhs->w / s;
    return v;
}
/* Normalize, Dot, Length, Inner Product, Outer Product, Cross Product */
AV_API av_vec3f
av_vec3f_normalize(const av_vec3f *v)
{
    av_float l = AV_SQRTF(v->x * v->x + v->y * v->y + v->z * v->z);
    av_vec3f n;
    n.x = v->x / l;
    n.y = v->y / l;
    n.z = v->z / l;
    return n;
}
AV_API av_vec4f
av_vec4f_normalize(const av_vec4f *v)
{
    av_float l = AV_SQRTF(v->x * v->x + v->y * v->y +
                           v->z * v->z + v->w * v->w);
    av_vec4f n;
    n.x = v->x / l;
    n.y = v->y / l;
    n.z = v->z / l;
    n.w = v->w / l;
    return n;
}
AV_API av_float
av_vec3f_dot(const av_vec3f *lhs, const av_vec3f *rhs)
{
    return lhs->x * rhs->x +
           lhs->y * rhs->y +
           lhs->z * rhs->z;
}
AV_API av_float
av_vec4f_dot(const av_vec4f *lhs, const av_vec4f *rhs)
{
    return lhs->x * rhs->x +
           lhs->y * rhs->y +
           lhs->z * rhs->z +
           lhs->w * rhs->w;
}
AV_API av_int
av_vec3i_dot(const av_vec3i *lhs, const av_vec3i *rhs)
{
    return lhs->x * rhs->x +
           lhs->y * rhs->y +
           lhs->z * rhs->z; 
}
AV_API av_int
av_vec4i_dot(const av_vec4i *lhs, const av_vec4i *rhs)
{
    return lhs->x * rhs->x +
           lhs->y * rhs->y +
           lhs->z * rhs->z +
           lhs->w * rhs->w;
}
AV_API av_float
av_vec3f_length(const av_vec3f *v)
{
    return AV_SQRTF(v->x * v->x + v->y * v->y + v->z * v->z);
}
AV_API av_float
av_vec4f_length(const av_vec4f *v)
{
    return AV_SQRTF(v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w);
}
AV_API av_float
av_vec3f_inner(const av_vec3f *v)
{
    return v->x * v->x + v->y * v->y + v->z * v->z;
}
AV_API av_float
av_vec4f_inner(const av_vec4f *v)
{
    return v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w;
}
AV_API av_int
av_vec3i_inner(const av_vec3i *v)
{
    return v->x * v->x + v->y * v->y + v->z * v->z;
}
AV_API av_int
av_vec4i_inner(const av_vec4i *v)
{
    return v->x * v->x + v->y * v->y + v->z * v->z + v->w * v->w;
}
AV_API av_mat3x3f
av_vec3f_outer(const av_vec3f *lhs, const av_vec3f *rhs)
{
    av_mat3x3f o;
    o.v00 = lhs->x * rhs->x;
    o.v10 = lhs->y * rhs->x;
    o.v20 = lhs->z * rhs->x;
    o.v01 = lhs->x * rhs->y;
    o.v11 = lhs->y * rhs->y;
    o.v21 = lhs->z * rhs->y;
    o.v02 = lhs->x * rhs->z;
    o.v12 = lhs->y * rhs->z;
    o.v22 = lhs->z * rhs->z;
    return o;
}
AV_API av_mat4x4f
av_vec4f_outer(const av_vec4f *lhs, const av_vec4f *rhs)
{
    av_mat4x4f o;
    o.v00 = lhs->x * rhs->x;
    o.v10 = lhs->y * rhs->x;
    o.v20 = lhs->z * rhs->x;
    o.v30 = lhs->w * rhs->x;
    o.v01 = lhs->x * rhs->y;
    o.v11 = lhs->y * rhs->y;
    o.v21 = lhs->z * rhs->y;
    o.v31 = lhs->w * rhs->y;
    o.v02 = lhs->x * rhs->z;
    o.v12 = lhs->y * rhs->z;
    o.v22 = lhs->z * rhs->z;
    o.v32 = lhs->w * rhs->z;
    o.v03 = lhs->x * rhs->w;
    o.v13 = lhs->y * rhs->w;
    o.v23 = lhs->w * rhs->z;
    o.v33 = lhs->w * rhs->w;
    return o;
}
AV_API av_mat3x3i
av_vec3i_outer(const av_vec3i *lhs, const av_vec3i *rhs)
{
    av_mat3x3i o;
    o.v00 = lhs->x * rhs->x;
    o.v10 = lhs->y * rhs->x;
    o.v20 = lhs->z * rhs->x;
    o.v01 = lhs->x * rhs->y;
    o.v11 = lhs->y * rhs->y;
    o.v21 = lhs->z * rhs->y;
    o.v02 = lhs->x * rhs->z;
    o.v12 = lhs->y * rhs->z;
    o.v22 = lhs->z * rhs->z;
    return o;
}
AV_API av_mat4x4i
av_vec4i_outer(const av_vec4i *lhs, const av_vec4i *rhs)
{
    av_mat4x4i o;;
    o.v00 = lhs->x * rhs->x;
    o.v10 = lhs->y * rhs->x;
    o.v20 = lhs->z * rhs->x;
    o.v30 = lhs->w * rhs->x;
    o.v01 = lhs->x * rhs->y;
    o.v11 = lhs->y * rhs->y;
    o.v21 = lhs->z * rhs->y;
    o.v31 = lhs->w * rhs->y;
    o.v02 = lhs->x * rhs->z;
    o.v12 = lhs->y * rhs->z;
    o.v22 = lhs->z * rhs->z;
    o.v32 = lhs->w * rhs->z;
    o.v03 = lhs->x * rhs->w;
    o.v13 = lhs->y * rhs->w;
    o.v23 = lhs->w * rhs->z;
    o.v33 = lhs->w * rhs->w;
    return o;
}
AV_API av_vec3f
av_vec3f_cross(const av_vec3f *lhs, const av_vec3f *rhs)
{
    av_vec3f c;
    c.x = lhs->y * rhs->z - lhs->z * rhs->y;
    c.y = lhs->z * rhs->x - lhs->x * rhs->z;
    c.z = lhs->x * rhs->y - lhs->y * rhs->z;
    return c;
}
AV_API av_vec3i
av_vec3i_cross(const av_vec3i *lhs, const av_vec3i *rhs)
{
    av_vec3i c;
    c.x = lhs->y * rhs->z - lhs->z * rhs->y;
    c.y = lhs->z * rhs->x - lhs->x * rhs->z;
    c.z = lhs->x * rhs->y - lhs->y * rhs->z;
    return c;
}
/* Matrix addition, subtraction, multiplication */
AV_API av_mat3x3f
av_mat3x3f_add(const av_mat3x3f *lhs, const av_mat3x3f *rhs)
{
    av_mat3x3f m;
    m.v00 = lhs->v00 + rhs->v00;
    m.v10 = lhs->v10 + rhs->v10;
    m.v20 = lhs->v20 + rhs->v20;
    m.v01 = lhs->v01 + rhs->v01;
    m.v11 = lhs->v11 + rhs->v11;
    m.v21 = lhs->v21 + rhs->v21;
    m.v02 = lhs->v02 + rhs->v02;
    m.v12 = lhs->v12 + rhs->v12;
    m.v22 = lhs->v22 + rhs->v22;
    return m;
}
AV_API av_mat4x4f
av_mat4x4f_add(const av_mat4x4f *lhs, const av_mat4x4f *rhs)
{
    av_mat4x4f m;
    m.v00 = lhs->v00 + rhs->v00;
    m.v10 = lhs->v10 + rhs->v10;
    m.v20 = lhs->v20 + rhs->v20;
    m.v30 = lhs->v30 + rhs->v30;
    m.v01 = lhs->v01 + rhs->v01;
    m.v11 = lhs->v11 + rhs->v11;
    m.v21 = lhs->v21 + rhs->v21;
    m.v31 = lhs->v31 + rhs->v31;
    m.v02 = lhs->v02 + rhs->v02;
    m.v12 = lhs->v12 + rhs->v12;
    m.v22 = lhs->v22 + rhs->v22;
    m.v32 = lhs->v32 + rhs->v32;
    m.v03 = lhs->v03 + rhs->v03;
    m.v13 = lhs->v13 + rhs->v13;
    m.v23 = lhs->v23 + rhs->v23;
    m.v33 = lhs->v33 + rhs->v33;
    return m;
}
AV_API av_mat3x3i
av_mat3x3i_add(const av_mat3x3i *lhs, const av_mat3x3i *rhs)
{
    av_mat3x3i m;
    m.v00 = lhs->v00 + rhs->v00;
    m.v10 = lhs->v10 + rhs->v10;
    m.v20 = lhs->v20 + rhs->v20;
    m.v01 = lhs->v01 + rhs->v01;
    m.v11 = lhs->v11 + rhs->v11;
    m.v21 = lhs->v21 + rhs->v21;
    m.v02 = lhs->v02 + rhs->v02;
    m.v12 = lhs->v12 + rhs->v12;
    m.v22 = lhs->v22 + rhs->v22;
    return m;
}
AV_API av_mat4x4i
av_mat4x4i_add(const av_mat4x4i *lhs, const av_mat4x4i *rhs)
{
    av_mat4x4i m;
    m.v00 = lhs->v00 + rhs->v00;
    m.v10 = lhs->v10 + rhs->v10;
    m.v20 = lhs->v20 + rhs->v20;
    m.v30 = lhs->v30 + rhs->v30;
    m.v01 = lhs->v01 + rhs->v01;
    m.v11 = lhs->v11 + rhs->v11;
    m.v21 = lhs->v21 + rhs->v21;
    m.v31 = lhs->v31 + rhs->v31;
    m.v02 = lhs->v02 + rhs->v02;
    m.v12 = lhs->v12 + rhs->v12;
    m.v22 = lhs->v22 + rhs->v22;
    m.v32 = lhs->v32 + rhs->v32;
    m.v03 = lhs->v03 + rhs->v03;
    m.v13 = lhs->v13 + rhs->v13;
    m.v23 = lhs->v23 + rhs->v23;
    m.v33 = lhs->v33 + rhs->v33;
    return m;
}
AV_API av_mat3x3f
av_mat3x3f_sub(const av_mat3x3f *lhs, const av_mat3x3f *rhs)
{
    av_mat3x3f m;
    m.v00 = lhs->v00 - rhs->v00;
    m.v10 = lhs->v10 - rhs->v10;
    m.v20 = lhs->v20 - rhs->v20;
    m.v01 = lhs->v01 - rhs->v01;
    m.v11 = lhs->v11 - rhs->v11;
    m.v21 = lhs->v21 - rhs->v21;
    m.v02 = lhs->v02 - rhs->v02;
    m.v12 = lhs->v12 - rhs->v12;
    m.v22 = lhs->v22 - rhs->v22;
    return m;
}
AV_API av_mat4x4f
av_mat4x4f_sub(const av_mat4x4f *lhs, const av_mat4x4f *rhs)
{
    av_mat4x4f m;
    m.v00 = lhs->v00 - rhs->v00;
    m.v10 = lhs->v10 - rhs->v10;
    m.v20 = lhs->v20 - rhs->v20;
    m.v30 = lhs->v30 - rhs->v30;
    m.v01 = lhs->v01 - rhs->v01;
    m.v11 = lhs->v11 - rhs->v11;
    m.v21 = lhs->v21 - rhs->v21;
    m.v31 = lhs->v31 - rhs->v31;
    m.v02 = lhs->v02 - rhs->v02;
    m.v12 = lhs->v12 - rhs->v12;
    m.v22 = lhs->v22 - rhs->v22;
    m.v32 = lhs->v32 - rhs->v32;
    m.v03 = lhs->v03 - rhs->v03;
    m.v13 = lhs->v13 - rhs->v13;
    m.v23 = lhs->v23 - rhs->v23;
    m.v33 = lhs->v33 - rhs->v33;
    return m;
}
AV_API av_mat3x3i
av_mat3x3i_sub(const av_mat3x3i *lhs, const av_mat3x3i *rhs)
{
    av_mat3x3i m;
    m.v00 = lhs->v00 - rhs->v00;
    m.v10 = lhs->v10 - rhs->v10;
    m.v20 = lhs->v20 - rhs->v20;
    m.v01 = lhs->v01 - rhs->v01;
    m.v11 = lhs->v11 - rhs->v11;
    m.v21 = lhs->v21 - rhs->v21;
    m.v02 = lhs->v02 - rhs->v02;
    m.v12 = lhs->v12 - rhs->v12;
    m.v22 = lhs->v22 - rhs->v22;
    return m;
}
AV_API av_mat4x4i
av_mat4x4i_sub(const av_mat4x4i *lhs, const av_mat4x4i *rhs)
{
    av_mat4x4i m;
    m.v00 = lhs->v00 - rhs->v00;
    m.v10 = lhs->v10 - rhs->v10;
    m.v20 = lhs->v20 - rhs->v20;
    m.v30 = lhs->v30 - rhs->v30;
    m.v01 = lhs->v01 - rhs->v01;
    m.v11 = lhs->v11 - rhs->v11;
    m.v21 = lhs->v21 - rhs->v21;
    m.v31 = lhs->v31 - rhs->v31;
    m.v02 = lhs->v02 - rhs->v02;
    m.v12 = lhs->v12 - rhs->v12;
    m.v22 = lhs->v22 - rhs->v22;
    m.v32 = lhs->v32 - rhs->v32;
    m.v03 = lhs->v03 - rhs->v03;
    m.v13 = lhs->v13 - rhs->v13;
    m.v23 = lhs->v23 - rhs->v23;
    m.v33 = lhs->v33 - rhs->v33;
    return m;
}
AV_API av_mat3x3f
av_mat3x3f_mul(const av_mat3x3f *lhs, const av_mat3x3f *rhs)
{
    av_mat3x3f m;
    av_float *mptr;
    const av_float *lptr, *rptr;
    int i, j, k;
    mptr = &m.v00;
    lptr = &lhs->v00;
    rptr = &rhs->v00;
    /* i iterates over m's colums, j iterates over m's rows */
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mptr[3 * i + j] = 0;
            for (k = 0; k < 3; ++k) {
                mptr[3 * i + j] += lptr[3 * k + j] * rptr[3 * i + k]; 
            }
        }
    }
    return m;
}
AV_API av_mat4x4f
av_mat4x4f_mul(const av_mat4x4f *lhs, const av_mat4x4f *rhs)
{
    av_mat4x4f m;
    av_float *mptr;
    const av_float *lptr, *rptr;
    int i, j, k;
    mptr = &m.v00;
    lptr = &lhs->v00;
    rptr = &rhs->v00;
    /* i iterates over m's colums, j iterates over m's rows */
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            mptr[4 * i + j] = 0;
            for (k = 0; k < 4; ++k) {
                mptr[4 * i + j] += lptr[4 * k + j] * rptr[4 * i + k]; 
            }
        }
    }
    return m;
}
AV_API av_mat3x3i
av_mat3x3i_mul(const av_mat3x3i *lhs, const av_mat3x3i *rhs)
{
    av_mat3x3i m;
    av_int *mptr;
    const av_int *lptr, *rptr;
    int i, j, k;
    mptr = &m.v00;
    lptr = &lhs->v00;
    rptr = &rhs->v00;
    /* i iterates over m's colums, j iterates over m's rows */
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mptr[3 * i + j] = 0;
            for (k = 0; k < 3; ++k) {
                mptr[3 * i + j] += lptr[3 * k + j] * rptr[3 * i + k]; 
            }
        }
    }
    return m;
}
AV_API av_mat4x4i
av_mat4x4i_mul(const av_mat4x4i *lhs, const av_mat4x4i *rhs)
{
    av_mat4x4i m;
    av_int *mptr;
    const av_int *lptr, *rptr;
    int i, j, k;
    mptr = &m.v00;
    lptr = &lhs->v00;
    rptr = &rhs->v00;
    /* i iterates over m's colums, j iterates over m's rows */
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            mptr[4 * i + j] = 0;
            for (k = 0; k < 4; ++k) {
                mptr[4 * i + j] += lptr[4 * k + j] * rptr[4 * i + k]; 
            }
        }
    }
    return m;
}
AV_API av_vec3f
av_mat3x3f_vec3f_mul(const av_mat3x3f *lhs, const av_vec3f *rhs)
{
    av_vec3f v;
    v.x = lhs->v00 * rhs->x + lhs->v01 * rhs->y + lhs->v02 * rhs->z;
    v.y = lhs->v10 * rhs->x + lhs->v11 * rhs->y + lhs->v12 * rhs->z;
    v.z = lhs->v20 * rhs->x + lhs->v21 * rhs->y + lhs->v22 * rhs->z;
    return v;
}
AV_API av_vec4f
av_mat4x4f_vec4f_mul(const av_mat4x4f *lhs, const av_vec4f *rhs)
{
    av_vec4f v;
    v.x = lhs->v00 * rhs->x + lhs->v01 * rhs->y +
          lhs->v02 * rhs->z + lhs->v03 * rhs->w;
    v.y = lhs->v10 * rhs->x + lhs->v11 * rhs->y +
          lhs->v12 * rhs->z + lhs->v13 * rhs->w;
    v.z = lhs->v20 * rhs->x + lhs->v21 * rhs->y +
          lhs->v22 * rhs->z + lhs->v23 * rhs->w;
    v.w = lhs->v30 * rhs->x + lhs->v31 * rhs->y +
          lhs->v32 * rhs->z + lhs->v33 * rhs->w;
    return v;
}
AV_API av_vec3i
av_mat3x3i_vec3i_mul(const av_mat3x3i *lhs, const av_vec3i *rhs)
{
    av_vec3i v;
    v.x = lhs->v00 * rhs->x + lhs->v01 * rhs->y + lhs->v02 * rhs->z;
    v.y = lhs->v10 * rhs->x + lhs->v11 * rhs->y + lhs->v12 * rhs->z;
    v.z = lhs->v20 * rhs->x + lhs->v21 * rhs->y + lhs->v22 * rhs->z;
    return v;
}
AV_API av_vec4i
av_mat4x4i_vec4i_mul(const av_mat4x4i *lhs, const av_vec4i *rhs)
{
    av_vec4i v;
    v.x = lhs->v00 * rhs->x + lhs->v01 * rhs->y +
          lhs->v02 * rhs->z + lhs->v03 * rhs->w;
    v.y = lhs->v10 * rhs->x + lhs->v11 * rhs->y +
          lhs->v12 * rhs->z + lhs->v13 * rhs->w;
    v.z = lhs->v20 * rhs->x + lhs->v21 * rhs->y +
          lhs->v22 * rhs->z + lhs->v23 * rhs->w;
    v.w = lhs->v30 * rhs->x + lhs->v31 * rhs->y +
          lhs->v32 * rhs->z + lhs->v33 * rhs->w;
    return v;
}
AV_API av_mat3x3f
av_mat3x3f_f_mul(av_float s, const av_mat3x3f* rhs)
{
    av_mat3x3f m;
    m.v00 = s * rhs->v00;
    m.v10 = s * rhs->v10;
    m.v20 = s * rhs->v20;
    m.v01 = s * rhs->v01;
    m.v11 = s * rhs->v11;
    m.v21 = s * rhs->v21;
    m.v02 = s * rhs->v02;
    m.v12 = s * rhs->v12;
    m.v22 = s * rhs->v22;
    return m;
}
AV_API av_mat4x4f
av_mat4x4f_f_mul(av_float s, const av_mat4x4f* rhs)
{
    av_mat4x4f m;
    m.v00 = s * rhs->v00;
    m.v10 = s * rhs->v10;
    m.v20 = s * rhs->v20;
    m.v30 = s * rhs->v30;
    m.v01 = s * rhs->v01;
    m.v11 = s * rhs->v11;
    m.v21 = s * rhs->v21;
    m.v31 = s * rhs->v31;
    m.v02 = s * rhs->v02;
    m.v12 = s * rhs->v12;
    m.v22 = s * rhs->v22;
    m.v32 = s * rhs->v32;
    m.v03 = s * rhs->v03;
    m.v13 = s * rhs->v13;
    m.v23 = s * rhs->v23;
    m.v33 = s * rhs->v33;
    return m;
}
AV_API av_mat3x3i
av_mat3x3i_i_mul(av_int s, const av_mat3x3i* rhs)
{
    av_mat3x3i m;
    m.v00 = s * rhs->v00;
    m.v10 = s * rhs->v10;
    m.v20 = s * rhs->v20;
    m.v01 = s * rhs->v01;
    m.v11 = s * rhs->v11;
    m.v21 = s * rhs->v21;
    m.v02 = s * rhs->v02;
    m.v12 = s * rhs->v12;
    m.v22 = s * rhs->v22;
    return m;
}
AV_API av_mat4x4i
av_mat4x4i_i_mul(av_int s, const av_mat4x4i* rhs)
{
    av_mat4x4i m;
    m.v00 = s * rhs->v00;
    m.v10 = s * rhs->v10;
    m.v20 = s * rhs->v20;
    m.v30 = s * rhs->v30;
    m.v01 = s * rhs->v01;
    m.v11 = s * rhs->v11;
    m.v21 = s * rhs->v21;
    m.v31 = s * rhs->v31;
    m.v02 = s * rhs->v02;
    m.v12 = s * rhs->v12;
    m.v22 = s * rhs->v22;
    m.v32 = s * rhs->v32;
    m.v03 = s * rhs->v03;
    m.v13 = s * rhs->v13;
    m.v23 = s * rhs->v23;
    m.v33 = s * rhs->v33;
    return m;
}
/* Matrix transpose, determinant, inverse */
AV_API av_mat3x3f
av_mat3x3f_transpose(const av_mat3x3f *m)
{
    av_mat3x3f t;
    t.v00 = m->v00;
    t.v10 = m->v01;
    t.v20 = m->v02;
    t.v01 = m->v10;
    t.v11 = m->v11;
    t.v21 = m->v12;
    t.v02 = m->v20;
    t.v12 = m->v21;
    t.v22 = m->v22;
    return t;
}
AV_API av_mat4x4f
av_mat4x4f_transpose(const av_mat4x4f *m)
{
    av_mat4x4f t;
    t.v00 = m->v00;
    t.v10 = m->v01;
    t.v20 = m->v02;
    t.v30 = m->v03;
    t.v01 = m->v10;
    t.v11 = m->v11;
    t.v21 = m->v12;
    t.v31 = m->v13;
    t.v02 = m->v20;
    t.v12 = m->v21;
    t.v22 = m->v22;
    t.v32 = m->v23;
    t.v03 = m->v30;
    t.v13 = m->v31;
    t.v23 = m->v32;
    t.v33 = m->v33;
    return t;
}
AV_API av_mat3x3i
av_mat3x3i_transpose(const av_mat3x3i *m)
{
    av_mat3x3i t;
    t.v00 = m->v00;
    t.v10 = m->v01;
    t.v20 = m->v02;
    t.v01 = m->v10;
    t.v11 = m->v11;
    t.v21 = m->v12;
    t.v02 = m->v20;
    t.v12 = m->v21;
    t.v22 = m->v22;
    return t;
}
AV_API av_mat4x4i
av_mat4x4i_transpose(const av_mat4x4i *m)
{
    av_mat4x4i t;
    t.v00 = m->v00;
    t.v10 = m->v01;
    t.v20 = m->v02;
    t.v30 = m->v03;
    t.v01 = m->v10;
    t.v11 = m->v11;
    t.v21 = m->v12;
    t.v31 = m->v13;
    t.v02 = m->v20;
    t.v12 = m->v21;
    t.v22 = m->v22;
    t.v32 = m->v23;
    t.v03 = m->v30;
    t.v13 = m->v31;
    t.v23 = m->v32;
    t.v33 = m->v33;
    return t;
}
AV_API av_float
av_mat3x3f_determinant(const av_mat3x3f *m)
{
    return m->v00 * m->v11 * m->v22 +
           m->v01 * m->v12 * m->v20 +
           m->v02 * m->v10 * m->v21 -
           m->v20 * m->v11 * m->v02 -
           m->v21 * m->v12 * m->v00 -
           m->v22 * m->v10 * m->v01;
}
AV_API av_float
av_mat4x4f_determinant(const av_mat4x4f *m)
{
    av_vec3f a, b, c, d, s, t, u, v, tmp1, tmp2;
    av_float x, y, z, w;
    x = m->v30;
    y = m->v31;
    z = m->v32;
    w = m->v33;
    a = *(const av_vec3f *)&m->v00;
    b = *(const av_vec3f *)&m->v01;
    c = *(const av_vec3f *)&m->v02;
    d = *(const av_vec3f *)&m->v03;
    s = av_vec3f_cross(&a, &b);
    t = av_vec3f_cross(&c, &d);
    tmp1 = av_vec3f_mul(&a, y);
    tmp2 = av_vec3f_mul(&b, x);
    u = av_vec3f_sub(&tmp1, &tmp2);
    tmp1 = av_vec3f_mul(&c, w);
    tmp2 = av_vec3f_mul(&d, z);
    v = av_vec3f_sub(&tmp1, &tmp2);
    return av_vec3f_dot(&s, &v) + av_vec3f_dot(&t, &u);
}
AV_API av_float
av_mat4x4f_fast_determinant(const av_mat4x4f *m)
{
    av_vec3f a, b, c, s;
    a = *(const av_vec3f *)&m->v00;
    b = *(const av_vec3f *)&m->v01;
    c = *(const av_vec3f *)&m->v02;
    s = av_vec3f_cross(&a, &b);
    return av_vec3f_dot(&s, &c);
}
AV_API av_int
av_mat3x3i_determinant(const av_mat3x3i *m)
{
    return m->v00 * m->v11 * m->v22 +
           m->v01 * m->v12 * m->v20 +
           m->v02 * m->v10 * m->v21 -
           m->v20 * m->v11 * m->v02 -
           m->v21 * m->v12 * m->v00 -
           m->v22 * m->v10 * m->v01;
}
AV_API av_int
av_mat4x4i_determinant(const av_mat4x4i *m)
{
    av_vec3i a, b, c, d, s, t, u, v, tmp1, tmp2;
    av_int x, y, z, w;
    x = m->v30;
    y = m->v31;
    z = m->v32;
    w = m->v33;
    a = *(const av_vec3i *)&m->v00;
    b = *(const av_vec3i *)&m->v01;
    c = *(const av_vec3i *)&m->v02;
    d = *(const av_vec3i *)&m->v03;
    s = av_vec3i_cross(&a, &b);
    t = av_vec3i_cross(&c, &d);
    tmp1 = av_vec3i_mul(&a, y);
    tmp2 = av_vec3i_mul(&b, x);
    u = av_vec3i_sub(&tmp1, &tmp2);
    tmp1 = av_vec3i_mul(&c, w);
    tmp2 = av_vec3i_mul(&d, z);
    v = av_vec3i_sub(&tmp1, &tmp2);
    return av_vec3i_dot(&s, &v) + av_vec3i_dot(&t, &u);
}
AV_API av_int
av_mat4x4i_fast_determinant(const av_mat4x4i *m)
{
    av_vec3i a, b, c, s;
    a = *(const av_vec3i *)&m->v00;
    b = *(const av_vec3i *)&m->v01;
    c = *(const av_vec3i *)&m->v02;
    s = av_vec3i_cross(&a, &b);
    return av_vec3i_dot(&s, &c);
}
AV_API av_mat3x3f
av_mat3x3f_inverse(const av_mat3x3f *m)
{
    /* Calculate the inverse via the adjugate matrix */ 
    av_mat3x3f inv;
    av_float det, invdet;
    det = av_mat3x3f_determinant(m);
#if AV_CHECK_ZERO_DETERMINANT
    if (det > -AV_EPSILON && det < AV_EPSILON) {
        return av_make_mat3x3f_zero();
    }
#endif
    invdet = (av_float)1.f / det;
    /* TODO(kevin): Get rid of the unnecessary transpose. We can just
     * as easy change the indices here */
    inv.v00 = invdet * (m->v11 * m->v22 - m->v12 * m->v21);
    inv.v01 = invdet * ((av_float)-1.f * (m->v01 * m->v22 - m->v02 * m->v21));
    inv.v02 = invdet * (m->v01 * m->v12 - m->v02 * m->v11);
    inv.v10 = invdet * ((av_float)-1.f * (m->v10 * m->v22 - m->v12 * m->v20));
    inv.v11 = invdet * (m->v00 * m->v22 - m->v02 * m->v20);
    inv.v12 = invdet * ((av_float)-1.f * (m->v00 * m->v12 - m->v02 * m->v10));
    inv.v20 = invdet * (m->v10 * m->v21 - m->v11 * m->v20);
    inv.v21 = invdet * ((av_float)-1.f * (m->v00 * m->v21 - m->v01 * m->v20));
    inv.v22 = invdet * (m->v00 * m->v11 - m->v01 * m->v10);
    return inv;
}
/* Eliminate row and column */
static av_mat3x3f
__av_get_submatrix_by_elimination(const av_mat4x4f* m, int row, int col)
{
    av_mat3x3f result;
    av_float* rc;
    const av_float* mc;
    int i, j, k, l;
    rc = &result.v00;
    mc = &m->v00;
    for (i = 0, k = 0; i < 4; ++i) {
        /* Because we use column major ordering, i & k iterate over columns and
         * j & l over rows. */
        if (i == col)
            continue;
        for (j = 0, l = 0; j < 4; ++j) {
            if (j == row)
                continue;
            rc[k * 3 + l] = mc[i * 4 + j];
            ++l;
        }
        ++k;
    }
    return result;
}
AV_API av_mat4x4f
av_mat4x4f_inverse(const av_mat4x4f *m)
{
    /* Calculate the inverse via the adjugate matrix */
    av_mat4x4f inv;
    av_mat3x3f tmp;
    av_float det, invdet, *invc;
    invc = &inv.v00;
    det = av_mat4x4f_determinant(m);
#if AV_CHECK_ZERO_DETERMINANT
    if (det > -AV_EPSILON && det < AV_EPSILON) {
        return av_make_mat4x4f_zero();
    }
#endif
    invdet = 1.f / det;
#define FILL_CELL(S, I, J) \
    tmp = __av_get_submatrix_by_elimination(m, (I), (J)); \
    invc[(J) * 4 + (I)] = (S) * invdet * av_mat3x3f_determinant(&tmp)
    FILL_CELL( 1.f, 0, 0);
    FILL_CELL(-1.f, 0, 1);
    FILL_CELL( 1.f, 0, 2);
    FILL_CELL(-1.f, 0, 3);
    FILL_CELL(-1.f, 1, 0);
    FILL_CELL( 1.f, 1, 1);
    FILL_CELL(-1.f, 1, 2);
    FILL_CELL( 1.f, 1, 3);
    FILL_CELL( 1.f, 2, 0);
    FILL_CELL(-1.f, 2, 1);
    FILL_CELL( 1.f, 2, 2);
    FILL_CELL(-1.f, 2, 3);
    FILL_CELL(-1.f, 3, 0);
    FILL_CELL( 1.f, 3, 1);
    FILL_CELL(-1.f, 3, 2);
    FILL_CELL( 1.f, 3, 3);
#undef FILL_CELL
    return inv;
} 
#endif /* IMPLEMENTATION */
