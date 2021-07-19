#pragma once

#ifndef _SSEUTIL_H
#define _SSEUTIL_H

#include "device_launch_parameters.h"
#include "device_functions.h"

#ifndef GRAVITY
#define GRAVITY 9.83219f
#endif  // !GRAVITY

#ifndef grid2Dwrite
#define grid2Dwrite(arr, x, y, pitch, value) (arr)[(y) * (pitch) + (x)] = (value)
#endif  // grid2Dwrite
#ifndef grid2Dread
#define grid2Dread(arr, x, y, pitch) (arr)[(y) * (pitch) + (x)]
#endif  // grid2Dread

namespace PhysIKA {
namespace SSEUtil {
inline void make_dimension2D(int width, int height, int bx, int by, dim3& gridDim, dim3& blockDim)
{
    gridDim.x = (width + bx - 1) / bx;
    gridDim.y = (height + by - 1) / by;
    gridDim.z = 1;

    blockDim.x = bx;
    blockDim.y = by;
    blockDim.z = 1;
}

__device__ inline float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int  old          = *address_as_i, assumed;
    do
    {
        assumed = old;
        old     = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__host__ __device__ inline bool inRange(int i, int j, int nx, int ny)
{
    return i >= 0 && i < nx && j >= 0 && j < ny;
}

__host__ __device__ inline int gridSafe2Idx(int i, int j, int nx, int ny)
{
    i = i < 0 ? 0 : i;
    i = i >= nx ? nx - 1 : i;
    j = j < 0 ? 0 : j;
    j = j >= ny ? ny - 1 : j;
    return i + j * nx;
}

__host__ __device__ inline int grid2Idx(int i, int j, int nx)
{
    return i + j * nx;
}
__host__ __device__ inline void idx2Grid(int& i, int& j, int idx, int nx)
{
    i = idx % nx;
    j = idx / nx;
}

__device__ inline float d_get_u(gridpoint gp)
{
    float h  = fmaxf(gp.x, 0.0f);
    float h4 = h * h * h * h;
    float u  = sqrtf(2.0f) * h * gp.y / (sqrtf(h4 + fmaxf(h4, EPSILON)));
    return u;
}

__device__ inline float d_get_v(gridpoint gp)
{
    float h  = fmaxf(gp.x, 0.0f);
    float h4 = h * h * h * h;
    float v  = sqrtf(2.0f) * h * gp.z / (sqrtf(h4 + fmaxf(h4, EPSILON)));
    return v;
}

__host__ __device__ inline float4 d_flux_u(float4 gpl, float4 gpr)
{
    float h = fmaxf(0.5f * (gpl.x + gpr.x), 0.0f);
    float b = 0.5f * (gpl.w + gpr.w);

    float hl  = fmaxf(gpl.x, 0.0f);
    float hl4 = hl * hl * hl * hl;
    float ul  = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + fmaxf(hl4, EPSILON)));

    float hr  = fmaxf(gpr.x, 0.0f);
    float hr4 = hr * hr * hr * hr;
    float ur  = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + fmaxf(hr4, EPSILON)));

    if (hl < EPSILON && hr < EPSILON)
    {

        return make_float4(0.0f);
    }

    float a_plus;
    float a_minus;
    a_plus  = fmaxf(fmaxf(ul + sqrtf(GRAVITY * (gpl.x /*+gpl.w*/)), ur + sqrtf(GRAVITY * (gpr.x /*+gpr.w*/))), 0);
    a_minus = fminf(fminf(ul - sqrtf(GRAVITY * (gpl.x /*+gpl.w*/)), ur - sqrtf(GRAVITY * (gpr.x /*+gpr.w*/))), 0);

    float4 delta_U = gpr - gpl;
    if (gpl.x > EPSILON && gpr.x > EPSILON)
    {
        delta_U.x += delta_U.w;
    }

    delta_U.w = 0.0f;

    float4 Fl = make_float4(gpl.y, gpl.y * ul, gpl.z * ul, 0.0f);
    float4 Fr = make_float4(gpr.y, gpr.y * ur, gpr.z * ur, 0.0f);

    float4 re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus) + a_plus * a_minus / (a_plus - a_minus) * delta_U;

    if (ul == 0 && ur == 0)  //abs(ul) <EPSILON && abs(ur) <EPSILON
    {
        re.x = 0;
        re.y = 0;
        re.z = 0;
    }
    return re;
}

__host__ __device__ inline float4 d_flux_v(float4 gpl, float4 gpr)
{
    float hl  = fmaxf(gpl.x, 0.0f);
    float hl4 = hl * hl * hl * hl;
    float vl  = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + fmaxf(hl4, EPSILON)));

    float hr  = fmaxf(gpr.x, 0.0f);
    float hr4 = hr * hr * hr * hr;
    float vr  = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + fmaxf(hr4, EPSILON)));

    if (hl < EPSILON && hr < EPSILON)
    {
        return make_float4(0.0f);
    }

    float a_plus  = fmaxf(fmaxf(vl + sqrtf(GRAVITY * (gpl.x /* + gpl.w*/)), vr + sqrtf(GRAVITY * (gpr.x /* + gpr.w*/))), 0);
    float a_minus = fminf(fminf(vl - sqrtf(GRAVITY * (gpl.x /* + gpl.w*/)), vr - sqrtf(GRAVITY * (gpr.x /* + gpr.w*/))), 0);

    float b = 0.5f * (gpl.w + gpr.w);

    float4 delta_U = gpr - gpl;
    if (gpl.x > EPSILON && gpr.x > EPSILON)
    {
        delta_U.x += delta_U.w;
    }
    delta_U.w = 0.0f;

    float4 Fl = make_float4(gpl.z, gpl.y * vl, gpl.z * vl, 0.0f);
    float4 Fr = make_float4(gpr.z, gpr.y * vr, gpr.z * vr, 0.0f);

    float4 re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus) + a_plus * a_minus / (a_plus - a_minus) * delta_U;

    if (vl == 0 && vr == 0)
    {
        re.x = 0;
        re.y = 0;
        re.z = 0;
    }
    return re;
}

__host__ __device__ inline void getSphereAngleInfo(const Vector3f& relp, float& cosphi, float& sinphi, float& costheta, float& sintheta)
{
    float r   = relp.norm();
    float rxy = sqrtf(relp[0] * relp[0] + relp[2] * relp[2]);
    sinphi    = rxy < 1e-6 ? 0 : relp[2] / rxy;
    cosphi    = rxy < 1e-6 ? 1 : relp[0] / rxy;
    sintheta  = r < 1e-6 ? 0 : rxy / r;
    costheta  = r < 1e-6 ? 1 : relp[1] / r;
}

__host__ __device__ inline float SHfunc00(float sinphi, float cosphi, float sintheta, float costheta)
{
    return 0.28209479f;  // sqrt(1/4/pi);
}

__host__ __device__ inline float SHfunc11_(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(3/4/pi) *sin(phi) * sin(theta)
    return 0.48860251f * sinphi * sintheta;
}

__host__ __device__ inline float SHfunc10(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(3/4/pi) * cos(theta)
    return 0.48860251f * costheta;
}

__host__ __device__ inline float SHfunc11(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(3/4/pi) *sin(phi) * cos(theta)
    return 0.48860251f * cosphi * sintheta;
}

__host__ __device__ inline float SHfunc22_(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(15/4/pi) * sin(phi)*cos(phi) * sin(theta)*sin(theta)
    return 1.0925484f * sinphi * cosphi * sintheta * sintheta;
}

__host__ __device__ inline float SHfunc21_(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(15/4/pi) * sin(phi)* sin(theta)*cos(theta)
    return 1.0925484f * sinphi * sintheta * costheta;
}

__host__ __device__ inline float SHfunc20(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(5/16/pi) * (3*cos(theta)*cos(theta)-1)
    return 0.3153915f * (3 * costheta * costheta - 1);
}

__host__ __device__ inline float SHfunc21(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(15/4/pi) * cos(phi)* sin(theta)*cos(theta)
    return 1.0925484f * cosphi * sintheta * costheta;
}

__host__ __device__ inline float SHfunc22(float sinphi, float cosphi, float sintheta, float costheta)
{
    // sqrt(15/16/pi) * (cos(phi) *cos(phi) - sin(phi)*sin(phi))* sin(theta)*sin(theta)
    return 1.0925484f * 0.5f * (cosphi * cosphi - sinphi * sinphi) * sintheta * sintheta;
}

template <typename T>
inline void bindTexture2D(texture<T, 2, cudaReadModeElementType>& texture_grid, DeviceArrayPitch2D<T>& arr)
{
    cudaChannelFormatDesc grid_channeldesc = cudaCreateChannelDesc<T>();
    cudaBindTexture2D(0, &texture_grid, arr.GetDataPtr(), &grid_channeldesc, arr.Nx(), arr.Ny(), arr.Pitch() * sizeof(T));
}

template <typename T>
inline void bindTexture2D(texture<T, 2, cudaReadModeElementType>& texture_grid, T* arr, int nx, int ny, int pitch)
{
    cudaChannelFormatDesc grid_channeldesc = cudaCreateChannelDesc<T>();
    cudaBindTexture2D(0, &texture_grid, arr, &grid_channeldesc, nx, ny, pitch * sizeof(T));
}
}  // namespace SSEUtil
}  // namespace PhysIKA

#endif  // _SSEUTIL_H
