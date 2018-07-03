#pragma once
#include "cuda_helper_math.h"


////////////////////////////////////////////////////////////////////////////////
// template functions
// Created by Xiaowei He
////////////////////////////////////////////////////////////////////////////////

template <typename Coord, typename Real>
inline __device__ __host__ Coord Make(Real a) {
	Coord coord; return coord;
}

template <>
inline __device__ __host__ float2 Make(float a) {
	return make_float2(a);
}

template <>
inline __device__ __host__ float3 Make(float a) {
	return make_float3(a);
}

template <typename Coord>
inline __device__ void AtomicAdd(Coord& dst, Coord src) {

}

template <>
inline __device__ void AtomicAdd(float& dst, float src) {
	atomicAdd(&dst, src);
}

template <>
inline __device__ void AtomicAdd(float2& dst, float2 src) {
	atomicAdd(&dst.x, src.x);
	atomicAdd(&dst.y, src.y);
}

template <>
inline __device__ void AtomicAdd(float3& dst, float3 src) {
	atomicAdd(&dst.x, src.x);
	atomicAdd(&dst.y, src.y);
	atomicAdd(&dst.z, src.z);
}

template <>
inline __device__ void AtomicAdd(float4& dst, float4 src) {
	atomicAdd(&dst.x, src.x);
	atomicAdd(&dst.y, src.y);
	atomicAdd(&dst.z, src.z);
	atomicAdd(&dst.w, src.w);
}
