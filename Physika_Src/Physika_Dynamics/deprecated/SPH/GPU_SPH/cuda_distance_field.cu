/*
 * @file cuda_distance_field.cu
 * @Brief class CudaDistanceField
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <fstream>
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"

#include "Physika_Dynamics/SPH/GPU_SPH/cuda_distance_field.h"

namespace Physika {

__device__  float distanceToPlane(const float3 &p, const float3 &o, const float3 &n)
{
    return fabs(dot(p - o, n));
}

__device__  float distanceToSegment(float3& pos, float3& lo, float3& hi)
{
    float3 seg = hi - lo;
    float3 edge1 = pos - lo;
    float3 edge2 = pos - hi;
    if (dot(edge1, seg) < 0.0f)
    {
        return length(edge1);
    }
    if (dot(edge2, -seg) < 0.0f)
    {
        return length(edge2);
    }
    float length1 = dot(edge1, edge1);
    normalize(seg);
    float length2 = dot(edge1, seg);
    return sqrt(length1 - length2*length2);
}

__device__  float distanceToSqure(float3& pos, float3& lo, float3& hi, int axis)
{
    float3 n;
    float3 corner1, corner2, corner3, corner4;
    float3 loCorner, hiCorner, p;
    switch (axis)
    {
    case 0:
        corner1 = make_float3(lo.x, lo.y, lo.z);
        corner2 = make_float3(lo.x, hi.y, lo.z);
        corner3 = make_float3(lo.x, hi.y, hi.z);
        corner4 = make_float3(lo.x, lo.y, hi.z);
        n = make_float3(1.0f, 0.0f, 0.0f);

        loCorner = make_float3(lo.y, lo.z, 0.0f);
        hiCorner = make_float3(hi.y, hi.z, 0.0f);
        p = make_float3(pos.y, pos.z, 0.0f);
        break;
    case 1:
        corner1 = make_float3(lo.x, lo.y, lo.z);
        corner2 = make_float3(lo.x, lo.y, hi.z);
        corner3 = make_float3(hi.x, lo.y, hi.z);
        corner4 = make_float3(hi.x, lo.y, lo.z);
        n = make_float3(0.0f, 1.0f, 0.0f);

        loCorner = make_float3(lo.x, lo.z, 0.0f);
        hiCorner = make_float3(hi.x, hi.z, 0.0f);
        p = make_float3(pos.x, pos.z, 0.0f);
        break;
    case 2:
        corner1 = make_float3(lo.x, lo.y, lo.z);
        corner2 = make_float3(hi.x, lo.y, lo.z);
        corner3 = make_float3(hi.x, hi.y, lo.z);
        corner4 = make_float3(lo.x, hi.y, lo.z);
        n = make_float3(0.0f, 0.0f, 1.0f);

        loCorner = make_float3(lo.x, lo.y, 0.0f);
        hiCorner = make_float3(hi.x, hi.y, 0.0f);
        p = make_float3(pos.x, pos.y, 0.0f);
        break;
    }

    float dist1 = distanceToSegment(pos, corner1, corner2);
    float dist2 = distanceToSegment(pos, corner2, corner3);
    float dist3 = distanceToSegment(pos, corner3, corner4);
    float dist4 = distanceToSegment(pos, corner4, corner1);
    float dist5 = abs(dot(pos - corner1, n));
    if (p.x < hiCorner.x && p.x > loCorner.x && p.y < hiCorner.y && p.y > loCorner.y)
        return dist5;
    else
        return min(min(dist1, dist2), min(dist3, dist4));
}

__device__  float distanceToBox(float3& pos, float3& lo, float3& hi)
{
    float3 corner0 = make_float3(lo.x, lo.y, lo.z);
    float3 corner1 = make_float3(hi.x, lo.y, lo.z);
    float3 corner2 = make_float3(hi.x, hi.y, lo.z);
    float3 corner3 = make_float3(lo.x, hi.y, lo.z);
    float3 corner4 = make_float3(lo.x, lo.y, hi.z);
    float3 corner5 = make_float3(hi.x, lo.y, hi.z);
    float3 corner6 = make_float3(hi.x, hi.y, hi.z);
    float3 corner7 = make_float3(lo.x, hi.y, hi.z);
    float dist0 = length(pos - corner0);
    float dist1 = length(pos - corner1);
    float dist2 = length(pos - corner2);
    float dist3 = length(pos - corner3);
    float dist4 = length(pos - corner4);
    float dist5 = length(pos - corner5);
    float dist6 = length(pos - corner6);
    float dist7 = length(pos - corner7);
    if (pos.x < hi.x && pos.x > lo.x && pos.y < hi.y && pos.y > lo.y && pos.z < hi.z && pos.z > lo.z)
    {
        float distx = min(abs(pos.x - hi.x), abs(pos.x - lo.x));
        float disty = min(abs(pos.y - hi.y), abs(pos.y - lo.y));
        float distz = min(abs(pos.z - hi.z), abs(pos.z - lo.z));
        float mindist = min(distx, disty);
        mindist = min(mindist, distz);
        return mindist;
    }
    else
    {
        float distx1 = distanceToSqure(pos, corner0, corner7, 0);
        float distx2 = distanceToSqure(pos, corner1, corner6, 0);
        float disty1 = distanceToSqure(pos, corner0, corner5, 1);
        float disty2 = distanceToSqure(pos, corner3, corner6, 1);
        float distz1 = distanceToSqure(pos, corner0, corner2, 2);
        float distz2 = distanceToSqure(pos, corner4, corner6, 2);
        return -min(min(min(distx1, distx2), min(disty1, disty2)), min(distz1, distz2));
    }
}

__device__  float distanceToCylinder(float3& pos, float3& center, float radius, float height, int axis)
{
    float distR;
    float distH;
    switch (axis)
    {
    case 0:
        distH = abs(pos.x - center.x);
        distR = length(make_float3(0.0f, pos.y - center.y, pos.z - center.z));
        break;
    case 1:
        distH = abs(pos.y - center.y);
        distR = length(make_float3(pos.x - center.x, 0.0f, pos.z - center.z));
        break;
    case 2:
        distH = abs(pos.z - center.z);
        distR = length(make_float3(pos.x - center.x, pos.y - center.y, 0.0f));
        break;
    }

    float halfH = height / 2.0f;
    if (distH <= halfH && distR <= radius)
    {
        return -min(halfH - distH, radius - distR);
    }
    else if (distH > halfH && distR <= radius)
    {
        return distH - halfH;
    }
    else if (distH <= halfH && distR > radius)
    {
        return distR - radius;
    }
    else
    {
        float l1 = distR - radius;
        float l2 = distH - halfH;
        return sqrt(l1*l1 + l2*l2);
    }
}

__device__  float distanceToSphere(float3& pos, float3& center, float radius)
{
    return length(pos - center) - radius;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

static uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static uint cudaGridSize(uint totalSize, uint blockSize)
{
    return iDivUp(totalSize, blockSize);
}

static dim3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
{
    dim3 gridDims;
    gridDims.x = iDivUp(totalSize.x, blockSize.x);
    gridDims.y = iDivUp(totalSize.y, blockSize.y);
    gridDims.z = iDivUp(totalSize.z, blockSize.z);

    return gridDims;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

CudaDistanceField::CudaDistanceField()
{

}

CudaDistanceField::~CudaDistanceField()
{

}

CudaDistanceField::CudaDistanceField(std::string filename)
{
    readSDF(filename);
    bInvert = false;
}

void CudaDistanceField::setSpace(const float3 p0, const float3 p1, int nbx, int nby, int nbz)
{
    left = p0;

    h = (p1 - p0)*make_float3(1.0f / float(nbx + 1), 1.0f / float(nby + 1), 1.0f / float(nbz + 1));

    gDist.setSpace(nbx + 1, nby + 1, nbz + 1);
}



void CudaDistanceField::translate(const float3 &t) 
{
    left += t;
}


__global__ void K_Scale(CudaGrid1f distance, float s)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    int k = threadIdx.z + (blockIdx.z * blockDim.z);

    if (i >= distance.nx) return;
    if (j >= distance.ny) return;
    if (k >= distance.nz) return;

    distance(i, j, k) = s*distance(i, j, k);
}

void CudaDistanceField::scale(const float s) {
    left.x *= s;
    left.y *= s;
    left.z *= s;
    h.x *= s;
    h.y *= s;
    h.z *= s;

    dim3 blockSize = make_uint3(8, 8, 8);
    dim3 gridDims = cudaGridSize3D(make_uint3(gDist.nx, gDist.ny, gDist.nz), blockSize);

    K_Scale<<<gridDims, blockSize >>>(gDist, s);
}

/*********************************************************************************************************/
__global__ void K_Invert(CudaGrid1f distance)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    int k = threadIdx.z + (blockIdx.z * blockDim.z);

    if (i >= distance.nx) return;
    if (j >= distance.ny) return;
    if (k >= distance.nz) return;

    distance(i, j, k) = -distance(i, j, k);
}
/*********************************************************************************************************/

void CudaDistanceField::invert()
{
    dim3 blockSize = make_uint3(8, 8, 8);
    dim3 gridDims = cudaGridSize3D(make_uint3(gDist.nx, gDist.ny, gDist.nz), blockSize);

    K_Invert <<<gridDims, blockSize >>>(gDist);
}

/*********************************************************************************************************/
__global__ void K_DistanceFieldToBox(CudaGrid1f distance, float3 start, float3 h, float3 lo, float3 hi, bool inverted)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    int k = threadIdx.z + (blockIdx.z * blockDim.z);

    if (i >= distance.nx) return;
    if (j >= distance.ny) return;
    if (k >= distance.nz) return;

    int sign = inverted ? 1.0f : -1.0f;
    float3 p = start + make_float3(i, j, k)*h;

    distance(i, j, k) = sign*distanceToBox(p, lo, hi);
}
/*********************************************************************************************************/

void CudaDistanceField::distanceFieldToBox(float3& lo, float3& hi, bool inverted)
{
    bInvert = inverted;

    dim3 blockSize = make_uint3(4, 4, 4);
    dim3 gridDims = cudaGridSize3D(make_uint3(gDist.nx, gDist.ny, gDist.nz), blockSize);

    K_DistanceFieldToBox<<<gridDims, blockSize >>>(gDist, left, h, lo, hi, inverted);
}

/*********************************************************************************************************/
__global__ void K_DistanceFieldToCylinder(CudaGrid1f distance, float3 start, float3 h, float3 center, float radius, float height, int axis, bool inverted)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    int k = threadIdx.z + (blockIdx.z * blockDim.z);

    if (i >= distance.nx) return;
    if (j >= distance.ny) return;
    if (k >= distance.nz) return;

    int sign = inverted ? -1.0f : 1.0f;

    float3 p = start + make_float3(i, j, k)*h;

    distance(i, j, k) = sign*distanceToCylinder(p, center, radius, height, axis);
}
/*********************************************************************************************************/

void CudaDistanceField::distanceFieldToCylinder(float3& center, float radius, float height, int axis, bool inverted)
{
    bInvert = inverted;

    dim3 blockSize = make_uint3(8, 8, 8);
    dim3 gridDims = cudaGridSize3D(make_uint3(gDist.nx, gDist.ny, gDist.nz), blockSize);

    K_DistanceFieldToCylinder<<<gridDims, blockSize >>>(gDist, left, h, center, radius, height, axis, inverted);
}

/*********************************************************************************************************/
__global__ void K_DistanceFieldToSphere(CudaGrid1f distance, float3 start, float3 h, float3 center, float radius, bool inverted)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    int k = threadIdx.z + (blockIdx.z * blockDim.z);

    if (i >= distance.nx) return;
    if (j >= distance.ny) return;
    if (k >= distance.nz) return;

    int sign = inverted ? -1.0f : 1.0f;

    float3 p = start + make_float3(i, j, k)*h;

    float3 dir = p - center;

    distance(i, j, k) = sign*(length(dir) - radius);
}
/*********************************************************************************************************/

void CudaDistanceField::distanceFieldToSphere(float3& center, float radius, bool inverted)
{
    bInvert = inverted;

    dim3 blockSize = make_uint3(8, 8, 8);
    dim3 gridDims = cudaGridSize3D(make_uint3(gDist.nx, gDist.ny, gDist.nz), blockSize);

    K_DistanceFieldToSphere <<<gridDims, blockSize >>>(gDist, left, h, center, radius, inverted);
}

__device__ void CudaDistanceField::getDistance(const float3 &p, float &d, float3 &g)
{
    // get cell and lerp values
    float3 fp = (p - left)*make_float3(1.0f / h.x, 1.0f / h.y, 1.0f / h.z);
    const int i = (int)floor(fp.x);
    const int j = (int)floor(fp.y);
    const int k = (int)floor(fp.z);

    if (i < 0 || i >= gDist.nx - 1 || j < 0 || j >= gDist.ny - 1 || k < 0 || k >= gDist.nz - 1)
    {
        if (bInvert) d = -100000.0f;
        else d = 100000.0f;
        g = make_float3(0.0f);
        return;
    }

    float3 ip = make_float3(i, j, k);
    float3 alphav = fp - ip;
    float alpha = alphav.x;
    float beta = alphav.y;
    float gamma = alphav.z;

    float d000 = gDist(i, j, k);
    float d100 = gDist(i + 1, j, k);
    float d010 = gDist(i, j + 1, k);
    float d110 = gDist(i + 1, j + 1, k);
    float d001 = gDist(i, j, k + 1);
    float d101 = gDist(i + 1, j, k + 1);
    float d011 = gDist(i, j + 1, k + 1);
    float d111 = gDist(i + 1, j + 1, k + 1);

    float dx00 = lerp(d000, d100, alpha);
    float dx10 = lerp(d010, d110, alpha);
    float dxy0 = lerp(dx00, dx10, beta);

    float dx01 = lerp(d001, d101, alpha);
    float dx11 = lerp(d011, d111, alpha);
    float dxy1 = lerp(dx01, dx11, beta);

    float d0y0 = lerp(d000, d010, beta);
    float d0y1 = lerp(d001, d011, beta);
    float d0yz = lerp(d0y0, d0y1, gamma);

    float d1y0 = lerp(d100, d110, beta);
    float d1y1 = lerp(d101, d111, beta);
    float d1yz = lerp(d1y0, d1y1, gamma);

    float dx0z = lerp(dx00, dx01, gamma);
    float dx1z = lerp(dx10, dx11, gamma);

    g.x = d0yz - d1yz;
    g.y = dx0z - dx1z;
    g.z = dxy0 - dxy1;

    float l = length(g);
    if (l < 0.0001f) g = make_float3(0.0f);
    else g = normalize(g);

    d = (1.0f - gamma) * dxy0 + gamma * dxy1;
}

__host__ __device__ float CudaDistanceField::lerp(float a, float b, float alpha) const
{
    return (1.0f - alpha)*a + alpha *b;
}

void CudaDistanceField::readSDF(std::string filename)
{
    std::ifstream input(filename.c_str(), std::ios::in);
    int nbx, nby, nbz;
    int xx, yy, zz;
    input >> xx;
    input >> yy;
    input >> zz;
    input >> left.x;
    input >> left.y;
    input >> left.z;
    float t_h;
    input >> t_h;

    std::cout << "SDF: " << xx << ", " << yy << ", " << zz << std::endl;
    std::cout << "SDF: " << left.x << ", " << left.y << ", " << left.z << std::endl;
    std::cout << "SDF: " << left.x + t_h*xx << ", " << left.y + t_h*yy << ", " << left.z + t_h*zz << std::endl;

    nbx = xx;
    nby = yy;
    nbz = zz;
    h.x = t_h;
    h.y = t_h;
    h.z = t_h;

    int idd = 0;
    float* distances = new float[(nbx)*(nby)*(nbz)];
    for (int k = 0; k < zz; k++) {
        for (int j = 0; j < yy; j++) {
            for (int i = 0; i < xx; i++) {
                float dist;
                input >> dist;
                distances[i + nbx*(j + nby*k)] = dist;
            }
        }
    }
    input.close();

    gDist.setSpace(nbx, nby, nbz);
    cudaCheck(cudaMemcpy(gDist.data, distances, (nbx)*(nby)*(nbz) * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "read data successfully" << std::endl;
}

void CudaDistanceField::release()
{
    gDist.release();
}



}//end of namespace Physika