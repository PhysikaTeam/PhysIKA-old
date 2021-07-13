#pragma once
#include "geometry.h"

HDFUNC inline void printf3(const char* msg, cfloat3 f)
{
    printf("%s: %f %f %f\n", msg, f.x, f.y, f.z);
}

HDFUNC inline void printMat(const char* msg, cmat3& f)
{
    printf("%s: %f %f %f\n%f %f %f\n%f %f %f\n", msg, f[0][0], f[0][1], f[0][2], f[1][0], f[1][1], f[1][2], f[2][0], f[2][1], f[2][2]);
}