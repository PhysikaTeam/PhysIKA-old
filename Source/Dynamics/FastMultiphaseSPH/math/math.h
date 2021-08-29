#pragma once

#include "vec_define.h"

inline float rand_in_range(float a, float b)
{
    return rand() / ( float )RAND_MAX * (b - a) + a;
}

void inline fscanf3(FILE* fp, cfloat3& vec)
{
    fscanf(fp, "%f %f %f ", &vec.x, &vec.y, &vec.z);
}
void inline fscanfn(FILE* fp, float* val, int n)
{
    for (int i = 0; i < n; i++)
        fscanf(fp, "%f ", &val[i]);
}

void inline fprintf3(FILE* fp, cfloat3& vec)
{
    fprintf(fp, "%f %f %f ", vec.x, vec.y, vec.z);
}
void inline fprintfn(FILE* fp, float* val, int n)
{
    for (int i = 0; i < n; i++)
        fprintf(fp, "%f ", val[i]);
}