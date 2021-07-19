/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: device collision mesh class, should not be used directly
 * @version    : 1.0
 */

#pragma once

#include "CollisionTools.cuh"
#include "CollisionTri3f.cuh"
#include "CollisionBox.cuh"

typedef struct
{
    uint    numFace, numVert;
    float3 *_dx, *_dx0;
    tri3f*  _df;
    g_box*  _dfBx;

    // init function
    void init()
    {
        numFace = 0;
        numVert = 0;
        _dx0 = _dx = NULL;
        _df        = NULL;
        _dfBx      = NULL;
    }

    void destroy()
    {
        if (_dx == NULL)
            return;

        checkCudaErrors(cudaFree(_dx));
        checkCudaErrors(cudaFree(_dx0));
        checkCudaErrors(cudaFree(_df));
        checkCudaErrors(cudaFree(_dfBx));
    }

    void computeWSdata(float thickness, bool ccd);
} g_mesh;
