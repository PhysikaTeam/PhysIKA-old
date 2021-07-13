#define OUTPUT_TXT

// CUDA Runtime
#include <cuda_runtime.h>

#include <cuda_profiler_api.h>
#include <assert.h>

#include <string>
using namespace std;

typedef unsigned int uint;

#include "CollisionTools.cuh"
#include "Framework/Topology/Primitive3D.h"
#include "CollisionTri3f.cuh"
#include "CollisionBVH.cuh"
#include "CollisionPair.cuh"
#include "CollisionTriContact.cuh"
#include "CollisionMesh.cuh"

#include <math.h>
#include <stdarg.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

//=======================================================

cudaDeviceProp deviceProp;
extern void    initPairsGPU();

void initGPU()
{
    static int devID = 0;

    if (devID == 0)
    {
        cudaGetDevice(&devID);
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        initPairsGPU();
    }
}

//=======================================================

g_mesh    theCloth;
g_bvh     theBVH[2];
g_front   theFront[2];
g_pair    thePairs[2];  // potentially colliding pairs
g_pair    retPairs;     //results
g_pairCCD retPairsCCD;

//=======================================================

void initPairsGPU()
{
    //pairs[0].init(MAX_PAIR_NUM); // MAX_PAIR_NUM);
    thePairs[1].init(MAX_PAIR_NUM);
    retPairs.init(MAX_PAIR_NUM);

    retPairsCCD.init(MAX_PAIR_NUM / 10);
}

void pushMesh2GPU(int numFace, int numVert, void* faces, void* nodes)
{
    theCloth.init();

    theCloth.numFace = numFace;
    theCloth.numVert = numVert;

    cudaMalloc(( void** )&theCloth._df, numFace * sizeof(tri3f));
    cudaMalloc(( void** )&theCloth._dfBx, numFace * sizeof(g_box));
    cudaMalloc(( void** )&theCloth._dx, numVert * sizeof(float3));
    cudaMalloc(( void** )&theCloth._dx0, numVert * sizeof(float3));

    cudaMemcpy(theCloth._df, faces, sizeof(tri3f) * numFace, cudaMemcpyHostToDevice);
    cudaMemcpy(theCloth._dx, nodes, sizeof(float3) * numVert, cudaMemcpyHostToDevice);
    cudaMemcpy(theCloth._dx0, theCloth._dx, sizeof(float3) * numVert, cudaMemcpyDeviceToDevice);

    theCloth.computeWSdata(0, false);
}

void updateMesh2GPU(void* nodes, void* prenodes, float thickness)
{
    cudaMemcpy(theCloth._dx, nodes, sizeof(float3) * theCloth.numVert, cudaMemcpyHostToDevice);
    cudaMemcpy(theCloth._dx0, prenodes, sizeof(float3) * theCloth.numVert, cudaMemcpyHostToDevice);
    theCloth.computeWSdata(thickness, true);

    float3* dx  = new float3[theCloth.numVert];
    float3* dx0 = new float3[theCloth.numVert];

    cudaMemcpy(dx, theCloth._dx, sizeof(float3) * theCloth.numVert, cudaMemcpyDeviceToHost);
    cudaMemcpy(dx0, theCloth._dx0, sizeof(float3) * theCloth.numVert, cudaMemcpyDeviceToHost);

    vector<float3> tem;
    for (int i = 0; i < theCloth.numVert; i++)
    {
        tem.push_back(dx[i]);
        tem.push_back(dx0[i]);
    }

    int ss = 0;
    ss++;
}

//=======================================================

void pushBVHIdx(int max_level, unsigned int* level_idx, bool isCloth)
{
    theBVH[isCloth]._max_level = max_level;
    theBVH[isCloth]._level_idx = new uint[max_level];
    memcpy(theBVH[isCloth]._level_idx, level_idx, sizeof(uint) * max_level);
}

void pushBVH(unsigned int length, int* ids, bool isCloth)
{
    theBVH[isCloth]._num = length;
    checkCudaErrors(cudaMalloc(( void** )&theBVH[isCloth]._bvh, length * sizeof(int) * 2));
    checkCudaErrors(cudaMemcpy(theBVH[isCloth]._bvh, ids, length * sizeof(int) * 2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(( void** )&theBVH[isCloth]._bxs, length * sizeof(g_box)));
    checkCudaErrors(cudaMemset(theBVH[isCloth]._bxs, 0, length * sizeof(g_box)));
    theBVH[isCloth].hBxs = NULL;

    theBVH[isCloth]._triBxs   = isCloth ? theCloth._dfBx : NULL;
    theBVH[isCloth]._triCones = NULL;
}

void pushBVHLeaf(unsigned int length, int* idf, bool isCloth)
{
    checkCudaErrors(cudaMalloc(( void** )&theBVH[isCloth]._bvh_leaf, length * sizeof(int)));
    checkCudaErrors(cudaMemcpy(theBVH[isCloth]._bvh_leaf, idf, length * sizeof(int), cudaMemcpyHostToDevice));
}

//======================================================

void refitBVH_Serial(bool isCloth, int length)
{
    refit_serial_kernel<<<1, 1, 0>>>(theBVH[isCloth]._bvh, theBVH[isCloth]._bxs, theBVH[isCloth]._triBxs, theBVH[isCloth]._cones, theBVH[isCloth]._triCones, length == 0 ? theBVH[isCloth]._num : length);

    getLastCudaError("refit_serial_kernel");
    cudaThreadSynchronize();
}

void refitBVH_Parallel(bool isCloth, int st, int length)
{
    BLK_PAR(length);

    refit_kernel<<<B, T>>>(theBVH[isCloth]._bvh, theBVH[isCloth]._bxs, theBVH[isCloth]._triBxs, theBVH[isCloth]._cones, theBVH[isCloth]._triCones, st, length);

    getLastCudaError("refit_kernel");
    cudaThreadSynchronize();
}

void refitBVH(bool isCloth)
{
    // before refit, need to get _tri_boxes !!!!
    // copying !!!
    for (int i = theBVH[isCloth]._max_level - 1; i >= 0; i--)
    {
        int st = theBVH[isCloth]._level_idx[i];
        int ed = (i != theBVH[isCloth]._max_level - 1) ? theBVH[isCloth]._level_idx[i + 1] - 1 : theBVH[isCloth]._num - 1;

        int length = ed - st + 1;
        if (i < 5)
        {
            refitBVH_Serial(isCloth, length + st);
            break;
        }
        else
        {
            refitBVH_Parallel(isCloth, st, length);
        }
    }
}

//===============================================

void pushFront(bool self, int num, unsigned int* data)
{
    g_front* f = &theFront[self];

    f->init();
    f->push(num, ( uint4* )data);
}

//===============================================
// show memory usage of GPU
void reportMemory(char* tag)
{
    //return;

#ifdef OUTPUT_TXT
    size_t      free_byte;
    size_t      total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status)
    {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    float free_db  = ( float )free_byte;
    float total_db = ( float )total_byte;
    float used_db  = total_db - free_db;
    printf("%s: GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
           tag,
           used_db / 1024.0 / 1024.0,
           free_db / 1024.0 / 1024.0,
           total_db / 1024.0 / 1024.0);
#endif
}

//===============================================

#define STACK_SIZE 50
#define EMPTY (nIdx == 0)

#define PUSH_PAIR(nd1, nd2)   \
    {                         \
        nStack[nIdx].x = nd1; \
        nStack[nIdx].y = nd2; \
        nIdx++;               \
    }

#define POP_PAIR(nd1, nd2)    \
    {                         \
        nIdx--;               \
        nd1 = nStack[nIdx].x; \
        nd2 = nStack[nIdx].y; \
    }

#define NEXT(n1, n2) POP_PAIR(n1, n2)

inline __device__ void pushToFront(int a, int b, uint4* front, uint* idx, uint ptr)
{
    //	(*idx)++;
    if (*idx < MAX_FRONT_NUM)
    {
        uint offset   = atomicAdd(idx, 1);
        front[offset] = make_uint4(a, b, 0, ptr);
    }
}

inline __device__ void sprouting(int left, int right, int* bvhA, g_box* bxsA, int* bvhB, g_box* bxsB, uint4* front, uint* frontIdx, int2* pairs, uint* pairIdx, bool update, uint ptr, tri3f* Atris)
{
    uint2 nStack[STACK_SIZE];
    uint  nIdx = 0;

    while (1)
    {
        if (isLeaf(left, bvhA) && isLeaf(right, bvhB))
        {
            if (update)
                pushToFront(left, right, front, frontIdx, ptr);

            if (overlaps(left, right, bxsA, bxsB))
            {
                if (!covertex(getTriID(left, bvhA), getTriID(right, bvhB), Atris))
                    addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
            }
        }
        else
        {
            if (!overlaps(left, right, bxsA, bxsB))
            {
                if (update)
                    pushToFront(left, right, front, frontIdx, ptr);
            }
            else
            {
                if (isLeaf(left, bvhA))
                {
                    PUSH_PAIR(left, getLeftChild(right, bvhB));
                    PUSH_PAIR(left, getRightChild(right, bvhB));
                }
                else
                {
                    PUSH_PAIR(getLeftChild(left, bvhA), right);
                    PUSH_PAIR(getRightChild(left, bvhA), right);
                }
            }
        }

        if (EMPTY)
            return;

        NEXT(left, right);
    }
}

__device__ void doPropogate(
    uint4* front,
    uint*  frontIdx,
    int    num,
    int*   bvhA,
    g_box* bxsA,
    int    bvhAnum,
    int*   bvhB,
    g_box* bxsB,
    int    bvhBnum,
    int2*  pairs,
    uint*  pairIdx,
    bool   update,
    tri3f* Atris,
    int    idx,
    bool*  flags)
{
    uint4 node = front[idx];
    if (node.z != 0)
    {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
        atomicAdd(frontIdx + 1, 1);
#endif
        return;
    }

#ifdef USE_NC
    if (flags != NULL && flags[node.w] == 0)
    {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
        atomicAdd(frontIdx + 2, 1);
#endif
        return;
    }
#endif

    uint left  = node.x;
    uint right = node.y;

    if (isLeaf(left, bvhA) && isLeaf(right, bvhB))
    {
        bool tem = overlaps(left, right, bxsA, bxsB);
        if (bxsA[left]._min.z > bxsB[right]._max.z)
            tem = false;
        if (bxsA[left]._min.z == bxsB[right]._max.z)
            tem = false;
        if (overlaps(left, right, bxsA, bxsB))
            if (bvhA != bvhB)
                addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
            else
            {  // for self ccd, we need to remove adjacent triangles, they will be processed seperatedly with orphan set
                if (!covertex(getTriID(left, bvhA), getTriID(right, bvhB), Atris))
                    addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
            }
        return;
    }

    if (!overlaps(left, right, bxsA, bxsB))
        return;

    if (update)
        front[idx].z = 1;

    int ptr = node.w;
    if (isLeaf(left, bvhA))
    {
        sprouting(left, getLeftChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
        sprouting(left, getRightChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
    }
    else
    {
        sprouting(getLeftChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
        sprouting(getRightChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
    }
}

__global__ void kernelPropogate(uint4* front, uint* frontIdx, int num, int* bvhA, g_box* bxsA, int bvhAnum, int* bvhB, g_box* bxsB, int bvhBnum, uint* tt, int2* pairs, uint* pairIdx, bool update, tri3f* Atris, int stride, bool* flags)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < stride; i++)
    {
        int j = idx * stride + i;
        if (j >= num)
            return;
        atomicAdd(tt, 1);
        doPropogate(front, frontIdx, num, bvhA, bxsA, bvhAnum, bvhB, bxsB, bvhBnum, pairs, pairIdx, update, Atris, j, flags);
    }
}

int g_front::propogate(bool& update, bool self, bool ccd)
{
    uint dummy[1];
    cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
    printf("Before propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
    uint dummy2[5] = { 0, 0, 0, 0, 0 };
    cutilSafeCall(cudaMemcpy(_dIdx + 1, dummy2, 5 * sizeof(int), cudaMemcpyHostToDevice));
#endif

    if (dummy[0] != 0)
    {
        g_bvh* pb1   = &theBVH[1];
        g_bvh* pb2   = (self) ? &theBVH[1] : &theBVH[0];
        tri3f* faces = (self ? theCloth._df : NULL);

        int stride = 4;
#ifdef FIX_BT_NUM
        BLK_PAR2(dummy[0], stride);
#else
        BLK_PAR3(dummy[0], stride, getBlkSize(( void* )kernelPropogate));
#endif
        uint* tt;
        cutilSafeCall(cudaMalloc(( void** )&tt, 1 * sizeof(uint)));
        uint ttt = 0;
        cutilSafeCall(cudaMemcpy(tt, &ttt, 1 * sizeof(uint), cudaMemcpyHostToDevice));

        kernelPropogate<<<B, T>>>(_dFront, _dIdx, dummy[0], pb1->_bvh, pb1->_bxs, pb1->_num, pb2->_bvh, pb2->_bxs, pb2->_num, tt, thePairs[self]._dPairs, thePairs[self]._dIdx, update, faces, stride, self ? theBVH[1]._ctFlags : NULL);
        //thePairs[self]._dPairs, thePairs[self]._dIdx, update, faces, stride, (self && !ccd) ? theBVH[1]._ctFlags : NULL);
        //reportMemory("propogate");

        //cutilSafeCall(cudaMemcpy(&ttt, tt, 1 * sizeof(uint), cudaMemcpyDeviceToHost));

        cudaThreadSynchronize();
        getLastCudaError("kernelPropogate");
    }

    cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
    printf("After propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
    cutilSafeCall(cudaMemcpy(dummy2, _dIdx + 1, 5 * sizeof(int), cudaMemcpyDeviceToHost));
    //printf("Invalid = %d, NC culled = %d\n", dummy2[0], dummy2[1]);
#endif

    if (update && dummy[0] > SAFE_FRONT_NUM)
    {
        printf("Too long front, stop updating ...\n");
        update = false;
    }

    if (dummy[0] > MAX_FRONT_NUM)
    {
        printf("Too long front, exiting ...\n");
        exit(0);
    }
    return dummy[0];
}

//===============================================

__global__ void
kernel_face_ws(tri3f* face, float3* x, float3* ox, g_box* bxs, bool ccd, float thickness, int num)
{
    LEN_CHK(num);

    int id0 = face[idx].id0();
    int id1 = face[idx].id1();
    int id2 = face[idx].id2();

    float3 ox0 = ox[id0];
    float3 ox1 = ox[id1];
    float3 ox2 = ox[id2];
    float3 x0  = x[id0];
    float3 x1  = x[id1];
    float3 x2  = x[id2];

    bxs[idx].set(ox0, ox1);
    bxs[idx].add(ox2);

    if (ccd)
    {
        bxs[idx].add(x0);
        bxs[idx].add(x1);
        bxs[idx].add(x2);
    }
    //else
    bxs[idx].enlarge(thickness);
}

void g_mesh::computeWSdata(float thickness, bool ccd)
{
    if (numFace == 0)
        return;

    {
        int num = numFace;
        BLK_PAR(num);
        kernel_face_ws<<<B, T>>>(
            _df, _dx, _dx0, _dfBx, ccd, thickness, num);
        getLastCudaError("kernel_face_ws");
    }
}

//===============================================

__device__ void tri_DCD(
    uint    t1,
    uint    t2,
    uint    t3,
    uint    tt1,
    uint    tt2,
    uint    tt3,
    float3* cx,
    float3* cx0,
    int2*   pairRets,
    int4*   dv,
    int*    VF_EE,
    float*  dist,
    int*    CCDres,
    int*    t,
    uint*   pairIdx,
    int     fid1,
    int     fid2,
    float*  thickness)
{
    float3 p[3];
    p[0] = cx[t1];
    p[1] = cx[t2];
    p[2] = cx[t3];

    float3 pp[3];
    pp[0] = cx0[t1];
    pp[1] = cx0[t2];
    pp[2] = cx0[t3];

    float3 q[3];
    q[0] = cx[tt1];
    q[1] = cx[tt2];
    q[2] = cx[tt3];

    float3 qq[3];
    qq[0] = cx0[tt1];
    qq[1] = cx0[tt2];
    qq[2] = cx0[tt3];

    if (tri_contact(p[0], p[1], p[2], q[0], q[1], q[2]))

        if (fid1 > fid2)
            addPairDCD(fid2, fid1, 0, 0, 0, 0, 0, 0, NULL, pairRets, NULL, NULL, NULL, CCDres, pairIdx);
        else
            addPairDCD(fid1, fid2, 0, 0, 0, 0, 0, 0, NULL, pairRets, NULL, NULL, NULL, CCDres, pairIdx);
}

__global__ void kernelGetCollisions(
    int2*   pairs,
    int     num,
    float3* cx,
    float3* cx0,
    tri3f*  ctris,
    int2*   pairRets,
    int4*   dv,
    int*    VF_EE,
    float*  dist,
    int*    CCDres,
    uint*   pairIdx,
    int*    t,
    float*  thickness,
    int     stride)
{

    int idxx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < stride; i++)
    {

        int j = idxx * stride + i;
        if (j >= num)
            return;

        int idx = j;

        int2 pair = pairs[idx];
        int  fid1 = pair.x;
        int  fid2 = pair.y;

        int ss = 0;

        tri3f t1 = ctris[fid1];
        tri3f t2 = ctris[fid2];

        uint tt1[3];
        uint tt2[3];
        tt1[0] = t1.id0();
        tt1[1] = t1.id1();
        tt1[2] = t1.id2();

        tt2[0] = t2.id0();
        tt2[1] = t2.id1();
        tt2[2] = t2.id2();

#ifdef FOR_DEBUG
        bool find = false;
        if (fid1 == 369 && fid2 == 3564)
            find = true;
        if (fid2 == 369 && fid1 == 3564)
            find = true;
#endif
        float3 p[3];
        p[0] = cx[t1.id0()];
        p[1] = cx[t1.id1()];
        p[2] = cx[t1.id2()];

        float3 pp[3];
        pp[0] = cx0[t1.id0()];
        pp[1] = cx0[t1.id1()];
        pp[2] = cx0[t1.id2()];

        float3 q[3];
        q[0] = cx[t2.id0()];
        q[1] = cx[t2.id1()];
        q[2] = cx[t2.id2()];

        float3 qq[3];
        qq[0] = cx0[t2.id0()];
        qq[1] = cx0[t2.id1()];
        qq[2] = cx0[t2.id2()];

        bool iscovetex = false;
        for (int st = 0; st < 3; st++)
        {
            for (int ss = 0; ss < 3; ss++)
            {
                if (tt1[st] == tt2[ss])
                {
                    iscovetex = true;
                }
            }
        }

        if (iscovetex)
            continue;

        tri_DCD(t1.id0(), t1.id1(), t1.id2(), t2.id0(), t2.id1(), t2.id2(), cx, cx0, pairRets, dv, VF_EE, dist, CCDres, t, pairIdx, fid1, fid2, thickness);
    }
}

//===============================================

int g_pair::getCollisions(bool self, g_pairCCD& ret, int* time, float* thickness)
{
    int num = length();
    printf("pair = %d\n", num);

    if (num == 0)
        return 0;

    ret.clear();

    int stride = 4;
#ifdef FIX_BT_NUM
    BLK_PAR3(num, stride, 32);
#else
    BLK_PAR3(num, stride, getBlkSize(( void* )kernelGetCollisions));
#endif

    int* tem_time;
    int  tem = 0;
    cutilSafeCall(cudaMalloc(( void** )&tem_time, 1 * sizeof(int)));
    cutilSafeCall(cudaMemcpy(tem_time, &tem, 1 * sizeof(int), cudaMemcpyHostToDevice));

    float* gthickness;
    cutilSafeCall(cudaMalloc(( void** )&gthickness, 1 * sizeof(float)));
    cutilSafeCall(cudaMemcpy(gthickness, thickness, 1 * sizeof(float), cudaMemcpyHostToDevice));

    kernelGetCollisions<<<B, T>>>(_dPairs, num, theCloth._dx, theCloth._dx0, theCloth._df, ret._dPairs, ret._dv, ret._dVF_EE, ret.dist, ret.CCD_res, ret._dIdx, tem_time, gthickness, stride);

    getLastCudaError("kernelGetCollisions");

    int len = ret.length();
    printf("collision num = %d\n", len);

    cutilSafeCall(cudaMemcpy(time, tem_time, sizeof(int) * 1, cudaMemcpyDeviceToHost));

    cudaFree(tem_time);

    return len;
}
//===============================================

int getCollisionsGPU(int* rets, int* vf_ee, int* vertex_id, float* dist, int* time, int* CCDres, float* thickness)
{
    bool update = true;
    int  len    = 0;

    thePairs[1].clear();

    refitBVH(true);
    theFront[1].propogate(update, 1, false);
    cudaThreadSynchronize();

    len = thePairs[1].getCollisions(true, retPairsCCD, time, thickness);
    cudaThreadSynchronize();

    if (len > 0)
    {
        cutilSafeCall(cudaMemcpy(rets, retPairsCCD._dPairs, sizeof(uint) * 2 * len, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(vf_ee, retPairsCCD._dVF_EE, sizeof(int) * len, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(vertex_id, retPairsCCD._dv, sizeof(int) * 4 * len, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(dist, retPairsCCD.dist, sizeof(float) * len, cudaMemcpyDeviceToHost));
        cutilSafeCall(cudaMemcpy(CCDres, retPairsCCD.CCD_res, sizeof(int) * len, cudaMemcpyDeviceToHost));
    }
    return len;
}