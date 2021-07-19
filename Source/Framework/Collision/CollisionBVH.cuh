/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2021-05-30
 * @description: device bvh structure
 * @version    : 1.0
 */

#pragma once

#include "CollisionBox.cuh"

typedef unsigned int uint;

/**
 * Device bvh data structure
 */
typedef struct
{
    int     _num;       //!< numbers
    int*    _bvh;       //!< pointer to bvh
    int*    _bvh_leaf;  //!< bvh leaf indices
    g_box*  _bxs;       //!< bvh boxes
    g_cone* _cones;     //!< cones

    int   _max_level;
    uint* _level_idx;

    // for contour test
    int      _ctNum;
    uint *   _ctIdx, *_ctLst;
    g_cone*  _triCones;  //!< directly borrow from g_mesh ...
    bool*    _ctFlags;
    double3 *_ctPts, *_ctVels;

    uint2* _fids;
    g_box* _triBxs;  //!< directly borrow form g_mesh ...

    g_box* hBxs;  //!< host boxes for debug;

    /**
	 * get boxes and save result in hBxs
	 */
    void getBxs()
    {
        if (hBxs == NULL)
            hBxs = new g_box[_num];

        cudaMemcpy(hBxs, _bxs, _num * sizeof(g_box), cudaMemcpyDeviceToHost);
    }

    /**
	 * write host boxes data into the file
	 *
	 * @param[in] path file path
	 */
    void printBxs(char* path)
    {
        FILE* fp = fopen(path, "wt");
        for (int i = 0; i < _num; i++)
            hBxs[i].print(fp);
        fclose(fp);
    }

    /**
	 * debug usage
	 */
    void destory()
    {
        printf("unfinished!\n");
    }
} g_bvh;

/**
 * Device cone front data structure
 */
typedef struct
{
    uint3* _dFront;  //!< node_id, parent_id, valid 0/1
    uint*  _dIdx;    //!< indexes
    uint   _iMax;    //!< max index

    /**
	 * init cone font with a max number
	 *
	 * @param in max_num max number of front
	 */
    void init(int max_num)
    {
        //_iMax = max_num;
        _iMax = max_num * 2;  //allow invalid nodes ...

        //start from the root, so at lest one ...
        uint dummy[] = { 1 };
        cutilSafeCall(cudaMalloc(( void** )&_dIdx, 1 * sizeof(uint)));
        cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

        cutilSafeCall(cudaMalloc(( void** )&_dFront, _iMax * sizeof(uint3)));
        cutilSafeCall(cudaMemset(_dFront, 0, _iMax * sizeof(uint3)));
        printf("max self-cone front num = %d", _iMax);

        reportMemory("g_cone_front.init");
    }

    /**
	 * deallocate all resources
	 */
    void destroy()
    {
        cudaFree(_dFront);
        cudaFree(_dIdx);
    }

    /**
	 * deallocate all resources
	 */
    void reset()
    {
        //start from the root
        uint dummy[] = { 1 };
        cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

        uint3 dummy2;
        dummy2.x = dummy2.y = dummy2.z = 0;
        cutilSafeCall(cudaMemcpy(_dFront, &dummy2, 1 * sizeof(uint3), cudaMemcpyHostToDevice));
    }

    /**
	 * push data
	 *
	 * @param[in] length length of the data
	 * @param[in] front data pointer
	 */
    void push(int length, uint3* data)
    {
        uint dummy[] = { length };
        cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
        if (length)
            cutilSafeCall(cudaMemcpy(_dFront, data, length * sizeof(uint3), cudaMemcpyHostToDevice));
    }
} g_cone_front;

#define SAFE_FRONT_NUM 54000000
#define MAX_FRONT_NUM 55000000

/**
 * Device bvh front data structure
 */
typedef struct
{
    uint4* _dFront;  //<! left, right, valid 0/1, dummy
    uint*  _dIdx;    //<! index

    /**
	 * init front
	 */
    void init()
    {
        uint dummy[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        cutilSafeCall(cudaMalloc(( void** )&_dIdx, 10 * sizeof(uint)));
        cutilSafeCall(cudaMemcpy(_dIdx, dummy, 10 * sizeof(uint), cudaMemcpyHostToDevice));

        cutilSafeCall(cudaMalloc(( void** )&_dFront, MAX_FRONT_NUM * sizeof(uint4)));
        cutilSafeCall(cudaMemset(_dFront, 0, MAX_FRONT_NUM * sizeof(uint4)));
        reportMemory("g_front.init");
    }

    /**
	 * deallocate all resources
	 */
    void destroy()
    {
        cudaFree(_dFront);
        cudaFree(_dIdx);
    }

    /**
	 * push data to gpu
	 *
	 * @param[in] length number of the data
	 * @param[in] data   data pointer
	 */
    void push(int length, uint4* data)
    {
        uint dummy[] = { length };
        cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
        if (length)
            cutilSafeCall(cudaMemcpy(_dFront, data, length * sizeof(uint4), cudaMemcpyHostToDevice));
    }

    /**
	 * propogate kernel function
	 *
	 * @param[out] update number of the data
	 * @param[in]  self   for identification
	 * @param[in]  ccd    use ccd
	 */
    int propogate(bool& update, bool self, bool ccd);

    /**
	 * get length of the indices
	 *
	 * @return length of the indices
	 */
    uint length()
    {
        uint dummy[] = { 0 };
        cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
        return dummy[0];
    }

    /**
	 * set length of the indices
	 */
    void setLength(uint len)
    {
        cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
    }
} g_front;

/**
 * get the parent of the bvh node
 *
 * @param[in] i       index
 * @param[in] bvh_ids pointer to bvh ids
 * @param[in] num     offset
 */
inline __device__ int getParent(uint i, int* bvh_ids, int num)
{
    return bvh_ids[num + i];
}

/**
 * get triangle id of the bvh node
 *
 * @param[in] i       index
 * @param[in] bvh_ids pointer to bvh ids
 */
inline __device__ int getTriID(uint i, int* bvh_ids)
{
    return bvh_ids[i];
}

/**
 * get left child the bvh node
 *
 * @param[in] i       index
 * @param[in] bvh_ids pointer to bvh ids
 * @return index of the left child
 */
inline __device__ int getLeftChild(uint i, int* bvh_ids)
{
    return i - bvh_ids[i];
}

/**
 * get left child the bvh node
 *
 * @param[in] i       index
 * @param[in] bvh_ids pointer to bvh ids
 * @return index of the right child
 */
inline __device__ int getRightChild(uint i, int* bvh_ids)
{
    return i - bvh_ids[i] + 1;
}

/**
 * check the current node is leaf
 *
 * @param[in] i       index
 * @param[in] bvh_ids pointer to bvh ids
 * @return whether the node is leave node
 */
inline __device__ bool isLeaf(uint i, int* bvh_ids)
{
    return bvh_ids[i] >= 0;
}

/**
 * check if two boxes overlaped
 *
 * @param[in] i       index i
 * @param[in] i       index j
 * @param[in] Abxs    pointer to boxes containing i-th box
 * @param[in] Bbxs    pointer to boxes containing J-th box
 * @return whether two boxes overlaped
 */
inline __device__ bool overlaps(uint i, uint j, g_box* Abxs, g_box* Bbxs)
{
    return Abxs[i].overlaps(Bbxs[j]);
}

/**
 * refit device assistant function
 *
 * @param[in] i           index i
 * @param[in] bvh_ids     pointer to bvh ids
 * @param[in] bvh_boxess  pointer to bvh boxes
 * @param[in] tri_boxess  pointer to triangle boxes
 * @param[in] bvh_cones   pointer to bvh cones
 * @param[in] tri_cones   pointer to triangle cones
 */
inline __device__ void refit(int i, int* bvh_ids, g_box* bvh_boxes, g_box* tri_boxes, g_cone* bvh_cones, g_cone* tri_cones)
{
    if (isLeaf(i, bvh_ids))  // isLeaf
    {
        int fid      = getTriID(i, bvh_ids);
        bvh_boxes[i] = tri_boxes[fid];

        if (bvh_cones)
            bvh_cones[i].set(tri_cones[fid]);
    }
    else
    {
        int left  = getLeftChild(i, bvh_ids);
        int right = getRightChild(i, bvh_ids);

        bvh_boxes[i].set(bvh_boxes[left], bvh_boxes[right]);

        if (bvh_cones)
        {
            bvh_cones[i].set(bvh_cones[left]);
            bvh_cones[i] += bvh_cones[right];
        }
    }
}

/**
 * refit algorithm serialable
 *
 * @param[in] bvh_ids     pointer to bvh ids
 * @param[in] bvh_boxess  pointer to bvh boxes
 * @param[in] tri_boxess  pointer to triangle boxes
 * @param[in] bvh_cones   pointer to bvh cones
 * @param[in] tri_cones   pointer to triangle cones
 * @param[in] num         numbers
 */
__global__ void refit_serial_kernel(int* bvh_ids, g_box* bvh_boxes, g_box* tri_boxes, g_cone* bvh_cones, g_cone* tri_cones, int num)
{
    for (int i = num - 1; i >= 0; i--)
    {
        refit(i, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
    }
}

/**
 * refit algorithm
 *
 * @param[in] bvh_ids     pointer to bvh ids
 * @param[in] bvh_boxess  pointer to bvh boxes
 * @param[in] tri_boxess  pointer to triangle boxes
 * @param[in] bvh_cones   pointer to bvh cones
 * @param[in] tri_cones   pointer to triangle cones
 * @param[in] num         numbers
 */
__global__ void refit_kernel(int* bvh_ids, g_box* bvh_boxes, g_box* tri_boxes, g_cone* bvh_cones, g_cone* tri_cones, int st, int num)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num)
        return;

    refit(idx + st, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
}
