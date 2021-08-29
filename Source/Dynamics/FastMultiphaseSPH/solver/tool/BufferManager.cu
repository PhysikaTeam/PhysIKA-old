#include "cuda_runtime_api.h"
#include "BufferHandler.h"
#include "../cuda_common.h"

#include "BufferManager.h"

void BufferManager::setupDeviceSortHandlers()
{
    void** tmpp = ( void** )&d_sort_handlers;
    cudaMalloc(tmpp,
               sizeof(SortHandler) * h_sort_handlers.size());

    cudaMemcpy(d_sort_handlers, h_sort_handlers.data(), sizeof(SortHandler) * h_sort_handlers.size(), cudaMemcpyHostToDevice);
}

__global__ void reorder_by_handlers(SortHandler* handlers,
                                    uint*        dataIdx,
                                    int          num_handler,
                                    int          num_p)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_p)
        return;
    uint srcIdx = dataIdx[index];

    // index : target idx
    // originalIdx : source idx

    for (int i = 0; i < num_handler; i++)
    {  //for each handler
        SortHandler h           = handlers[i];
        char*       data        = ( char* )h.data;
        char*       sorted_data = ( char* )h.sorted_data;

        for (int k = 0; k < h.element_sz; k++)
        {
            sorted_data[index * h.element_sz + k] = data[srcIdx * h.element_sz + k];
        }
    }
    dataIdx[index] = index;  // point to itself
}

void reorder_by_handlers_host(SortHandler* d_handlers,
                              uint*        sorted_indices,
                              int          num_handler,
                              int          num_p)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_p, 256, num_blocks, num_threads);

    reorder_by_handlers<<<num_blocks, num_threads>>>(d_handlers,
                                                     sorted_indices,
                                                     num_handler,
                                                     num_p);
    cudaThreadSynchronize();
}

__global__ void reorderDataForward(SortHandler* handlers,
                                   uint*        indicesAfterSort,
                                   int          num_handler,
                                   int          num_p)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= num_p)
        return;
    uint targetIdx = indicesAfterSort[index];
    //printf("%d\n", targetIdx);
    for (int i = 0; i < num_handler; i++)
    {  //for each handler
        SortHandler h           = handlers[i];
        char*       data        = ( char* )h.data;
        char*       sorted_data = ( char* )h.sorted_data;

        for (int k = 0; k < h.element_sz; k++)
        {
            sorted_data[targetIdx * h.element_sz + k] = data[index * h.element_sz + k];
        }
    }
}

void reorderDataForwardHost(SortHandler* d_handlers,
                            uint*        indicesAfterSort,
                            int          num_handler,
                            int          num_p)
{
    uint num_threads, num_blocks;
    computeBlockSize(num_p, 256, num_blocks, num_threads);

    reorderDataForward<<<num_blocks, num_threads>>>(d_handlers,
                                                    indicesAfterSort,
                                                    num_handler,
                                                    num_p);
    cudaThreadSynchronize();
}