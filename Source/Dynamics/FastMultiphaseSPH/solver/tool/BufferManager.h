#pragma once

#include <string>
#include <vector>
#include <map>

#include "../../math/geometry.h"
#include "cuda_runtime_api.h"
#include "BufferHandler.h"

class BufferManager
{
public:
    int                                      num_particles;
    std::vector<HostDeviceHandler>           hd_array;
    std::vector<SortHandler>                 h_sort_handlers;
    SortHandler*                             d_sort_handlers;  //copy on device
    std::map<std::string, HostDeviceHandler> buffer_dict;

    void setupDeviceSortHandlers();

    void copy2device(int r, int len)
    {
        auto iter = buffer_dict.begin();
        for (; iter != buffer_dict.end(); iter++)
        {
            auto handler = iter->second;
            auto srchead = ( char* )handler.h_data + r * handler.element_sz;
            auto dsthead = ( char* )handler.d_data + r * handler.element_sz;

            cudaMemcpy(dsthead,
                       srchead,
                       len * handler.element_sz,
                       cudaMemcpyHostToDevice);
        }
    }

    void copy2host(int r, int len)
    {
        auto iter = buffer_dict.begin();
        for (; iter != buffer_dict.end(); iter++)
        {
            auto handler = iter->second;
            auto srchead = ( char* )handler.d_data + r * handler.element_sz;
            auto dsthead = ( char* )handler.h_data + r * handler.element_sz;
            cudaMemcpy(dsthead,
                       srchead,
                       len * handler.element_sz,
                       cudaMemcpyDeviceToHost);
        }
    }

    void copySortedData(int len)
    {
        for (auto& h : h_sort_handlers)
        {
            char* srchead = ( char* )h.sorted_data;
            char* dsthead = ( char* )h.data;
            cudaMemcpy(dsthead,
                       srchead,
                       len * h.element_sz,
                       cudaMemcpyDeviceToDevice);
        }
    }

    template <typename T>
    void allocHostDeviceBuffer(
        std::vector<T>& host_vec,
        T*&             device_data,
        int             max_nump)
    {
        HostDeviceHandler hh;
        hh.h_data     = host_vec.data();
        hh.sz         = max_nump;
        hh.element_sz = sizeof(T);

        cudaMalloc(( void** )&device_data, max_nump * sizeof(T));
        hh.d_data = device_data;
        hd_array.push_back(hh);
    }

    template <typename T>
    void alloc_sortable(
        T*& device_data,
        int max_nump)
    {
        SortHandler sh;
        sh.data       = device_data;
        sh.sz         = max_nump;
        sh.element_sz = sizeof(T);
        void* sort_buf;

        cudaMalloc(&sort_buf, max_nump * sizeof(T));
        sh.sorted_data = sort_buf;
        h_sort_handlers.push_back(sh);
    }

    template <typename T>
    void alloc_device_only(
        T*& device_data,
        int max_nump)
    {
        void** tmpp = ( void** )&device_data;
        cudaMalloc(tmpp, sizeof(T) * max_nump);
    }
};

// backward: sorted position <- original position
// traversing sorted position
void reorder_by_handlers_host(SortHandler* d_handlers,
                              uint*        sorted_indices,
                              int          num_handler,
                              int          num_p);

// forward: original position -> sorted position
// traversing original position
void reorderDataForwardHost(SortHandler* d_handlers,
                            uint*        indicesAfterSort,
                            int          num_handler,
                            int          num_p);