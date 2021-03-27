/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Guoping Long, longguoping@gmail.com
//    Niko Li, newlife20080214@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <iomanip>
#include <fstream>
#include "cl_programcache.hpp"

//#define PRINT_KERNEL_RUN_TIME
#define RUN_TIMES 100
#ifndef CL_MEM_USE_PERSISTENT_MEM_AMD
#define CL_MEM_USE_PERSISTENT_MEM_AMD 0
#endif
//#define AMD_DOUBLE_DIFFER

namespace cv {
namespace ocl {

DevMemType gDeviceMemType = DEVICE_MEM_DEFAULT;
DevMemRW gDeviceMemRW = DEVICE_MEM_R_W;
int gDevMemTypeValueMap[5] = {0,
                              CL_MEM_ALLOC_HOST_PTR,
                              CL_MEM_USE_HOST_PTR,
                              CL_MEM_COPY_HOST_PTR,
                              CL_MEM_USE_PERSISTENT_MEM_AMD};
int gDevMemRWValueMap[3] = {CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};

void finish()
{
    clFinish(getClCommandQueue(Context::getContext()));
}

bool isCpuDevice()
{
    const DeviceInfo& info = Context::getContext()->getDeviceInfo();
    return (info.deviceType == CVCL_DEVICE_TYPE_CPU);
}

size_t queryWaveFrontSize(cl_kernel kernel)
{
    const DeviceInfo& info = Context::getContext()->getDeviceInfo();
    if (info.deviceType == CVCL_DEVICE_TYPE_CPU)
        return 1;
    size_t wavefront = 0;
    CV_Assert(kernel != NULL);
    openCLSafeCall(clGetKernelWorkGroupInfo(kernel, getClDeviceID(Context::getContext()),
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &wavefront, NULL));
    return wavefront;
}


void openCLReadBuffer(Context *ctx, cl_mem dst_buffer, void *host_buffer, size_t size)
{
    cl_int status;
    status = clEnqueueReadBuffer(getClCommandQueue(ctx), dst_buffer, CL_TRUE, 0,
                                 size, host_buffer, 0, NULL, NULL);
    openCLVerifyCall(status);
}

cl_mem openCLCreateBuffer(Context *ctx, size_t flag , size_t size)
{
    cl_int status;
    cl_mem buffer = clCreateBuffer(getClContext(ctx), (cl_mem_flags)flag, size, NULL, &status);
    openCLVerifyCall(status);
    return buffer;
}

#define MEMORY_CORRUPTION_GUARD
#ifdef MEMORY_CORRUPTION_GUARD
//#define CHECK_MEMORY_CORRUPTION
#define CHECK_MEMORY_CORRUPTION_PRINT_ERROR
#define CHECK_MEMORY_CORRUPTION_RAISE_ERROR
static const int __memory_corruption_guard_bytes = 64*1024;
#ifdef CHECK_MEMORY_CORRUPTION
static const int __memory_corruption_check_pattern = 0x14326547; // change pattern for sizeof(int)==8
#endif
struct CheckBuffers
{
    cl_mem mainBuffer;
    size_t size;
    size_t widthInBytes, height;
    CheckBuffers()
        : mainBuffer(NULL), size(0), widthInBytes(0), height(0)
    {
        // nothing
    }
    CheckBuffers(cl_mem _mainBuffer, size_t _size, size_t _widthInBytes, size_t _height)
        : mainBuffer(_mainBuffer), size(_size), widthInBytes(_widthInBytes), height(_height)
    {
        // nothing
    }
};
static std::map<cl_mem, CheckBuffers> __check_buffers;
#endif

void openCLMallocPitch(Context *ctx, void **dev_ptr, size_t *pitch,
                       size_t widthInBytes, size_t height)
{
    openCLMallocPitchEx(ctx, dev_ptr, pitch, widthInBytes, height, gDeviceMemRW, gDeviceMemType);
}

void openCLMallocPitchEx(Context *ctx, void **dev_ptr, size_t *pitch,
                       size_t widthInBytes, size_t height, DevMemRW rw_type, DevMemType mem_type)
{
    cl_int status;
    size_t size = widthInBytes * height;
    bool useSubBuffers =
#ifndef MEMORY_CORRUPTION_GUARD
            false;
#else
            true;
#endif
    const DeviceInfo& devInfo = ctx->getDeviceInfo();
    if (useSubBuffers && devInfo.isIntelDevice)
    {
        useSubBuffers = false; // TODO FIXIT We observe memory leaks then we working with sub-buffers
                               // on the CPU device of Intel OpenCL SDK (Linux). We will investigate this later.
    }
    if (!useSubBuffers)
    {
        *dev_ptr = clCreateBuffer(getClContext(ctx), gDevMemRWValueMap[rw_type]|gDevMemTypeValueMap[mem_type],
                                  size, 0, &status);
        openCLVerifyCall(status);
    }
#ifdef MEMORY_CORRUPTION_GUARD
    else
    {
        size_t allocSize = size + __memory_corruption_guard_bytes * 2;
        cl_mem mainBuffer = clCreateBuffer(getClContext(ctx), gDevMemRWValueMap[rw_type]|gDevMemTypeValueMap[mem_type],
                allocSize, 0, &status);
        openCLVerifyCall(status);
        cl_buffer_region r = {__memory_corruption_guard_bytes, size};
        *dev_ptr = clCreateSubBuffer(mainBuffer,
                gDevMemRWValueMap[rw_type]|gDevMemTypeValueMap[mem_type],
                CL_BUFFER_CREATE_TYPE_REGION, &r,
                &status);
        openCLVerifyCall(status);
#ifdef CHECK_MEMORY_CORRUPTION
        std::vector<int> tmp(__memory_corruption_guard_bytes / sizeof(int),
                __memory_corruption_check_pattern);
        CV_Assert(tmp.size() * sizeof(int) == __memory_corruption_guard_bytes);
        openCLVerifyCall(clEnqueueWriteBuffer(getClCommandQueue(ctx),
                mainBuffer, CL_FALSE, 0, __memory_corruption_guard_bytes, &tmp[0],
                0, NULL, NULL));
        openCLVerifyCall(clEnqueueWriteBuffer(getClCommandQueue(ctx),
                mainBuffer, CL_FALSE, __memory_corruption_guard_bytes + size, __memory_corruption_guard_bytes, &tmp[0],
                0, NULL, NULL));
        clFinish(getClCommandQueue(ctx));
#endif
        CheckBuffers data(mainBuffer, size, widthInBytes, height);
        cv::AutoLock lock(getInitializationMutex());
        __check_buffers.insert(std::pair<cl_mem, CheckBuffers>((cl_mem)*dev_ptr, data));
    }
#endif
    *pitch = widthInBytes;
}

void openCLMemcpy2D(Context *ctx, void *dst, size_t dpitch,
                    const void *src, size_t spitch,
                    size_t width, size_t height, openCLMemcpyKind kind, int channels)
{
    size_t buffer_origin[3] = {0, 0, 0};
    size_t host_origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};
    if(kind == clMemcpyHostToDevice)
    {
        if(dpitch == width || channels == 3 || height == 1)
        {
            openCLSafeCall(clEnqueueWriteBuffer(getClCommandQueue(ctx), (cl_mem)dst, CL_TRUE,
                                                0, width * height, src, 0, NULL, NULL));
        }
        else
        {
            openCLSafeCall(clEnqueueWriteBufferRect(getClCommandQueue(ctx), (cl_mem)dst, CL_TRUE,
                                                    buffer_origin, host_origin, region, dpitch, 0, spitch, 0, src, 0, 0, 0));
        }
    }
    else if(kind == clMemcpyDeviceToHost)
    {
        if(spitch == width || channels == 3 || height == 1)
        {
            openCLSafeCall(clEnqueueReadBuffer(getClCommandQueue(ctx), (cl_mem)src, CL_TRUE,
                                               0, width * height, dst, 0, NULL, NULL));
        }
        else
        {
            openCLSafeCall(clEnqueueReadBufferRect(getClCommandQueue(ctx), (cl_mem)src, CL_TRUE,
                                                   buffer_origin, host_origin, region, spitch, 0, dpitch, 0, dst, 0, 0, 0));
        }
    }
}

void openCLCopyBuffer2D(Context *ctx, void *dst, size_t dpitch, int dst_offset,
                        const void *src, size_t spitch,
                        size_t width, size_t height, int src_offset)
{
    size_t src_origin[3] = {src_offset % spitch, src_offset / spitch, 0};
    size_t dst_origin[3] = {dst_offset % dpitch, dst_offset / dpitch, 0};
    size_t region[3] = {width, height, 1};

    openCLSafeCall(clEnqueueCopyBufferRect(getClCommandQueue(ctx), (cl_mem)src, (cl_mem)dst, src_origin, dst_origin,
                                           region, spitch, 0, dpitch, 0, 0, 0, 0));
}

void openCLFree(void *devPtr)
{
    openCLSafeCall(clReleaseMemObject((cl_mem)devPtr));
#ifdef MEMORY_CORRUPTION_GUARD
#ifdef CHECK_MEMORY_CORRUPTION
    bool failBefore = false, failAfter = false;
#endif
    CheckBuffers data;
    {
        cv::AutoLock lock(getInitializationMutex());
        std::map<cl_mem, CheckBuffers>::iterator i = __check_buffers.find((cl_mem)devPtr);
        if (i != __check_buffers.end())
        {
            data = i->second;
            __check_buffers.erase(i);
        }
    }
    if (data.mainBuffer != NULL)
    {
#ifdef CHECK_MEMORY_CORRUPTION
        Context* ctx = Context::getContext();
        std::vector<uchar> checkBefore(__memory_corruption_guard_bytes);
        std::vector<uchar> checkAfter(__memory_corruption_guard_bytes);
        openCLVerifyCall(clEnqueueReadBuffer(getClCommandQueue(ctx),
                data.mainBuffer, CL_FALSE, 0, __memory_corruption_guard_bytes, &checkBefore[0],
                0, NULL, NULL));
        openCLVerifyCall(clEnqueueReadBuffer(getClCommandQueue(ctx),
                data.mainBuffer, CL_FALSE, __memory_corruption_guard_bytes + data.size, __memory_corruption_guard_bytes, &checkAfter[0],
                0, NULL, NULL));
        clFinish(getClCommandQueue(ctx));

        std::vector<int> tmp(__memory_corruption_guard_bytes / sizeof(int),
                __memory_corruption_check_pattern);

        if (memcmp(&checkBefore[0], &tmp[0], __memory_corruption_guard_bytes) != 0)
        {
            failBefore = true;
        }
        if (memcmp(&checkAfter[0], &tmp[0], __memory_corruption_guard_bytes) != 0)
        {
            failAfter = true;
        }
#else
        // TODO FIXIT Attach clReleaseMemObject call to event completion callback
        // TODO 2013/12/04 Disable workaround
        // Context* ctx = Context::getContext();
        // clFinish(getClCommandQueue(ctx));
#endif
        openCLSafeCall(clReleaseMemObject(data.mainBuffer));
    }
#if defined(CHECK_MEMORY_CORRUPTION)
    if (failBefore)
    {
#ifdef CHECK_MEMORY_CORRUPTION_PRINT_ERROR
        std::cerr << "ERROR: Memory corruption detected: before buffer: " << cv::format("widthInBytes=%d height=%d", (int)data.widthInBytes, (int)data.height) << std::endl;
#endif
#ifdef CHECK_MEMORY_CORRUPTION_RAISE_ERROR
        CV_Error(CV_StsInternal, "Memory corruption detected: before buffer");
#endif
    }
    if (failAfter)
    {
#ifdef CHECK_MEMORY_CORRUPTION_PRINT_ERROR
        std::cerr << "ERROR: Memory corruption detected: after buffer: " << cv::format("widthInBytes=%d height=%d", (int)data.widthInBytes, (int)data.height) << std::endl;
#endif
#ifdef CHECK_MEMORY_CORRUPTION_RAISE_ERROR
        CV_Error(CV_StsInternal, "Memory corruption detected: after buffer");
#endif
    }
#endif // CHECK_MEMORY_CORRUPTION
#endif // MEMORY_CORRUPTION_GUARD
}

cl_kernel openCLGetKernelFromSource(const Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName)
{
    return openCLGetKernelFromSource(ctx, source, kernelName, NULL);
}

cl_kernel openCLGetKernelFromSource(const Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName,
                                    const char *build_options)
{
    cl_kernel kernel;
    cl_int status = 0;
    CV_Assert(ProgramCache::getProgramCache() != NULL);
    cl_program program = ProgramCache::getProgramCache()->getProgram(ctx, source, build_options);
    CV_Assert(program != NULL);
    kernel = clCreateKernel(program, kernelName.c_str(), &status);
    openCLVerifyCall(status);
    openCLVerifyCall(clReleaseProgram(program));
    return kernel;
}

void openCLVerifyKernel(const Context *ctx, cl_kernel kernel, size_t *localThreads)
{
    size_t kernelWorkGroupSize;
    openCLSafeCall(clGetKernelWorkGroupInfo(kernel, getClDeviceID(ctx),
                                            CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernelWorkGroupSize, 0));
    CV_Assert( localThreads[0] <= ctx->getDeviceInfo().maxWorkItemSizes[0] );
    CV_Assert( localThreads[1] <= ctx->getDeviceInfo().maxWorkItemSizes[1] );
    CV_Assert( localThreads[2] <= ctx->getDeviceInfo().maxWorkItemSizes[2] );
    CV_Assert( localThreads[0] * localThreads[1] * localThreads[2] <= kernelWorkGroupSize );
    CV_Assert( localThreads[0] * localThreads[1] * localThreads[2] <= ctx->getDeviceInfo().maxWorkGroupSize );
}

#ifdef PRINT_KERNEL_RUN_TIME
static double total_execute_time = 0;
static double total_kernel_time = 0;
#endif

static std::string removeDuplicatedWhiteSpaces(const char * buildOptions)
{
    if (buildOptions == NULL)
        return "";

    size_t length = strlen(buildOptions), didx = 0, sidx = 0;
    while (sidx < length && buildOptions[sidx] == 0)
        ++sidx;

    std::string opt;
    opt.resize(length);

    for ( ; sidx < length; ++sidx)
        if (buildOptions[sidx] != ' ')
            opt[didx++] = buildOptions[sidx];
        else if ( !(didx > 0 && opt[didx - 1] == ' ') )
            opt[didx++] = buildOptions[sidx];

    return opt;
}

cl_kernel openCLGetKernelFromSource(Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName, int channels,
                          int depth, const char *build_options)
{
    //construct kernel name
    //The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
    //for example split_C2_D3, represent the split kernel with channels = 2 and dataType Depth = 3(Data type is short)
    stringstream idxStr;
    if(channels != -1)
        idxStr << "_C" << channels;
    if(depth != -1)
        idxStr << "_D" << depth;
    kernelName += idxStr.str();

    std::string fixedOptions = removeDuplicatedWhiteSpaces(build_options);
    cl_kernel kernel = openCLGetKernelFromSource(ctx, source, kernelName, fixedOptions.c_str());
    return kernel;
}

void openCLExecuteKernel(Context *ctx, cl_kernel kernel, size_t globalThreads[3],
                          size_t localThreads[3],  vector< pair<size_t, const void *> > &args)
{
    if ( localThreads != NULL)
    {
        globalThreads[0] = roundUp(globalThreads[0], localThreads[0]);
        globalThreads[1] = roundUp(globalThreads[1], localThreads[1]);
        globalThreads[2] = roundUp(globalThreads[2], localThreads[2]);

        cv::ocl::openCLVerifyKernel(ctx, kernel, localThreads);
    }
    for(size_t i = 0; i < args.size(); i ++)
        openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));

#ifndef PRINT_KERNEL_RUN_TIME
    openCLSafeCall(clEnqueueNDRangeKernel(getClCommandQueue(ctx), kernel, 3, NULL, globalThreads,
                                          localThreads, 0, NULL, NULL));
#else
    cl_event event = NULL;
    openCLSafeCall(clEnqueueNDRangeKernel(getClCommandQueue(ctx), kernel, 3, NULL, globalThreads,
                                          localThreads, 0, NULL, &event));

    cl_ulong start_time, end_time, queue_time;
    double execute_time = 0;
    double total_time   = 0;

    openCLSafeCall(clWaitForEvents(1, &event));
    openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong), &start_time, 0));

    openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong), &end_time, 0));

    openCLSafeCall(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                           sizeof(cl_ulong), &queue_time, 0));

    execute_time = (double)(end_time - start_time) / (1000 * 1000);
    total_time = (double)(end_time - queue_time) / (1000 * 1000);

    total_execute_time += execute_time;
    total_kernel_time += total_time;
    clReleaseEvent(event);
#endif

    clFlush(getClCommandQueue(ctx));
    openCLSafeCall(clReleaseKernel(kernel));
}

void openCLExecuteKernel_(Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName, size_t globalThreads[3],
                          size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
                          int depth, const char *build_options)
{
    cl_kernel kernel = openCLGetKernelFromSource(ctx, source, kernelName, channels, depth, build_options);

    openCLExecuteKernel(ctx, kernel, globalThreads, localThreads, args);
}

void openCLExecuteKernel(Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName,
                         size_t globalThreads[3], size_t localThreads[3],
                         vector< pair<size_t, const void *> > &args, int channels, int depth)
{
    openCLExecuteKernel(ctx, source, kernelName, globalThreads, localThreads, args,
                        channels, depth, NULL);
}
void openCLExecuteKernel(Context *ctx, const cv::ocl::ProgramEntry* source, string kernelName,
                         size_t globalThreads[3], size_t localThreads[3],
                         vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options)

{
#ifndef PRINT_KERNEL_RUN_TIME
    openCLExecuteKernel_(ctx, source, kernelName, globalThreads, localThreads, args, channels, depth,
                         build_options);
#else
    string data_type[] = { "uchar", "char", "ushort", "short", "int", "float", "double"};
    cout << endl;
    cout << "Function Name: " << kernelName;
    if(depth >= 0)
        cout << " |data type: " << data_type[depth];
    cout << " |channels: " << channels;
    cout << " |Time Unit: " << "ms" << endl;

    total_execute_time = 0;
    total_kernel_time = 0;
    cout << "-------------------------------------" << endl;

    cout << setiosflags(ios::left) << setw(15) << "execute time";
    cout << setiosflags(ios::left) << setw(15) << "launch time";
    cout << setiosflags(ios::left) << setw(15) << "kernel time" << endl;
    int i = 0;
    for(i = 0; i < RUN_TIMES; i++)
        openCLExecuteKernel_(ctx, source, kernelName, globalThreads, localThreads, args, channels, depth,
                             build_options);

    cout << "average kernel execute time: " << total_execute_time / RUN_TIMES << endl; // "ms" << endl;
    cout << "average kernel total time:  " << total_kernel_time / RUN_TIMES << endl; // "ms" << endl;
#endif
}

void openCLExecuteKernelInterop(Context *ctx, const cv::ocl::ProgramSource& source, string kernelName,
                         size_t globalThreads[3], size_t localThreads[3],
                         vector< pair<size_t, const void *> > &args, int channels, int depth, const char *build_options)

{
    //construct kernel name
    //The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
    //for example split_C2_D2, represent the split kernel with channels = 2 and dataType Depth = 2 (Data type is char)
    stringstream idxStr;
    if(channels != -1)
        idxStr << "_C" << channels;
    if(depth != -1)
        idxStr << "_D" << depth;
    kernelName += idxStr.str();

    std::string name = std::string("custom_") + source.name;
    ProgramEntry program = { name.c_str(), source.programStr, source.programHash };
    cl_kernel kernel = openCLGetKernelFromSource(ctx, &program, kernelName, build_options);

    CV_Assert(globalThreads != NULL);
    if ( localThreads != NULL)
    {
        globalThreads[0] = roundUp(globalThreads[0], localThreads[0]);
        globalThreads[1] = roundUp(globalThreads[1], localThreads[1]);
        globalThreads[2] = roundUp(globalThreads[2], localThreads[2]);

        cv::ocl::openCLVerifyKernel(ctx, kernel, localThreads);
    }
    for(size_t i = 0; i < args.size(); i ++)
        openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));

    openCLSafeCall(clEnqueueNDRangeKernel(getClCommandQueue(ctx), kernel, 3, NULL, globalThreads,
                    localThreads, 0, NULL, NULL));

    clFinish(getClCommandQueue(ctx));
    openCLSafeCall(clReleaseKernel(kernel));
}

cl_mem load_constant(cl_context context, cl_command_queue command_queue, const void *value,
                     const size_t size)
{
    int status;
    cl_mem con_struct;

    con_struct = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
    openCLSafeCall(status);

    openCLSafeCall(clEnqueueWriteBuffer(command_queue, con_struct, 1, 0, size,
                                        value, 0, 0, 0));

    return con_struct;
}

}//namespace ocl
}//namespace cv
