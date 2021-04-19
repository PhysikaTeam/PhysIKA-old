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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
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

#ifndef __OPENCV_internal_shared_HPP__
#define __OPENCV_internal_shared_HPP__

#include <cuda_runtime.h>
#include <npp.h>
#include "NPP_staging.hpp"
#include "opencv2/gpu/devmem2d.hpp"
#include "safe_call.hpp"
#include "opencv2/gpu/device/common.hpp"

namespace cv { namespace gpu
{
    enum
    {
        BORDER_REFLECT101_GPU = 0,
        BORDER_REPLICATE_GPU,
        BORDER_CONSTANT_GPU,
        BORDER_REFLECT_GPU,
        BORDER_WRAP_GPU
    };

    class NppStreamHandler
    {
    public:
        inline explicit NppStreamHandler(cudaStream_t newStream = 0)
        {
            oldStream = nppGetStream();
            nppSetStream(newStream);
        }

        inline ~NppStreamHandler()
        {
            nppSetStream(oldStream);
        }

    private:
        cudaStream_t oldStream;
    };

    class NppStStreamHandler
    {
    public:
        inline explicit NppStStreamHandler(cudaStream_t newStream = 0)
        {
            oldStream = nppStSetActiveCUDAstream(newStream);
        }

        inline ~NppStStreamHandler()
        {
            nppStSetActiveCUDAstream(oldStream);
        }

    private:
        cudaStream_t oldStream;
    };
}}

#endif /* __OPENCV_internal_shared_HPP__ */
