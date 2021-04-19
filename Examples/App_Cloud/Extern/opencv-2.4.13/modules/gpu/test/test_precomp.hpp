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

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <cmath>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <limits>
#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "cvconfig.h"

#ifdef HAVE_CUDA
    #include <cuda_runtime.h>

    #include "opencv2/ts/ts.hpp"
    #include "opencv2/ts/ts_perf.hpp"
    #include "opencv2/ts/gpu_test.hpp"

    #include "opencv2/core/core.hpp"
    #include "opencv2/core/opengl_interop.hpp"
    #include "opencv2/highgui/highgui.hpp"
    #include "opencv2/calib3d/calib3d.hpp"
    #include "opencv2/imgproc/imgproc.hpp"
    #include "opencv2/video/video.hpp"
    #include "opencv2/gpu/gpu.hpp"
    #include "opencv2/legacy/legacy.hpp"

    #include "interpolation.hpp"
    #include "main_test_nvidia.h"
#endif

#endif
