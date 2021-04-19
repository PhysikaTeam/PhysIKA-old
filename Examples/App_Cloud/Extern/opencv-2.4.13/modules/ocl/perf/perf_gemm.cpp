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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "perf_precomp.hpp"

using namespace perf;
using std::tr1::get;
using std::tr1::tuple;

///////////// gemm ////////////////////////

#ifdef HAVE_CLAMDBLAS

typedef tuple<Size, int> GemmParams;
typedef TestBaseWithParam<GemmParams> GemmFixture;

OCL_PERF_TEST_P(GemmFixture, Gemm, ::testing::Combine(
                    ::testing::Values(Size(1000, 1000), Size(1500, 1500)),
            ::testing::Values((int)cv::GEMM_1_T, (int)cv::GEMM_1_T | (int)cv::GEMM_2_T)))
{
    const GemmParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src1(srcSize, CV_32FC1), src2(srcSize, CV_32FC1),
            src3(srcSize, CV_32FC1), dst(srcSize, CV_32FC1);

    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);
    randu(src3, -10.0f, 10.0f);

    declare.in(src1, src2, src3).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2),
                oclSrc3(src3), oclDst(srcSize, CV_32FC1);

        OCL_TEST_CYCLE() cv::ocl::gemm(oclSrc1, oclSrc2, 1.0, oclSrc3, 1.0, oclDst, type);

        oclDst.download(dst);

        SANITY_CHECK(dst, 0.01);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::gemm(src1, src2, 1.0, src3, 1.0, dst, type);

        SANITY_CHECK(dst, 0.01);
    }
    else
        OCL_PERF_ELSE
}

#endif
