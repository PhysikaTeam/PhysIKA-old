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
using std::tr1::tuple;
using std::tr1::get;

///////////// ConvertTo////////////////////////

typedef Size_MatType ConvertToFixture;

OCL_PERF_TEST_P(ConvertToFixture, ConvertTo, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst;
    const int dstType = CV_MAKE_TYPE(CV_32F, src.channels());

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    dst.create(srcSize, dstType);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, dstType);

        OCL_TEST_CYCLE() oclSrc.convertTo(oclDst, dstType);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.convertTo(dst, dstType);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// copyTo////////////////////////

typedef Size_MatType CopyToFixture;

OCL_PERF_TEST_P(CopyToFixture, CopyTo, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() oclSrc.copyTo(oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.copyTo(dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// setTo////////////////////////

typedef Size_MatType SetToFixture;

OCL_PERF_TEST_P(SetToFixture, SetTo, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar val(1, 2, 3, 4);

    Mat src(srcSize, type);
    declare.in(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(srcSize, type);

        OCL_TEST_CYCLE() oclSrc.setTo(val);
        oclSrc.download(src);

        SANITY_CHECK(src);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.setTo(val);

        SANITY_CHECK(src);
    }
    else
        OCL_PERF_ELSE
}

#if 0

/////////////////// upload ///////////////////////////

typedef tuple<Size, MatDepth, int> UploadParams;
typedef TestBaseWithParam<uploadParams> UploadFixture;

PERF_TEST_P(UploadFixture, Upload,
            testing::Combine(
                OCL_TYPICAL_MAT_SIZES,
                testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F),
                testing::Range(1, 5)))
{
    const UploadParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params);
    const int type = CV_MAKE_TYPE(depth, cn);

    Mat src(srcSize, type), dst;
    declare.in(src, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclDst;

        for(; startTimer(), next(); ocl::finish(), stopTimer(), oclDst.release())
            oclDst.upload(src);
    }
    else if (RUN_PLAIN_IMPL)
    {
        for(; startTimer(), next(); ocl::finish(), stopTimer(), dst.release())
            dst = src.clone();
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK_NOTHING();
}

/////////////////// download ///////////////////////////

typedef TestBaseWithParam<uploadParams> DownloadFixture;

PERF_TEST_P(DownloadFixture, Download,
            testing::Combine(
                OCL_TYPICAL_MAT_SIZES,
                testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F),
                testing::Range(1, 5)))
{
    const UploadParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int depth = get<1>(params), cn = get<2>(params);
    const int type = CV_MAKE_TYPE(depth, cn);

    Mat src(srcSize, type), dst;
    declare.in(src, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src);

        for(; startTimer(), next(); ocl::finish(), stopTimer(), dst.release())
            oclSrc.download(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        for(; startTimer(), next(); ocl::finish(), stopTimer(), dst.release())
            dst = src.clone();
    }
    else
        OCL_PERF_ELSE

    SANITY_CHECK_NOTHING();
}

#endif
