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
#include <functional>

using namespace perf;

///////////// HOG////////////////////////

struct RectLess :
        public std::binary_function<cv::Rect, cv::Rect, bool>
{
    bool operator()(const cv::Rect& a,
        const cv::Rect& b) const
    {
        if (a.x != b.x)
            return a.x < b.x;
        else if (a.y != b.y)
            return a.y < b.y;
        else if (a.width != b.width)
            return a.width < b.width;
        else
            return a.height < b.height;
    }
};

OCL_PERF_TEST(HOGFixture, HOG)
{
    Mat src = imread(getDataPath("gpu/hog/road.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_TRUE(!src.empty()) << "can't open input image road.png";

    vector<cv::Rect> found_locations;
    declare.in(src);

    if (RUN_PLAIN_IMPL)
    {
        HOGDescriptor hog;
        hog.setSVMDetector(hog.getDefaultPeopleDetector());

        TEST_CYCLE() hog.detectMultiScale(src, found_locations);

        std::sort(found_locations.begin(), found_locations.end(), RectLess());
        SANITY_CHECK(found_locations, 1 + DBL_EPSILON);
    }
    else if (RUN_OCL_IMPL)
    {
        ocl::HOGDescriptor ocl_hog;
        ocl_hog.setSVMDetector(ocl_hog.getDefaultPeopleDetector());
        ocl::oclMat oclSrc(src);

        OCL_TEST_CYCLE() ocl_hog.detectMultiScale(oclSrc, found_locations);

        std::sort(found_locations.begin(), found_locations.end(), RectLess());
        SANITY_CHECK(found_locations, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}
