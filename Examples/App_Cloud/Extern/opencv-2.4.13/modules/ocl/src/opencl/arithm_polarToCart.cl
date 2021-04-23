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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define CV_PI M_PI
#else
#define CV_PI M_PI_F
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////polarToCart with magnitude//////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void arithm_polarToCart_mag_D5 (__global float *src1, int src1_step, int src1_offset,//magnitue
                                         __global float *src2, int src2_step, int src2_offset,//angle
                                         __global float *dst1, int dst1_step, int dst1_offset,
                                         __global float *dst2, int dst2_step, int dst2_offset,
                                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, x + src1_offset);
        int src2_index = mad24(y, src2_step, x + src2_offset);

        int dst1_index = mad24(y, dst1_step, x + dst1_offset);
        int dst2_index = mad24(y, dst2_step, x + dst2_offset);

        float x = src1[src1_index];
        float y = src2[src2_index];

#ifdef DEGREE
        float ascale = CV_PI/180.0f;
        float alpha = y * ascale;
#else
        float alpha = y;
#endif
        float a = cos(alpha) * x;
        float b = sin(alpha) * x;

        dst1[dst1_index] = a;
        dst2[dst2_index] = b;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_polarToCart_mag_D6 (__global double *src1, int src1_step, int src1_offset,//magnitue
                                         __global double *src2, int src2_step, int src2_offset,//angle
                                         __global double *dst1, int dst1_step, int dst1_offset,
                                         __global double *dst2, int dst2_step, int dst2_offset,
                                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, x + src1_offset);
        int src2_index = mad24(y, src2_step, x + src2_offset);

        int dst1_index = mad24(y, dst1_step, x + dst1_offset);
        int dst2_index = mad24(y, dst2_step, x + dst2_offset);

        double x = src1[src1_index];
        double y = src2[src2_index];

#ifdef DEGREE
        float ascale = CV_PI/180.0;
        float alpha = y * ascale;
#else
        float alpha = y;
#endif
        double a = cos(alpha) * x;
        double b = sin(alpha) * x;

        dst1[dst1_index] = a;
        dst2[dst2_index] = b;
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////polarToCart without magnitude//////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void arithm_polarToCart_D5 (__global float *src,  int src_step,  int src_offset,//angle
                                     __global float *dst1, int dst1_step, int dst1_offset,
                                     __global float *dst2, int dst2_step, int dst2_offset,
                                     int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index  = mad24(y, src_step,  x + src_offset);

        int dst1_index = mad24(y, dst1_step, x + dst1_offset);
        int dst2_index = mad24(y, dst2_step, x + dst2_offset);

        float y = src[src_index];

#ifdef DEGREE
        float ascale = CV_PI/180.0f;
        float alpha = y * ascale;
#else
        float alpha = y;
#endif
        float a = cos(alpha);
        float b = sin(alpha);

        dst1[dst1_index] = a;
        dst2[dst2_index] = b;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_polarToCart_D6 (__global float *src,  int src_step,  int src_offset,//angle
                                     __global float *dst1, int dst1_step, int dst1_offset,
                                     __global float *dst2, int dst2_step, int dst2_offset,
                                     int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index  = mad24(y, src_step,  x + src_offset);

        int dst1_index = mad24(y, dst1_step, x + dst1_offset);
        int dst2_index = mad24(y, dst2_step, x + dst2_offset);

        double y = src[src_index];

#ifdef DEGREE
        float ascale = CV_PI/180.0f;
        float alpha = y * ascale;
#else
        float alpha = y;
#endif
        double a = cos(alpha);
        double b = sin(alpha);

        dst1[dst1_index] = a;
        dst2[dst2_index] = b;
    }
}
#endif
