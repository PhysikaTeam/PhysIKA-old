//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin zero.lin@amd.com
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
//

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef BORDER_CONSTANT
#define EXTRAPOLATE(x, y, v) v = scalar;
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, y, v) \
    { \
        x = max(min(x, src_cols - 1), 0); \
        y = max(min(y, src_rows - 1), 0); \
        v = src[mad24(y, src_step, x + src_offset)]; \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, y, v) \
    { \
        if (x < 0) \
            x -= ((x - src_cols + 1) / src_cols) * src_cols; \
        if (x >= src_cols) \
            x %= src_cols; \
        \
        if (y < 0) \
            y -= ((y - src_rows + 1) / src_rows) * src_rows; \
        if( y >= src_rows ) \
            y %= src_rows; \
        v = src[mad24(y, src_step, x + src_offset)]; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#ifdef BORDER_REFLECT
#define DELTA int delta = 0
#else
#define DELTA int delta = 1
#endif
#define EXTRAPOLATE(x, y, v) \
    { \
        DELTA; \
        if (src_cols == 1) \
            x = 0; \
        else \
            do \
            { \
                if( x < 0 ) \
                    x = -x - 1 + delta; \
                else \
                    x = src_cols - 1 - (x - src_cols) - delta; \
            } \
            while (x >= src_cols || x < 0); \
        \
        if (src_rows == 1) \
            y = 0; \
        else \
            do \
            { \
                if( y < 0 ) \
                    y = -y - 1 + delta; \
                else \
                    y = src_rows - 1 - (y - src_rows) - delta; \
            } \
            while (y >= src_rows || y < 0); \
        v = src[mad24(y, src_step, x + src_offset)]; \
    }
#else
#error No extrapolation method
#endif

#define NEED_EXTRAPOLATION(gx, gy) (gx >= src_cols || gy >= src_rows || gx < 0 || gy < 0)

__kernel void copymakeborder
                        (__global const GENTYPE *src,
                         __global GENTYPE *dst,
                         int dst_cols, int dst_rows,
                         int src_cols, int src_rows,
                         int src_step, int src_offset,
                         int dst_step, int dst_offset,
                         int top, int left, GENTYPE scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src_x = x - left;
        int src_y = y - top;
        int dst_index = mad24(y, dst_step, x + dst_offset);

        if (NEED_EXTRAPOLATION(src_x, src_y))
            EXTRAPOLATE(src_x, src_y, dst[dst_index])
        else
        {
            int src_index = mad24(src_y, src_step, src_x + src_offset);
            dst[dst_index] = src[src_index];
        }
    }
}
