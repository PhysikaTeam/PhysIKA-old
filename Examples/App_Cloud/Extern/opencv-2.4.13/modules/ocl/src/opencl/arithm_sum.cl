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
//    Shengen Yan,yanshengen@gmail.com
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
#endif

#if FUNC_SUM
#define FUNC(a, b) b += a;
#elif FUNC_ABS_SUM
#define FUNC(a, b) b += a >= (dstT)(0) ? a : -a;
#elif FUNC_SQR_SUM
#define FUNC(a, b) b += a * a;
#else
#error No sum function
#endif

/**************************************Array buffer SUM**************************************/

__kernel void arithm_op_sum(__global srcT * src, int src_step, int src_offset, int src_cols,
                            int total, int groupnum, __global dstT * dst)
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int id = get_global_id(0);

    __local dstT localmem_sum[128];
    dstT sum = (dstT)(0), temp;

    for (int grainSize = groupnum << 8; id < total; id += grainSize)
    {
        temp = convertToDstT(src[mad24(id / src_cols, src_step, id % src_cols + src_offset)]);
        FUNC(temp, sum);
    }

    if (lid > 127)
        localmem_sum[lid - 128] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 128)
        localmem_sum[lid] = sum + localmem_sum[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = 64; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;
            localmem_sum[lid] = localmem_sum[lid] + localmem_sum[lid2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        dst[gid] = localmem_sum[0];
}
