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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include <climits>

namespace cv
{

template<typename T> static inline Scalar rawToScalar(const T& v)
{
    Scalar s;
    typedef typename DataType<T>::channel_type T1;
    int i, n = DataType<T>::channels;
    for( i = 0; i < n; i++ )
        s.val[i] = ((T1*)&v)[i];
    return s;
}

/****************************************************************************************\
*                                        sum                                             *
\****************************************************************************************/

template<typename T, typename ST>
static int sum_(const T* src0, const uchar* mask, ST* dst, int len, int cn )
{
    const T* src = src0;
    if( !mask )
    {
        int i=0;
        int k = cn % 4;
        if( k == 1 )
        {
            ST s0 = dst[0];

            #if CV_ENABLE_UNROLLED
            for(; i <= len - 4; i += 4, src += cn*4 )
                s0 += src[0] + src[cn] + src[cn*2] + src[cn*3];
            #endif
            for( ; i < len; i++, src += cn )
                s0 += src[0];
            dst[0] = s0;
        }
        else if( k == 2 )
        {
            ST s0 = dst[0], s1 = dst[1];
            for( i = 0; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
            }
            dst[0] = s0;
            dst[1] = s1;
        }
        else if( k == 3 )
        {
            ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
            for( i = 0; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
            }
            dst[0] = s0;
            dst[1] = s1;
            dst[2] = s2;
        }

        for( ; k < cn; k += 4 )
        {
            src = src0 + k;
            ST s0 = dst[k], s1 = dst[k+1], s2 = dst[k+2], s3 = dst[k+3];
            for( i = 0; i < len; i++, src += cn )
            {
                s0 += src[0]; s1 += src[1];
                s2 += src[2]; s3 += src[3];
            }
            dst[k] = s0;
            dst[k+1] = s1;
            dst[k+2] = s2;
            dst[k+3] = s3;
        }
        return len;
    }

    int i, nzm = 0;
    if( cn == 1 )
    {
        ST s = dst[0];
        for( i = 0; i < len; i++ )
            if( mask[i] )
            {
                s += src[i];
                nzm++;
            }
        dst[0] = s;
    }
    else if( cn == 3 )
    {
        ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
        for( i = 0; i < len; i++, src += 3 )
            if( mask[i] )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
                nzm++;
            }
        dst[0] = s0;
        dst[1] = s1;
        dst[2] = s2;
    }
    else
    {
        for( i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                int k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= cn - 4; k += 4 )
                {
                    ST s0, s1;
                    s0 = dst[k] + src[k];
                    s1 = dst[k+1] + src[k+1];
                    dst[k] = s0; dst[k+1] = s1;
                    s0 = dst[k+2] + src[k+2];
                    s1 = dst[k+3] + src[k+3];
                    dst[k+2] = s0; dst[k+3] = s1;
                }
                #endif
                for( ; k < cn; k++ )
                    dst[k] += src[k];
                nzm++;
            }
    }
    return nzm;
}


static int sum8u( const uchar* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum8s( const schar* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16u( const ushort* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16s( const short* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32s( const int* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32f( const float* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum64f( const double* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

typedef int (*SumFunc)(const uchar*, const uchar* mask, uchar*, int, int);

static SumFunc getSumFunc(int depth)
{
    static SumFunc sumTab[] =
    {
        (SumFunc)GET_OPTIMIZED(sum8u), (SumFunc)sum8s,
        (SumFunc)sum16u, (SumFunc)sum16s,
        (SumFunc)sum32s,
        (SumFunc)GET_OPTIMIZED(sum32f), (SumFunc)sum64f,
        0
    };

    return sumTab[depth];
}

template<typename T>
static int countNonZero_(const T* src, int len )
{
    int i=0, nz = 0;
    #if CV_ENABLE_UNROLLED
    for(; i <= len - 4; i += 4 )
        nz += (src[i] != 0) + (src[i+1] != 0) + (src[i+2] != 0) + (src[i+3] != 0);
    #endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero8u( const uchar* src, int len )
{
    int i=0, nz = 0;
#if CV_SSE2
    if(USE_SSE2)//5x-6x
    {
        __m128i pattern = _mm_setzero_si128 ();
        static uchar tab[256];
        static volatile bool initialized = false;
        if( !initialized )
        {
            // we compute inverse popcount table,
            // since we pass (img[x] == 0) mask as index in the table.
            for( int j = 0; j < 256; j++ )
            {
                int val = 0;
                for( int mask = 1; mask < 256; mask += mask )
                    val += (j & mask) == 0;
                tab[j] = (uchar)val;
            }
            initialized = true;
        }

        for (; i<=len-16; i+=16)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src+i));
            int val = _mm_movemask_epi8(_mm_cmpeq_epi8(r0, pattern));
            nz += tab[val & 255] + tab[val >> 8];
        }
    }
#endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero16u( const ushort* src, int len )
{ return countNonZero_(src, len); }

static int countNonZero32s( const int* src, int len )
{ return countNonZero_(src, len); }

static int countNonZero32f( const float* src, int len )
{ return countNonZero_(src, len); }

static int countNonZero64f( const double* src, int len )
{ return countNonZero_(src, len); }

typedef int (*CountNonZeroFunc)(const uchar*, int);

static CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[] =
    {
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64f), 0
    };

    return countNonZeroTab[depth];
}

template<typename T, typename ST, typename SQT>
static int sumsqr_(const T* src0, const uchar* mask, ST* sum, SQT* sqsum, int len, int cn )
{
    const T* src = src0;

    if( !mask )
    {
        int i;
        int k = cn % 4;

        if( k == 1 )
        {
            ST s0 = sum[0];
            SQT sq0 = sqsum[0];
            for( i = 0; i < len; i++, src += cn )
            {
                T v = src[0];
                s0 += v; sq0 += (SQT)v*v;
            }
            sum[0] = s0;
            sqsum[0] = sq0;
        }
        else if( k == 2 )
        {
            ST s0 = sum[0], s1 = sum[1];
            SQT sq0 = sqsum[0], sq1 = sqsum[1];
            for( i = 0; i < len; i++, src += cn )
            {
                T v0 = src[0], v1 = src[1];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
            }
            sum[0] = s0; sum[1] = s1;
            sqsum[0] = sq0; sqsum[1] = sq1;
        }
        else if( k == 3 )
        {
            ST s0 = sum[0], s1 = sum[1], s2 = sum[2];
            SQT sq0 = sqsum[0], sq1 = sqsum[1], sq2 = sqsum[2];
            for( i = 0; i < len; i++, src += cn )
            {
                T v0 = src[0], v1 = src[1], v2 = src[2];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                s2 += v2; sq2 += (SQT)v2*v2;
            }
            sum[0] = s0; sum[1] = s1; sum[2] = s2;
            sqsum[0] = sq0; sqsum[1] = sq1; sqsum[2] = sq2;
        }

        for( ; k < cn; k += 4 )
        {
            src = src0 + k;
            ST s0 = sum[k], s1 = sum[k+1], s2 = sum[k+2], s3 = sum[k+3];
            SQT sq0 = sqsum[k], sq1 = sqsum[k+1], sq2 = sqsum[k+2], sq3 = sqsum[k+3];
            for( i = 0; i < len; i++, src += cn )
            {
                T v0, v1;
                v0 = src[0], v1 = src[1];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                v0 = src[2], v1 = src[3];
                s2 += v0; sq2 += (SQT)v0*v0;
                s3 += v1; sq3 += (SQT)v1*v1;
            }
            sum[k] = s0; sum[k+1] = s1;
            sum[k+2] = s2; sum[k+3] = s3;
            sqsum[k] = sq0; sqsum[k+1] = sq1;
            sqsum[k+2] = sq2; sqsum[k+3] = sq3;
        }
        return len;
    }

    int i, nzm = 0;

    if( cn == 1 )
    {
        ST s0 = sum[0];
        SQT sq0 = sqsum[0];
        for( i = 0; i < len; i++ )
            if( mask[i] )
            {
                T v = src[i];
                s0 += v; sq0 += (SQT)v*v;
                nzm++;
            }
        sum[0] = s0;
        sqsum[0] = sq0;
    }
    else if( cn == 3 )
    {
        ST s0 = sum[0], s1 = sum[1], s2 = sum[2];
        SQT sq0 = sqsum[0], sq1 = sqsum[1], sq2 = sqsum[2];
        for( i = 0; i < len; i++, src += 3 )
            if( mask[i] )
            {
                T v0 = src[0], v1 = src[1], v2 = src[2];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                s2 += v2; sq2 += (SQT)v2*v2;
                nzm++;
            }
        sum[0] = s0; sum[1] = s1; sum[2] = s2;
        sqsum[0] = sq0; sqsum[1] = sq1; sqsum[2] = sq2;
    }
    else
    {
        for( i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    T v = src[k];
                    ST s = sum[k] + v;
                    SQT sq = sqsum[k] + (SQT)v*v;
                    sum[k] = s; sqsum[k] = sq;
                }
                nzm++;
            }
    }
    return nzm;
}


static int sqsum8u( const uchar* src, const uchar* mask, int* sum, int* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum8s( const schar* src, const uchar* mask, int* sum, int* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum16u( const ushort* src, const uchar* mask, int* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum16s( const short* src, const uchar* mask, int* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum32s( const int* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum32f( const float* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum64f( const double* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

typedef int (*SumSqrFunc)(const uchar*, const uchar* mask, uchar*, uchar*, int, int);

static SumSqrFunc getSumSqrTab(int depth)
{
    static SumSqrFunc sumSqrTab[] =
    {
        (SumSqrFunc)GET_OPTIMIZED(sqsum8u), (SumSqrFunc)sqsum8s, (SumSqrFunc)sqsum16u, (SumSqrFunc)sqsum16s,
        (SumSqrFunc)sqsum32s, (SumSqrFunc)GET_OPTIMIZED(sqsum32f), (SumSqrFunc)sqsum64f, 0
    };

    return sumSqrTab[depth];
}

}

cv::Scalar cv::sum( InputArray _src )
{
    Mat src = _src.getMat();
    int k, cn = src.channels(), depth = src.depth();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src.total();
    int rows = src.size[0], cols = (int)(total_size/rows);
    if( src.dims == 2 || (src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        typedef IppStatus (CV_STDCALL* ippiSumFunc)(const void*, int, IppiSize, double *, int);
        ippiSumFunc ippFunc =
            type == CV_8UC1 ? (ippiSumFunc)ippiSum_8u_C1R :
            type == CV_8UC3 ? (ippiSumFunc)ippiSum_8u_C3R :
            type == CV_8UC4 ? (ippiSumFunc)ippiSum_8u_C4R :
            type == CV_16UC1 ? (ippiSumFunc)ippiSum_16u_C1R :
            type == CV_16UC3 ? (ippiSumFunc)ippiSum_16u_C3R :
            type == CV_16UC4 ? (ippiSumFunc)ippiSum_16u_C4R :
            type == CV_16SC1 ? (ippiSumFunc)ippiSum_16s_C1R :
            type == CV_16SC3 ? (ippiSumFunc)ippiSum_16s_C3R :
            type == CV_16SC4 ? (ippiSumFunc)ippiSum_16s_C4R :
            type == CV_32FC1 ? (ippiSumFunc)ippiSum_32f_C1R :
            type == CV_32FC3 ? (ippiSumFunc)ippiSum_32f_C3R :
            type == CV_32FC4 ? (ippiSumFunc)ippiSum_32f_C4R :
            0;
        if( ippFunc )
        {
            Ipp64f res[4];
            if( ippFunc(src.data, (int)src.step[0], sz, res, ippAlgHintAccurate) >= 0 )
            {
                Scalar sc;
                for( int i = 0; i < cn; i++ )
                {
                    sc[i] = res[i];
                }
                return sc;
            }
        }
    }
#endif

    SumFunc func = getSumFunc(depth);

    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    Scalar s;
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    size_t esz = 0;
    bool blockSum = depth < CV_32S;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf;

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], 0, (uchar*)buf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
        }
    }
    return s;
}

int cv::countNonZero( InputArray _src )
{
    Mat src = _src.getMat();
    CountNonZeroFunc func = getCountNonZeroTab(src.depth());

    CV_Assert( src.channels() == 1 && func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func( ptrs[0], total );

    return nz;
}

cv::Scalar cv::mean( InputArray _src, InputArray _mask )
{
    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_Assert( mask.empty() || mask.type() == CV_8U );

    int k, cn = src.channels(), depth = src.depth();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src.total();
    int rows = src.size[0], cols = (int)(total_size/rows);
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC1)(const void *, int, void *, int, IppiSize, Ipp64f *);
            ippiMaskMeanFuncC1 ippFuncC1 =
            type == CV_8UC1 ? (ippiMaskMeanFuncC1)ippiMean_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanFuncC1)ippiMean_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanFuncC1)ippiMean_32f_C1MR :
            0;
            if( ippFuncC1 )
            {
                Ipp64f res;
                if( ippFuncC1(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, &res) >= 0 )
                {
                    return Scalar(res);
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC3)(const void *, int, void *, int, IppiSize, int, Ipp64f *);
            ippiMaskMeanFuncC3 ippFuncC3 =
            type == CV_8UC3 ? (ippiMaskMeanFuncC3)ippiMean_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanFuncC3)ippiMean_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanFuncC3)ippiMean_32f_C3CMR :
            0;
            if( ippFuncC3 )
            {
                Ipp64f res1, res2, res3;
                if( ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 1, &res1) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 2, &res2) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 3, &res3) >= 0 )
                {
                    return Scalar(res1, res2, res3);
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanFunc)(const void*, int, IppiSize, double *, int);
            ippiMeanFunc ippFunc =
                type == CV_8UC1 ? (ippiMeanFunc)ippiMean_8u_C1R :
                type == CV_8UC3 ? (ippiMeanFunc)ippiMean_8u_C3R :
                type == CV_8UC4 ? (ippiMeanFunc)ippiMean_8u_C4R :
                type == CV_16UC1 ? (ippiMeanFunc)ippiMean_16u_C1R :
                type == CV_16UC3 ? (ippiMeanFunc)ippiMean_16u_C3R :
                type == CV_16UC4 ? (ippiMeanFunc)ippiMean_16u_C4R :
                type == CV_16SC1 ? (ippiMeanFunc)ippiMean_16s_C1R :
                type == CV_16SC3 ? (ippiMeanFunc)ippiMean_16s_C3R :
                type == CV_16SC4 ? (ippiMeanFunc)ippiMean_16s_C4R :
                type == CV_32FC1 ? (ippiMeanFunc)ippiMean_32f_C1R :
                type == CV_32FC3 ? (ippiMeanFunc)ippiMean_32f_C3R :
                type == CV_32FC4 ? (ippiMeanFunc)ippiMean_32f_C4R :
                0;
            if( ippFunc )
            {
                Ipp64f res[4];
                if( ippFunc(src.data, (int)src.step[0], sz, res, ippAlgHintAccurate) >= 0 )
                {
                    Scalar sc;
                    for( int i = 0; i < cn; i++ )
                    {
                        sc[i] = res[i];
                    }
                    return sc;
                }
            }
        }
    }
#endif

    SumFunc func = getSumFunc(depth);

    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    Scalar s;
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    bool blockSum = depth <= CV_16S;
    size_t esz = 0, nz0 = 0;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf;

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)buf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }
    return s*(nz0 ? 1./nz0 : 0);
}


void cv::meanStdDev( InputArray _src, OutputArray _mean, OutputArray _sdv, InputArray _mask )
{
    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_Assert( mask.empty() || mask.type() == CV_8U );

    int k, cn = src.channels(), depth = src.depth();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src.total();
    int rows = src.size[0], cols = (int)(total_size/rows);
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        Ipp64f mean_temp[3];
        Ipp64f stddev_temp[3];
        Ipp64f *pmean = &mean_temp[0];
        Ipp64f *pstddev = &stddev_temp[0];
        Mat mean, stddev;
        int dcn_mean = -1;
        if( _mean.needed() )
        {
            if( !_mean.fixedSize() )
                _mean.create(cn, 1, CV_64F, -1, true);
            mean = _mean.getMat();
            dcn_mean = (int)mean.total();
            pmean = (Ipp64f *)mean.data;
        }
        int dcn_stddev = -1;
        if( _sdv.needed() )
        {
            if( !_sdv.fixedSize() )
                _sdv.create(cn, 1, CV_64F, -1, true);
            stddev = _sdv.getMat();
            dcn_stddev = (int)stddev.total();
            pstddev = (Ipp64f *)stddev.data;
        }
        for( int c = cn; c < dcn_mean; c++ )
            pmean[c] = 0;
        for( int c = cn; c < dcn_stddev; c++ )
            pstddev[c] = 0;
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC1)(const void *, int, void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC1 ippFuncC1 =
            type == CV_8UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_32f_C1MR :
            0;
            if( ippFuncC1 )
            {
                if( ippFuncC1(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, pmean, pstddev) >= 0 )
                    return;
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC3)(const void *, int, void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC3 ippFuncC3 =
            type == CV_8UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CMR :
            0;
            if( ippFuncC3 )
            {
                if( ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                    return;
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC1)(const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC1 ippFuncC1 =
            type == CV_8UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_8u_C1R :
            type == CV_16UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_16u_C1R :
            //type == CV_32FC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_32f_C1R ://Aug 2013: bug in IPP 7.1, 8.0
            0;
            if( ippFuncC1 )
            {
                if( ippFuncC1(src.data, (int)src.step[0], sz, pmean, pstddev) >= 0 )
                    return;
            }
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC3)(const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC3 ippFuncC3 =
            type == CV_8UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CR :
            type == CV_16UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CR :
            type == CV_32FC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CR :
            0;
            if( ippFuncC3 )
            {
                if( ippFuncC3(src.data, (int)src.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                    return;
            }
        }
    }
#endif


    SumSqrFunc func = getSumSqrTab(depth);

    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0, nz0 = 0;
    AutoBuffer<double> _buf(cn*4);
    double *s = (double*)_buf, *sq = s + cn;
    int *sbuf = (int*)s, *sqbuf = (int*)sq;
    bool blockSum = depth <= CV_16S, blockSqSum = depth <= CV_8S;
    size_t esz = 0;

    for( k = 0; k < cn; k++ )
        s[k] = sq[k] = 0;

    if( blockSum )
    {
        intSumBlockSize = 1 << 15;
        blockSize = std::min(blockSize, intSumBlockSize);
        sbuf = (int*)(sq + cn);
        if( blockSqSum )
            sqbuf = sbuf + cn;
        for( k = 0; k < cn; k++ )
            sbuf[k] = sqbuf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)sbuf, (uchar*)sqbuf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += sbuf[k];
                    sbuf[k] = 0;
                }
                if( blockSqSum )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        sq[k] += sqbuf[k];
                        sqbuf[k] = 0;
                    }
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    double scale = nz0 ? 1./nz0 : 0.;
    for( k = 0; k < cn; k++ )
    {
        s[k] *= scale;
        sq[k] = std::sqrt(std::max(sq[k]*scale - s[k]*s[k], 0.));
    }

    for( j = 0; j < 2; j++ )
    {
        const double* sptr = j == 0 ? s : sq;
        _OutputArray _dst = j == 0 ? _mean : _sdv;
        if( !_dst.needed() )
            continue;

        if( !_dst.fixedSize() )
            _dst.create(cn, 1, CV_64F, -1, true);
        Mat dst = _dst.getMat();
        int dcn = (int)dst.total();
        CV_Assert( dst.type() == CV_64F && dst.isContinuous() &&
                   (dst.cols == 1 || dst.rows == 1) && dcn >= cn );
        double* dptr = dst.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
        for( ; k < dcn; k++ )
            dptr[k] = 0;
    }
}

/****************************************************************************************\
*                                       minMaxLoc                                        *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
minMaxIdx_( const T* src, const uchar* mask, WT* _minVal, WT* _maxVal,
            size_t* _minIdx, size_t* _maxIdx, int len, size_t startIdx )
{
    WT minVal = *_minVal, maxVal = *_maxVal;
    size_t minIdx = *_minIdx, maxIdx = *_maxIdx;

    if( !mask )
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }
    else
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( mask[i] && val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( mask[i] && val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }

    *_minIdx = minIdx;
    *_maxIdx = maxIdx;
    *_minVal = minVal;
    *_maxVal = maxVal;
}

static void minMaxIdx_8u(const uchar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_8s(const schar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_16u(const ushort* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_16s(const short* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_32s(const int* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_32f(const float* src, const uchar* mask, float* minval, float* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_64f(const double* src, const uchar* mask, double* minval, double* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

typedef void (*MinMaxIdxFunc)(const uchar*, const uchar*, int*, int*, size_t*, size_t*, int, size_t);

static MinMaxIdxFunc getMinmaxTab(int depth)
{
    static MinMaxIdxFunc minmaxTab[] =
    {
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32f), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_64f),
        0
    };

    return minmaxTab[depth];
}

static void ofs2idx(const Mat& a, size_t ofs, int* idx)
{
    int i, d = a.dims;
    if( ofs > 0 )
    {
        ofs--;
        for( i = d-1; i >= 0; i-- )
        {
            int sz = a.size[i];
            idx[i] = (int)(ofs % sz);
            ofs /= sz;
        }
    }
    else
    {
        for( i = d-1; i >= 0; i-- )
            idx[i] = -1;
    }
}

}

void cv::minMaxIdx(InputArray _src, double* minVal,
                   double* maxVal, int* minIdx, int* maxIdx,
                   InputArray _mask)
{
    Mat src = _src.getMat(), mask = _mask.getMat();
    int depth = src.depth(), cn = src.channels();

    CV_Assert( (cn == 1 && (mask.empty() || mask.type() == CV_8U)) ||
               (cn >= 1 && mask.empty() && !minIdx && !maxIdx) );

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src.total();
    int rows = src.size[0], cols = (int)(total_size/rows);
    if( cn == 1 && ( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) ) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMinMaxIndxFuncC1)(const void *, int, const void *, int, IppiSize, Ipp32f *, Ipp32f *, IppiPoint *, IppiPoint *);
            ippiMaskMinMaxIndxFuncC1 ippFuncC1 =
            type == CV_8UC1 ? (ippiMaskMinMaxIndxFuncC1)ippiMinMaxIndx_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMinMaxIndxFuncC1)ippiMinMaxIndx_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMinMaxIndxFuncC1)ippiMinMaxIndx_32f_C1MR :
            0;
            if( ippFuncC1 )
            {
                Ipp32f min, max;
                IppiPoint minp, maxp;
                if( ippFuncC1(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, &min, &max, &minp, &maxp) >= 0 )
                {
                    if( minVal )
                        *minVal = (double)min;
                    if( maxVal )
                        *maxVal = (double)max;
                    if( !minp.x && !minp.y && !maxp.x && !maxp.y && !mask.data[0] )
                        minp.x = maxp.x = -1;
                    if( minIdx )
                    {
                        size_t minidx = minp.y * cols + minp.x + 1;
                        ofs2idx(src, minidx, minIdx);
                    }
                    if( maxIdx )
                    {
                        size_t maxidx = maxp.y * cols + maxp.x + 1;
                        ofs2idx(src, maxidx, maxIdx);
                    }
                    return;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMinMaxIndxFuncC1)(const void *, int, IppiSize, Ipp32f *, Ipp32f *, IppiPoint *, IppiPoint *);
            ippiMinMaxIndxFuncC1 ippFuncC1 =
                type == CV_8UC1 ? (ippiMinMaxIndxFuncC1)ippiMinMaxIndx_8u_C1R :
                type == CV_16UC1 ? (ippiMinMaxIndxFuncC1)ippiMinMaxIndx_16u_C1R :
                type == CV_32FC1 ? (ippiMinMaxIndxFuncC1)ippiMinMaxIndx_32f_C1R :
                0;
            if( ippFuncC1 )
            {
                Ipp32f min, max;
                IppiPoint minp, maxp;
                if( ippFuncC1(src.data, (int)src.step[0], sz, &min, &max, &minp, &maxp) >= 0 )
                {
                    if( minVal )
                        *minVal = (double)min;
                    if( maxVal )
                        *maxVal = (double)max;
                    if( minIdx )
                    {
                        size_t minidx = minp.y * cols + minp.x + 1;
                        ofs2idx(src, minidx, minIdx);
                    }
                    if( maxIdx )
                    {
                        size_t maxidx = maxp.y * cols + maxp.x + 1;
                        ofs2idx(src, maxidx, maxIdx);
                    }
                    return;
                }
            }
        }
    }
#endif

    MinMaxIdxFunc func = getMinmaxTab(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);

    size_t minidx = 0, maxidx = 0;
    int iminval = INT_MAX, imaxval = INT_MIN;
    float  fminval = std::numeric_limits<float>::infinity(),  fmaxval = -fminval;
    double dminval = std::numeric_limits<double>::infinity(), dmaxval = -dminval;
    size_t startidx = 1;
    int *minval = &iminval, *maxval = &imaxval;
    int planeSize = (int)it.size*cn;

    if( depth == CV_32F )
        minval = (int*)&fminval, maxval = (int*)&fmaxval;
    else if( depth == CV_64F )
        minval = (int*)&dminval, maxval = (int*)&dmaxval;

    for( size_t i = 0; i < it.nplanes; i++, ++it, startidx += planeSize )
        func( ptrs[0], ptrs[1], minval, maxval, &minidx, &maxidx, planeSize, startidx );

    if (!src.empty() && mask.empty())
    {
        if( minidx == 0 )
             minidx = 1;
         if( maxidx == 0 )
             maxidx = 1;
    }

    if( minidx == 0 )
        dminval = dmaxval = 0;
    else if( depth == CV_32F )
        dminval = fminval, dmaxval = fmaxval;
    else if( depth <= CV_32S )
        dminval = iminval, dmaxval = imaxval;

    if( minVal )
        *minVal = dminval;
    if( maxVal )
        *maxVal = dmaxval;

    if( minIdx )
        ofs2idx(src, minidx, minIdx);
    if( maxIdx )
        ofs2idx(src, maxidx, maxIdx);
}

void cv::minMaxLoc( InputArray _img, double* minVal, double* maxVal,
                    Point* minLoc, Point* maxLoc, InputArray mask )
{
    Mat img = _img.getMat();
    CV_Assert(img.dims <= 2);

    minMaxIdx(_img, minVal, maxVal, (int*)minLoc, (int*)maxLoc, mask);
    if( minLoc )
        std::swap(minLoc->x, minLoc->y);
    if( maxLoc )
        std::swap(maxLoc->x, maxLoc->y);
}

/****************************************************************************************\
*                                         norm                                           *
\****************************************************************************************/

namespace cv
{

float normL2Sqr_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    if( USE_SSE2 )
    {
        float CV_DECL_ALIGNED(16) buf[4];
        __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

        for( ; j <= n - 8; j += 8 )
        {
            __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
            __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
            d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
            d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
        }
        _mm_store_ps(buf, _mm_add_ps(d0, d1));
        d = buf[0] + buf[1] + buf[2] + buf[3];
    }
    else
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
            d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        }
    }

    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}


float normL1_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    if( USE_SSE2 )
    {
        float CV_DECL_ALIGNED(16) buf[4];
        static const int CV_DECL_ALIGNED(16) absbuf[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();
        __m128 absmask = _mm_load_ps((const float*)absbuf);

        for( ; j <= n - 8; j += 8 )
        {
            __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
            __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
            d0 = _mm_add_ps(d0, _mm_and_ps(t0, absmask));
            d1 = _mm_add_ps(d1, _mm_and_ps(t1, absmask));
        }
        _mm_store_ps(buf, _mm_add_ps(d0, d1));
        d = buf[0] + buf[1] + buf[2] + buf[3];
    }
    else
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            d += std::abs(a[j] - b[j]) + std::abs(a[j+1] - b[j+1]) +
                    std::abs(a[j+2] - b[j+2]) + std::abs(a[j+3] - b[j+3]);
        }
    }

    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

int normL1_(const uchar* a, const uchar* b, int n)
{
    int j = 0, d = 0;
#if CV_SSE
    if( USE_SSE2 )
    {
        __m128i d0 = _mm_setzero_si128();

        for( ; j <= n - 16; j += 16 )
        {
            __m128i t0 = _mm_loadu_si128((const __m128i*)(a + j));
            __m128i t1 = _mm_loadu_si128((const __m128i*)(b + j));

            d0 = _mm_add_epi32(d0, _mm_sad_epu8(t0, t1));
        }

        for( ; j <= n - 4; j += 4 )
        {
            __m128i t0 = _mm_cvtsi32_si128(*(const int*)(a + j));
            __m128i t1 = _mm_cvtsi32_si128(*(const int*)(b + j));

            d0 = _mm_add_epi32(d0, _mm_sad_epu8(t0, t1));
        }
        d = _mm_cvtsi128_si32(_mm_add_epi32(d0, _mm_unpackhi_epi64(d0, d0)));
    }
    else
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            d += std::abs(a[j] - b[j]) + std::abs(a[j+1] - b[j+1]) +
                    std::abs(a[j+2] - b[j+2]) + std::abs(a[j+3] - b[j+3]);
        }
    }
    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

static const uchar popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

static const uchar popCountTable2[] =
{
    0, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4
};

static const uchar popCountTable4[] =
{
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};

static int normHamming(const uchar* a, int n)
{
    int i = 0, result = 0;
#if CV_NEON
    uint32x4_t bits = vmovq_n_u32(0);
    for (; i <= n - 16; i += 16) {
        uint8x16_t A_vec = vld1q_u8 (a + i);
        uint8x16_t bitsSet = vcntq_u8 (A_vec);
        uint16x8_t bitSet8 = vpaddlq_u8 (bitsSet);
        uint32x4_t bitSet4 = vpaddlq_u16 (bitSet8);
        bits = vaddq_u32(bits, bitSet4);
    }
    uint64x2_t bitSet2 = vpaddlq_u32 (bits);
    result = vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),0);
    result += vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),2);
#else
    for( ; i <= n - 4; i += 4 )
            result += popCountTable[a[i]] + popCountTable[a[i+1]] +
            popCountTable[a[i+2]] + popCountTable[a[i+3]];
#endif
    for( ; i < n; i++ )
        result += popCountTable[a[i]];
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n)
{
    int i = 0, result = 0;
#if CV_NEON
    uint32x4_t bits = vmovq_n_u32(0);
    for (; i <= n - 16; i += 16) {
        uint8x16_t A_vec = vld1q_u8 (a + i);
        uint8x16_t B_vec = vld1q_u8 (b + i);
        uint8x16_t AxorB = veorq_u8 (A_vec, B_vec);
        uint8x16_t bitsSet = vcntq_u8 (AxorB);
        uint16x8_t bitSet8 = vpaddlq_u8 (bitsSet);
        uint32x4_t bitSet4 = vpaddlq_u16 (bitSet8);
        bits = vaddq_u32(bits, bitSet4);
    }
    uint64x2_t bitSet2 = vpaddlq_u32 (bits);
    result = vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),0);
    result += vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),2);
#else
    for( ; i <= n - 4; i += 4 )
        result += popCountTable[a[i] ^ b[i]] + popCountTable[a[i+1] ^ b[i+1]] +
                popCountTable[a[i+2] ^ b[i+2]] + popCountTable[a[i+3] ^ b[i+3]];
#endif
    for( ; i < n; i++ )
        result += popCountTable[a[i] ^ b[i]];
    return result;
}

static int normHamming(const uchar* a, int n, int cellSize)
{
    if( cellSize == 1 )
        return normHamming(a, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        CV_Error( CV_StsBadSize, "bad cell size (not 1, 2 or 4) in normHamming" );
    int i = 0, result = 0;
#if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i]] + tab[a[i+1]] + tab[a[i+2]] + tab[a[i+3]];
#endif
    for( ; i < n; i++ )
        result += tab[a[i]];
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n, int cellSize)
{
    if( cellSize == 1 )
        return normHamming(a, b, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        CV_Error( CV_StsBadSize, "bad cell size (not 1, 2 or 4) in normHamming" );
    int i = 0, result = 0;
    #if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i] ^ b[i]] + tab[a[i+1] ^ b[i+1]] +
                tab[a[i+2] ^ b[i+2]] + tab[a[i+3] ^ b[i+3]];
    #endif
    for( ; i < n; i++ )
        result += tab[a[i] ^ b[i]];
    return result;
}


template<typename T, typename ST> int
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, ST(fast_abs(src[k])));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL1_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += fast_abs(src[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL2_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    T v = src[k];
                    result += (ST)v*v;
                }
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffInf_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src1, src2, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, (ST)std::abs(src1[k] - src2[k]));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL1_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += std::abs(src1[k] - src2[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL2_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    ST v = src1[k] - src2[k];
                    result += v*v;
                }
            }
    }
    *_result = result;
    return 0;
}


#define CV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
{ return norm##L##_(src, mask, r, len, cn); } \
    static int normDiff##L##_##suffix(const type* src1, const type* src2, \
    const uchar* mask, ntype* r, int len, int cn) \
{ return normDiff##L##_(src1, src2, mask, r, (int)len, cn); }

#define CV_DEF_NORM_ALL(suffix, type, inftype, l1type, l2type) \
    CV_DEF_NORM_FUNC(Inf, suffix, type, inftype) \
    CV_DEF_NORM_FUNC(L1, suffix, type, l1type) \
    CV_DEF_NORM_FUNC(L2, suffix, type, l2type)

CV_DEF_NORM_ALL(8u, uchar, int, int, int)
CV_DEF_NORM_ALL(8s, schar, int, int, int)
CV_DEF_NORM_ALL(16u, ushort, int, int, double)
CV_DEF_NORM_ALL(16s, short, int, int, double)
CV_DEF_NORM_ALL(32s, int, int, double, double)
CV_DEF_NORM_ALL(32f, float, float, double, double)
CV_DEF_NORM_ALL(64f, double, double, double, double)


typedef int (*NormFunc)(const uchar*, const uchar*, uchar*, int, int);
typedef int (*NormDiffFunc)(const uchar*, const uchar*, const uchar*, uchar*, int, int);

static NormFunc getNormFunc(int normType, int depth)
{
    static NormFunc normTab[3][8] =
    {
        {
            (NormFunc)GET_OPTIMIZED(normInf_8u), (NormFunc)GET_OPTIMIZED(normInf_8s), (NormFunc)GET_OPTIMIZED(normInf_16u), (NormFunc)GET_OPTIMIZED(normInf_16s),
            (NormFunc)GET_OPTIMIZED(normInf_32s), (NormFunc)GET_OPTIMIZED(normInf_32f), (NormFunc)normInf_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL1_8u), (NormFunc)GET_OPTIMIZED(normL1_8s), (NormFunc)GET_OPTIMIZED(normL1_16u), (NormFunc)GET_OPTIMIZED(normL1_16s),
            (NormFunc)GET_OPTIMIZED(normL1_32s), (NormFunc)GET_OPTIMIZED(normL1_32f), (NormFunc)normL1_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL2_8u), (NormFunc)GET_OPTIMIZED(normL2_8s), (NormFunc)GET_OPTIMIZED(normL2_16u), (NormFunc)GET_OPTIMIZED(normL2_16s),
            (NormFunc)GET_OPTIMIZED(normL2_32s), (NormFunc)GET_OPTIMIZED(normL2_32f), (NormFunc)normL2_64f, 0
        }
    };

    return normTab[normType][depth];
}

static NormDiffFunc getNormDiffFunc(int normType, int depth)
{
    static NormDiffFunc normDiffTab[3][8] =
    {
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffInf_8u), (NormDiffFunc)normDiffInf_8s,
            (NormDiffFunc)normDiffInf_16u, (NormDiffFunc)normDiffInf_16s,
            (NormDiffFunc)normDiffInf_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffInf_32f),
            (NormDiffFunc)normDiffInf_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL1_8u), (NormDiffFunc)normDiffL1_8s,
            (NormDiffFunc)normDiffL1_16u, (NormDiffFunc)normDiffL1_16s,
            (NormDiffFunc)normDiffL1_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL1_32f),
            (NormDiffFunc)normDiffL1_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL2_8u), (NormDiffFunc)normDiffL2_8s,
            (NormDiffFunc)normDiffL2_16u, (NormDiffFunc)normDiffL2_16s,
            (NormDiffFunc)normDiffL2_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL2_32f),
            (NormDiffFunc)normDiffL2_64f, 0
        }
    };

    return normDiffTab[normType][depth];
}

}

double cv::norm( InputArray _src, int normType, InputArray _mask )
{
    Mat src = _src.getMat(), mask = _mask.getMat();
    int depth = src.depth(), cn = src.channels();

    normType &= 7;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
               ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && src.type() == CV_8U) );

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src.total();
    int rows = src.size[0], cols = (int)(total_size/rows);
    if( (src.dims == 2 || (src.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size
        && (normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormFuncC1 ippFuncC1 =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_32f_C1MR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_32f_C1MR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_32f_C1MR :
                0) : 0;
            if( ippFuncC1 )
            {
                Ipp64f norm;
                if( ippFuncC1(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, &norm) >= 0 )
                {
                    return normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormFuncC3 ippFuncC3 =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_32f_C3CMR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_32f_C3CMR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_32f_C3CMR :
                0) : 0;
            if( ippFuncC3 )
            {
                Ipp64f norm1, norm2, norm3;
                if( ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    ippFuncC3(src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    return normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiNormFunc)(const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            ippiNormFunc ippFunc =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiNormFunc)ippiNorm_Inf_8u_C1R :
                type == CV_8UC3 ? (ippiNormFunc)ippiNorm_Inf_8u_C3R :
                type == CV_8UC4 ? (ippiNormFunc)ippiNorm_Inf_8u_C4R :
                type == CV_16UC1 ? (ippiNormFunc)ippiNorm_Inf_16u_C1R :
                type == CV_16UC3 ? (ippiNormFunc)ippiNorm_Inf_16u_C3R :
                type == CV_16UC4 ? (ippiNormFunc)ippiNorm_Inf_16u_C4R :
                type == CV_16SC1 ? (ippiNormFunc)ippiNorm_Inf_16s_C1R :
                //type == CV_16SC3 ? (ippiNormFunc)ippiNorm_Inf_16s_C3R : //Aug 2013: problem in IPP 7.1, 8.0 : -32768
                //type == CV_16SC4 ? (ippiNormFunc)ippiNorm_Inf_16s_C4R : //Aug 2013: problem in IPP 7.1, 8.0 : -32768
                type == CV_32FC1 ? (ippiNormFunc)ippiNorm_Inf_32f_C1R :
                type == CV_32FC3 ? (ippiNormFunc)ippiNorm_Inf_32f_C3R :
                type == CV_32FC4 ? (ippiNormFunc)ippiNorm_Inf_32f_C4R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiNormFunc)ippiNorm_L1_8u_C1R :
                type == CV_8UC3 ? (ippiNormFunc)ippiNorm_L1_8u_C3R :
                type == CV_8UC4 ? (ippiNormFunc)ippiNorm_L1_8u_C4R :
                type == CV_16UC1 ? (ippiNormFunc)ippiNorm_L1_16u_C1R :
                type == CV_16UC3 ? (ippiNormFunc)ippiNorm_L1_16u_C3R :
                type == CV_16UC4 ? (ippiNormFunc)ippiNorm_L1_16u_C4R :
                type == CV_16SC1 ? (ippiNormFunc)ippiNorm_L1_16s_C1R :
                type == CV_16SC3 ? (ippiNormFunc)ippiNorm_L1_16s_C3R :
                type == CV_16SC4 ? (ippiNormFunc)ippiNorm_L1_16s_C4R :
                type == CV_32FC1 ? (ippiNormFunc)ippiNorm_L1_32f_C1R :
                type == CV_32FC3 ? (ippiNormFunc)ippiNorm_L1_32f_C3R :
                type == CV_32FC4 ? (ippiNormFunc)ippiNorm_L1_32f_C4R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiNormFunc)ippiNorm_L2_8u_C1R :
                type == CV_8UC3 ? (ippiNormFunc)ippiNorm_L2_8u_C3R :
                type == CV_8UC4 ? (ippiNormFunc)ippiNorm_L2_8u_C4R :
                type == CV_16UC1 ? (ippiNormFunc)ippiNorm_L2_16u_C1R :
                type == CV_16UC3 ? (ippiNormFunc)ippiNorm_L2_16u_C3R :
                type == CV_16UC4 ? (ippiNormFunc)ippiNorm_L2_16u_C4R :
                type == CV_16SC1 ? (ippiNormFunc)ippiNorm_L2_16s_C1R :
                type == CV_16SC3 ? (ippiNormFunc)ippiNorm_L2_16s_C3R :
                type == CV_16SC4 ? (ippiNormFunc)ippiNorm_L2_16s_C4R :
                type == CV_32FC1 ? (ippiNormFunc)ippiNorm_L2_32f_C1R :
                type == CV_32FC3 ? (ippiNormFunc)ippiNorm_L2_32f_C3R :
                type == CV_32FC4 ? (ippiNormFunc)ippiNorm_L2_32f_C4R :
                0) : 0;
            if( ippFunc )
            {
                Ipp64f norm_array[4];
                if( ippFunc(src.data, (int)src.step[0], sz, norm_array, ippAlgHintAccurate) >= 0 )
                {
                    Ipp64f norm = (normType == NORM_L2 || normType == NORM_L2SQR) ? norm_array[0] * norm_array[0] : norm_array[0];
                    for( int i = 1; i < cn; i++ )
                    {
                        norm =
                            normType == NORM_INF ? std::max(norm, norm_array[i]) :
                            normType == NORM_L1 ? norm + norm_array[i] :
                            normType == NORM_L2 || normType == NORM_L2SQR ? norm + norm_array[i] * norm_array[i] :
                            0;
                    }
                    return normType == NORM_L2 ? (double)std::sqrt(norm) : (double)norm;
                }
            }
        }
    }
#endif

    if( src.isContinuous() && mask.empty() )
    {
        size_t len = src.total()*cn;
        if( len == (size_t)(int)len )
        {
            if( depth == CV_32F )
            {
                const float* data = src.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL1_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normInf_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
            }
            if( depth == CV_8U )
            {
                const uchar* data = src.ptr<uchar>();

                if( normType == NORM_HAMMING )
                    return normHamming(data, (int)len);

                if( normType == NORM_HAMMING2 )
                    return normHamming(data, (int)len, 2);
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_and(src, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src, 0};
        uchar* ptrs[1];
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            result += normHamming(ptrs[0], total, cellSize);

        return result;
    }

    NormFunc func = getNormFunc(normType >> 1, depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    union
    {
        double d;
        int i;
        float f;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)it.size, blockSize = total, intSumBlockSize = 0, count = 0;
    bool blockSum = (normType == NORM_L1 && depth <= CV_16S) ||
            ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S);
    int isum = 0;
    int *ibuf = &result.i;
    size_t esz = 0;

    if( blockSum )
    {
        intSumBlockSize = (normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15))/cn;
        blockSize = std::min(blockSize, intSumBlockSize);
        ibuf = &isum;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], ptrs[1], (uchar*)ibuf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                result.d += isum;
                isum = 0;
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    if( normType == NORM_INF )
    {
        if( depth == CV_64F )
            ;
        else if( depth == CV_32F )
            result.d = result.f;
        else
            result.d = result.i;
    }
    else if( normType == NORM_L2 )
        result.d = std::sqrt(result.d);

    return result.d;
}


double cv::norm( InputArray _src1, InputArray _src2, int normType, InputArray _mask )
{
    if( normType & CV_RELATIVE )
    {
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
        Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();

        CV_Assert( src1.size == src2.size && src1.type() == src2.type() );

        normType &= 7;
        CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR ||
                ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && src1.type() == CV_8U) );
        size_t total_size = src1.total();
        int rows = src1.size[0], cols = (int)(total_size/rows);
        if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
            && cols > 0 && (size_t)rows*cols == total_size
            && (normType == NORM_INF || normType == NORM_L1 ||
                normType == NORM_L2 || normType == NORM_L2SQR) )
        {
            IppiSize sz = { cols, rows };
            int type = src1.type();
            if( !mask.empty() )
            {
                typedef IppStatus (CV_STDCALL* ippiMaskNormRelFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
                ippiMaskNormRelFuncC1 ippFuncC1 =
                    normType == NORM_INF ?
                    (type == CV_8UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_Inf_8u_C1MR :
                    type == CV_8SC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_Inf_8s_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_Inf_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_Inf_32f_C1MR :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L1_8u_C1MR :
                    type == CV_8SC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L1_8s_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L1_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L1_32f_C1MR :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L2_8u_C1MR :
                    type == CV_8SC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L2_8s_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L2_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormRelFuncC1)ippiNormRel_L2_32f_C1MR :
                    0) : 0;
                if( ippFuncC1 )
                {
                    Ipp64f norm;
                    if( ippFuncC1(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, &norm) >= 0 )
                        return normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm;
                }
            }
            else
            {
                typedef IppStatus (CV_STDCALL* ippiNormRelFunc)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
                ippiNormRelFunc ippFunc =
                    normType == NORM_INF ?
                    (type == CV_8UC1 ? (ippiNormRelFunc)ippiNormRel_Inf_8u_C1R :
                    type == CV_16UC1 ? (ippiNormRelFunc)ippiNormRel_Inf_16u_C1R :
                    type == CV_16SC1 ? (ippiNormRelFunc)ippiNormRel_Inf_16s_C1R :
                    type == CV_32FC1 ? (ippiNormRelFunc)ippiNormRel_Inf_32f_C1R :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8UC1 ? (ippiNormRelFunc)ippiNormRel_L1_8u_C1R :
                    type == CV_16UC1 ? (ippiNormRelFunc)ippiNormRel_L1_16u_C1R :
                    type == CV_16SC1 ? (ippiNormRelFunc)ippiNormRel_L1_16s_C1R :
                    type == CV_32FC1 ? (ippiNormRelFunc)ippiNormRel_L1_32f_C1R :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8UC1 ? (ippiNormRelFunc)ippiNormRel_L2_8u_C1R :
                    type == CV_16UC1 ? (ippiNormRelFunc)ippiNormRel_L2_16u_C1R :
                    type == CV_16SC1 ? (ippiNormRelFunc)ippiNormRel_L2_16s_C1R :
                    type == CV_32FC1 ? (ippiNormRelFunc)ippiNormRel_L2_32f_C1R :
                    0) : 0;
                if( ippFunc )
                {
                    Ipp64f norm;
                    if( ippFunc(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], sz, &norm, ippAlgHintAccurate) >= 0 )
                        return (double)norm;
                }
            }
        }
#endif
        return norm(_src1, _src2, normType & ~CV_RELATIVE, _mask)/(norm(_src2, normType, _mask) + DBL_EPSILON);
    }

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int depth = src1.depth(), cn = src1.channels();

    CV_Assert( src1.size == src2.size && src1.type() == src2.type() );

    normType &= 7;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
              ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && src1.type() == CV_8U) );

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    size_t total_size = src1.total();
    int rows = src1.size[0], cols = (int)(total_size/rows);
    if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size
        && (normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR) )
    {
        IppiSize sz = { cols, rows };
        int type = src1.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormDiffFuncC1 ippFuncC1 =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_32f_C1MR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_32f_C1MR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_8u_C1MR :
                type == CV_8SC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_8s_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_32f_C1MR :
                0) : 0;
            if( ippFuncC1 )
            {
                Ipp64f norm;
                if( ippFuncC1(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, &norm) >= 0 )
                    return normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm;
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC3)(const void *, int, const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormDiffFuncC3 ippFuncC3 =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_32f_C3CMR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_32f_C3CMR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_8u_C3CMR :
                type == CV_8SC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_8s_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_32f_C3CMR :
                0) : 0;
            if( ippFuncC3 )
            {
                Ipp64f norm1, norm2, norm3;
                if( ippFuncC3(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    ippFuncC3(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    ippFuncC3(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    return normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiNormDiffFunc)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            ippiNormDiffFunc ippFunc =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiNormDiffFunc)ippiNormDiff_Inf_8u_C1R :
                type == CV_8UC3 ? (ippiNormDiffFunc)ippiNormDiff_Inf_8u_C3R :
                type == CV_8UC4 ? (ippiNormDiffFunc)ippiNormDiff_Inf_8u_C4R :
                type == CV_16UC1 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16u_C1R :
                type == CV_16UC3 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16u_C3R :
                type == CV_16UC4 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16u_C4R :
                type == CV_16SC1 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16s_C1R :
                //type == CV_16SC3 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16s_C3R : //Aug 2013: problem in IPP 7.1, 8.0 : -32768
                //type == CV_16SC4 ? (ippiNormDiffFunc)ippiNormDiff_Inf_16s_C4R : //Aug 2013: problem in IPP 7.1, 8.0 : -32768
                type == CV_32FC1 ? (ippiNormDiffFunc)ippiNormDiff_Inf_32f_C1R :
                type == CV_32FC3 ? (ippiNormDiffFunc)ippiNormDiff_Inf_32f_C3R :
                type == CV_32FC4 ? (ippiNormDiffFunc)ippiNormDiff_Inf_32f_C4R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiNormDiffFunc)ippiNormDiff_L1_8u_C1R :
                type == CV_8UC3 ? (ippiNormDiffFunc)ippiNormDiff_L1_8u_C3R :
                type == CV_8UC4 ? (ippiNormDiffFunc)ippiNormDiff_L1_8u_C4R :
                type == CV_16UC1 ? (ippiNormDiffFunc)ippiNormDiff_L1_16u_C1R :
                type == CV_16UC3 ? (ippiNormDiffFunc)ippiNormDiff_L1_16u_C3R :
                type == CV_16UC4 ? (ippiNormDiffFunc)ippiNormDiff_L1_16u_C4R :
                type == CV_16SC1 ? (ippiNormDiffFunc)ippiNormDiff_L1_16s_C1R :
                type == CV_16SC3 ? (ippiNormDiffFunc)ippiNormDiff_L1_16s_C3R :
                type == CV_16SC4 ? (ippiNormDiffFunc)ippiNormDiff_L1_16s_C4R :
                type == CV_32FC1 ? (ippiNormDiffFunc)ippiNormDiff_L1_32f_C1R :
                type == CV_32FC3 ? (ippiNormDiffFunc)ippiNormDiff_L1_32f_C3R :
                type == CV_32FC4 ? (ippiNormDiffFunc)ippiNormDiff_L1_32f_C4R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiNormDiffFunc)ippiNormDiff_L2_8u_C1R :
                type == CV_8UC3 ? (ippiNormDiffFunc)ippiNormDiff_L2_8u_C3R :
                type == CV_8UC4 ? (ippiNormDiffFunc)ippiNormDiff_L2_8u_C4R :
                type == CV_16UC1 ? (ippiNormDiffFunc)ippiNormDiff_L2_16u_C1R :
                type == CV_16UC3 ? (ippiNormDiffFunc)ippiNormDiff_L2_16u_C3R :
                type == CV_16UC4 ? (ippiNormDiffFunc)ippiNormDiff_L2_16u_C4R :
                type == CV_16SC1 ? (ippiNormDiffFunc)ippiNormDiff_L2_16s_C1R :
                type == CV_16SC3 ? (ippiNormDiffFunc)ippiNormDiff_L2_16s_C3R :
                type == CV_16SC4 ? (ippiNormDiffFunc)ippiNormDiff_L2_16s_C4R :
                type == CV_32FC1 ? (ippiNormDiffFunc)ippiNormDiff_L2_32f_C1R :
                type == CV_32FC3 ? (ippiNormDiffFunc)ippiNormDiff_L2_32f_C3R :
                type == CV_32FC4 ? (ippiNormDiffFunc)ippiNormDiff_L2_32f_C4R :
                0) : 0;
            if( ippFunc )
            {
                Ipp64f norm_array[4];
                if( ippFunc(src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], sz, norm_array, ippAlgHintAccurate) >= 0 )
                {
                    Ipp64f norm = (normType == NORM_L2 || normType == NORM_L2SQR) ? norm_array[0] * norm_array[0] : norm_array[0];
                    for( int i = 1; i < src1.channels(); i++ )
                    {
                        norm =
                            normType == NORM_INF ? std::max(norm, norm_array[i]) :
                            normType == NORM_L1 ? norm + norm_array[i] :
                            normType == NORM_L2 || normType == NORM_L2SQR ? norm + norm_array[i] * norm_array[i] :
                            0;
                    }
                    return normType == NORM_L2 ? (double)std::sqrt(norm) : (double)norm;
                }
            }
        }
    }
#endif

    if( src1.isContinuous() && src2.isContinuous() && mask.empty() )
    {
        size_t len = src1.total()*src1.channels();
        if( len == (size_t)(int)len )
        {
            if( src1.depth() == CV_32F )
            {
                const float* data1 = src1.ptr<float>();
                const float* data2 = src2.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL1_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normDiffInf_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_xor(src1, src2, temp);
            bitwise_and(temp, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src1, &src2, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            result += normHamming(ptrs[0], ptrs[1], total, cellSize);

        return result;
    }

    NormDiffFunc func = getNormDiffFunc(normType >> 1, depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src1, &src2, &mask, 0};
    uchar* ptrs[3];
    union
    {
        double d;
        float f;
        int i;
        unsigned u;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)it.size, blockSize = total, intSumBlockSize = 0, count = 0;
    bool blockSum = (normType == NORM_L1 && depth <= CV_16S) ||
            ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S);
    unsigned isum = 0;
    unsigned *ibuf = &result.u;
    size_t esz = 0;

    if( blockSum )
    {
        intSumBlockSize = normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        ibuf = &isum;
        esz = src1.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], ptrs[1], ptrs[2], (uchar*)ibuf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                result.d += isum;
                isum = 0;
                count = 0;
            }
            ptrs[0] += bsz*esz;
            ptrs[1] += bsz*esz;
            if( ptrs[2] )
                ptrs[2] += bsz;
        }
    }

    if( normType == NORM_INF )
    {
        if( depth == CV_64F )
            ;
        else if( depth == CV_32F )
            result.d = result.f;
        else
            result.d = result.u;
    }
    else if( normType == NORM_L2 )
        result.d = std::sqrt(result.d);

    return result.d;
}


///////////////////////////////////// batch distance ///////////////////////////////////////

namespace cv
{

template<typename _Tp, typename _Rt>
void batchDistL1_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL1<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL1<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2Sqr_(const _Tp* src1, const _Tp* src2, size_t step2,
                     int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len));
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len)) : val0;
    }
}

static void batchDistHamming(const uchar* src1, const uchar* src2, size_t step2,
                             int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normHamming(src1, src2 + step2*i, len);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normHamming(src1, src2 + step2*i, len) : val0;
    }
}

static void batchDistHamming2(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normHamming(src1, src2 + step2*i, len, 2);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normHamming(src1, src2 + step2*i, len, 2) : val0;
    }
}

static void batchDistL1_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL1_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_32f(const float* src1, const float* src2, size_t step2,
                                int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

typedef void (*BatchDistFunc)(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, uchar* dist, const uchar* mask);


struct BatchDistInvoker : public ParallelLoopBody
{
    BatchDistInvoker( const Mat& _src1, const Mat& _src2,
                      Mat& _dist, Mat& _nidx, int _K,
                      const Mat& _mask, int _update,
                      BatchDistFunc _func)
    {
        src1 = &_src1;
        src2 = &_src2;
        dist = &_dist;
        nidx = &_nidx;
        K = _K;
        mask = &_mask;
        update = _update;
        func = _func;
    }

    void operator()(const Range& range) const
    {
        AutoBuffer<int> buf(src2->rows);
        int* bufptr = buf;

        for( int i = range.start; i < range.end; i++ )
        {
            func(src1->ptr(i), src2->ptr(), src2->step, src2->rows, src2->cols,
                 K > 0 ? (uchar*)bufptr : dist->ptr(i), mask->data ? mask->ptr(i) : 0);

            if( K > 0 )
            {
                int* nidxptr = nidx->ptr<int>(i);
                // since positive float's can be compared just like int's,
                // we handle both CV_32S and CV_32F cases with a single branch
                int* distptr = (int*)dist->ptr(i);

                int j, k;

                for( j = 0; j < src2->rows; j++ )
                {
                    int d = bufptr[j];
                    if( d < distptr[K-1] )
                    {
                        for( k = K-2; k >= 0 && distptr[k] > d; k-- )
                        {
                            nidxptr[k+1] = nidxptr[k];
                            distptr[k+1] = distptr[k];
                        }
                        nidxptr[k+1] = j + update;
                        distptr[k+1] = d;
                    }
                }
            }
        }
    }

    const Mat *src1;
    const Mat *src2;
    Mat *dist;
    Mat *nidx;
    const Mat *mask;
    int K;
    int update;
    BatchDistFunc func;
};

}

void cv::batchDistance( InputArray _src1, InputArray _src2,
                       OutputArray _dist, int dtype, OutputArray _nidx,
                       int normType, int K, InputArray _mask,
                       int update, bool crosscheck )
{
    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int type = src1.type();
    CV_Assert( type == src2.type() && src1.cols == src2.cols &&
              (type == CV_32F || type == CV_8U));
    CV_Assert( _nidx.needed() == (K > 0) );

    if( dtype == -1 )
    {
        dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ? CV_32S : CV_32F;
    }
    CV_Assert( (type == CV_8U && dtype == CV_32S) || dtype == CV_32F);

    K = std::min(K, src2.rows);

    _dist.create(src1.rows, (K > 0 ? K : src2.rows), dtype);
    Mat dist = _dist.getMat(), nidx;
    if( _nidx.needed() )
    {
        _nidx.create(dist.size(), CV_32S);
        nidx = _nidx.getMat();
    }

    if( update == 0 && K > 0 )
    {
        dist = Scalar::all(dtype == CV_32S ? (double)INT_MAX : (double)FLT_MAX);
        nidx = Scalar::all(-1);
    }

    if( crosscheck )
    {
        CV_Assert( K == 1 && update == 0 && mask.empty() );
        Mat tdist, tidx;
        batchDistance(src2, src1, tdist, dtype, tidx, normType, K, mask, 0, false);

        // if an idx-th element from src1 appeared to be the nearest to i-th element of src2,
        // we update the minimum mutual distance between idx-th element of src1 and the whole src2 set.
        // As a result, if nidx[idx] = i*, it means that idx-th element of src1 is the nearest
        // to i*-th element of src2 and i*-th element of src2 is the closest to idx-th element of src1.
        // If nidx[idx] = -1, it means that there is no such ideal couple for it in src2.
        // This O(N) procedure is called cross-check and it helps to eliminate some false matches.
        if( dtype == CV_32S )
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                int d = tdist.at<int>(i), d0 = dist.at<int>(idx);
                if( d < d0 )
                {
                    dist.at<int>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        else
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                float d = tdist.at<float>(i), d0 = dist.at<float>(idx);
                if( d < d0 )
                {
                    dist.at<float>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        return;
    }

    BatchDistFunc func = 0;
    if( type == CV_8U )
    {
        if( normType == NORM_L1 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL1_8u32s;
        else if( normType == NORM_L1 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL1_8u32f;
        else if( normType == NORM_L2SQR && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL2Sqr_8u32s;
        else if( normType == NORM_L2SQR && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2Sqr_8u32f;
        else if( normType == NORM_L2 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2_8u32f;
        else if( normType == NORM_HAMMING && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming;
        else if( normType == NORM_HAMMING2 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming2;
    }
    else if( type == CV_32F && dtype == CV_32F )
    {
        if( normType == NORM_L1 )
            func = (BatchDistFunc)batchDistL1_32f;
        else if( normType == NORM_L2SQR )
            func = (BatchDistFunc)batchDistL2Sqr_32f;
        else if( normType == NORM_L2 )
            func = (BatchDistFunc)batchDistL2_32f;
    }

    if( func == 0 )
        CV_Error_(CV_StsUnsupportedFormat,
                  ("The combination of type=%d, dtype=%d and normType=%d is not supported",
                   type, dtype, normType));

    parallel_for_(Range(0, src1.rows),
                  BatchDistInvoker(src1, src2, dist, nidx, K, mask, update, func));
}


void cv::findNonZero( InputArray _src, OutputArray _idx )
{
    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    int n = countNonZero(src);
    if( _idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous() )
        _idx.release();
    _idx.create(n, 1, CV_32SC2);
    Mat idx = _idx.getMat();
    CV_Assert(idx.isContinuous());
    Point* idx_ptr = (Point*)idx.data;

    for( int i = 0; i < src.rows; i++ )
    {
        const uchar* bin_ptr = src.ptr(i);
        for( int j = 0; j < src.cols; j++ )
            if( bin_ptr[j] )
                *idx_ptr++ = Point(j, i);
    }
}


CV_IMPL CvScalar cvSum( const CvArr* srcarr )
{
    cv::Scalar sum = cv::sum(cv::cvarrToMat(srcarr, false, true, 1));
    if( CV_IS_IMAGE(srcarr) )
    {
        int coi = cvGetImageCOI((IplImage*)srcarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            sum = cv::Scalar(sum[coi-1]);
        }
    }
    return sum;
}

CV_IMPL int cvCountNonZero( const CvArr* imgarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);
    return countNonZero(img);
}


CV_IMPL  CvScalar
cvAvg( const void* imgarr, const void* maskarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    cv::Scalar mean = !maskarr ? cv::mean(img) : cv::mean(img, cv::cvarrToMat(maskarr));
    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
        }
    }
    return mean;
}


CV_IMPL  void
cvAvgSdv( const CvArr* imgarr, CvScalar* _mean, CvScalar* _sdv, const void* maskarr )
{
    cv::Scalar mean, sdv;

    cv::Mat mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    cv::meanStdDev(cv::cvarrToMat(imgarr, false, true, 1), mean, sdv, mask );

    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
            sdv = cv::Scalar(sdv[coi-1]);
        }
    }

    if( _mean )
        *(cv::Scalar*)_mean = mean;
    if( _sdv )
        *(cv::Scalar*)_sdv = sdv;
}


CV_IMPL void
cvMinMaxLoc( const void* imgarr, double* _minVal, double* _maxVal,
             CvPoint* _minLoc, CvPoint* _maxLoc, const void* maskarr )
{
    cv::Mat mask, img = cv::cvarrToMat(imgarr, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);

    cv::minMaxLoc( img, _minVal, _maxVal,
                   (cv::Point*)_minLoc, (cv::Point*)_maxLoc, mask );
}


CV_IMPL  double
cvNorm( const void* imgA, const void* imgB, int normType, const void* maskarr )
{
    cv::Mat a, mask;
    if( !imgA )
    {
        imgA = imgB;
        imgB = 0;
    }

    a = cv::cvarrToMat(imgA, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    if( a.channels() > 1 && CV_IS_IMAGE(imgA) && cvGetImageCOI((const IplImage*)imgA) > 0 )
        cv::extractImageCOI(imgA, a);

    if( !imgB )
        return !maskarr ? cv::norm(a, normType) : cv::norm(a, normType, mask);

    cv::Mat b = cv::cvarrToMat(imgB, false, true, 1);
    if( b.channels() > 1 && CV_IS_IMAGE(imgB) && cvGetImageCOI((const IplImage*)imgB) > 0 )
        cv::extractImageCOI(imgB, b);

    return !maskarr ? cv::norm(a, b, normType) : cv::norm(a, b, normType, mask);
}
