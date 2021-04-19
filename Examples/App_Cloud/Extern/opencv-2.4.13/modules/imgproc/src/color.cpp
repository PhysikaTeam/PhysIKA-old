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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

/********************************* COPYRIGHT NOTICE *******************************\
  Original code for Bayer->BGR/RGB conversion is provided by Dirk Schaefer
  from MD-Mathematische Dienste GmbH. Below is the copyright notice:

    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not download,
    install, copy or use the software.

    Contributors License Agreement:

      Copyright (c) 2002,
      MD-Mathematische Dienste GmbH
      Im Defdahl 5-10
      44141 Dortmund
      Germany
      www.md-it.de

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain
    the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    The name of Contributor may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************/

#include "precomp.hpp"
#include <limits>
#include <iostream>

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
#define MAX_IPP8u   255
#define MAX_IPP16u  65535
#define MAX_IPP32f  1.0
static IppStatus sts = ippInit();
#endif

namespace cv
{

// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
template<typename _Tp> static void splineBuild(const _Tp* f, int n, _Tp* tab)
{
    _Tp cn = 0;
    int i;
    tab[0] = tab[1] = (_Tp)0;

    for(i = 1; i < n-1; i++)
    {
        _Tp t = 3*(f[i+1] - 2*f[i] + f[i-1]);
        _Tp l = 1/(4 - tab[(i-1)*4]);
        tab[i*4] = l; tab[i*4+1] = (t - tab[(i-1)*4+1])*l;
    }

    for(i = n-1; i >= 0; i--)
    {
        _Tp c = tab[i*4+1] - tab[i*4]*cn;
        _Tp b = f[i+1] - f[i] - (cn + c*2)*(_Tp)0.3333333333333333;
        _Tp d = (cn - c)*(_Tp)0.3333333333333333;
        tab[i*4] = f[i]; tab[i*4+1] = b;
        tab[i*4+2] = c; tab[i*4+3] = d;
        cn = c;
    }
}

// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}


template<typename _Tp> struct ColorChannel
{
    typedef float worktype_f;
    static _Tp max() { return std::numeric_limits<_Tp>::max(); }
    static _Tp half() { return (_Tp)(max()/2 + 1); }
};

template<> struct ColorChannel<float>
{
    typedef float worktype_f;
    static float max() { return 1.f; }
    static float half() { return 0.5f; }
};

/*template<> struct ColorChannel<double>
{
    typedef double worktype_f;
    static double max() { return 1.; }
    static double half() { return 0.5; }
};*/


///////////////////////////// Top-level template function ////////////////////////////////

template <typename Cvt>
class CvtColorLoop_Invoker : public ParallelLoopBody
{
    typedef typename Cvt::channel_type _Tp;
public:

    CvtColorLoop_Invoker(const Mat& _src, Mat& _dst, const Cvt& _cvt) :
        ParallelLoopBody(), src(_src), dst(_dst), cvt(_cvt)
    {
    }

    virtual void operator()(const Range& range) const
    {
        const uchar* yS = src.ptr<uchar>(range.start);
        uchar* yD = dst.ptr<uchar>(range.start);

        for( int i = range.start; i < range.end; ++i, yS += src.step, yD += dst.step )
            cvt((const _Tp*)yS, (_Tp*)yD, src.cols);
    }

private:
    const Mat& src;
    Mat& dst;
    const Cvt& cvt;

    const CvtColorLoop_Invoker& operator= (const CvtColorLoop_Invoker&);
};

template <typename Cvt>
void CvtColorLoop(const Mat& src, Mat& dst, const Cvt& cvt)
{
    parallel_for_(Range(0, src.rows), CvtColorLoop_Invoker<Cvt>(src, dst, cvt), src.total()/(double)(1<<16) );
}

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
typedef IppStatus (CV_STDCALL* ippiReorderFunc)(const void *, int, void *, int, IppiSize, const int *);
typedef IppStatus (CV_STDCALL* ippiGeneralFunc)(const void *, int, void *, int, IppiSize);
typedef IppStatus (CV_STDCALL* ippiColor2GrayFunc)(const void *, int, void *, int, IppiSize, const Ipp32f *);

template <typename Cvt>
class CvtColorIPPLoop_Invoker : public ParallelLoopBody
{
public:

    CvtColorIPPLoop_Invoker(const Mat& _src, Mat& _dst, const Cvt& _cvt, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), cvt(_cvt), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const
    {
        const void *yS = src.ptr<uchar>(range.start);
        void *yD = dst.ptr<uchar>(range.start);
        if( !cvt(yS, (int)src.step[0], yD, (int)dst.step[0], src.cols, range.end - range.start) )
            *ok = false;
    }

private:
    const Mat& src;
    Mat& dst;
    const Cvt& cvt;
    bool *ok;

    const CvtColorIPPLoop_Invoker& operator= (const CvtColorIPPLoop_Invoker&);
};

template <typename Cvt>
bool CvtColorIPPLoop(const Mat& src, Mat& dst, const Cvt& cvt)
{
    bool ok;
    parallel_for_(Range(0, src.rows), CvtColorIPPLoop_Invoker<Cvt>(src, dst, cvt, &ok), src.total()/(double)(1<<16) );
    return ok;
}

template <typename Cvt>
bool CvtColorIPPLoopCopy(Mat& src, Mat& dst, const Cvt& cvt)
{
    Mat temp;
    Mat &source = src;
    if( src.data == dst.data )
    {
        src.copyTo(temp);
        source = temp;
    }
    bool ok;
    parallel_for_(Range(0, source.rows), CvtColorIPPLoop_Invoker<Cvt>(source, dst, cvt, &ok), source.total()/(double)(1<<16) );
    return ok;
}

static IppStatus CV_STDCALL ippiSwapChannels_8u_C3C4Rf(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_8u_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP8u);
}

static IppStatus CV_STDCALL ippiSwapChannels_16u_C3C4Rf(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_16u_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP16u);
}

static IppStatus CV_STDCALL ippiSwapChannels_32f_C3C4Rf(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_32f_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP32f);
}

static ippiReorderFunc ippiSwapChannelsC3C4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3C4Rf, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3C4Rf, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3C4Rf, 0, 0
};

static ippiGeneralFunc ippiCopyAC4C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_AC4C3R, 0, (ippiGeneralFunc)ippiCopy_16u_AC4C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_AC4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC4C3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_AC4R, 0, (ippiReorderFunc)ippiSwapChannels_16u_AC4R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_AC4R, 0, 0
};

#if 0
static ippiColor2GrayFunc ippiColor2GrayC3Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_C3C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_C3C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_C3C1R, 0, 0
};

static ippiColor2GrayFunc ippiColor2GrayC4Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_AC4C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_AC4C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_AC4C1R, 0, 0
};
#endif

static ippiGeneralFunc ippiRGB2GrayC3Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_C3C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_C3C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_C3C1R, 0, 0
};

static ippiGeneralFunc ippiRGB2GrayC4Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_AC4C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_AC4C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_AC4C1R, 0, 0
};

static ippiGeneralFunc ippiCopyP3C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_P3C3R, 0, (ippiGeneralFunc)ippiCopy_16u_P3C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_P3C3R, 0, 0
};

static ippiGeneralFunc ippiRGB2XYZTab[] =
{
    (ippiGeneralFunc)ippiRGBToXYZ_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToXYZ_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToXYZ_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiXYZ2RGBTab[] =
{
    (ippiGeneralFunc)ippiXYZToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiXYZToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiXYZToRGB_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiRGB2HSVTab[] =
{
    (ippiGeneralFunc)ippiRGBToHSV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHSV_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiHSV2RGBTab[] =
{
    (ippiGeneralFunc)ippiHSVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHSVToRGB_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiRGB2HLSTab[] =
{
    (ippiGeneralFunc)ippiRGBToHLS_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHLS_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToHLS_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiHLS2RGBTab[] =
{
    (ippiGeneralFunc)ippiHLSToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHLSToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiHLSToRGB_32f_C3R, 0, 0
};

struct IPPGeneralFunctor
{
    IPPGeneralFunctor(ippiGeneralFunc _func) : func(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiGeneralFunc func;
};

struct IPPReorderFunctor
{
    IPPReorderFunctor(ippiReorderFunc _func, int _order0, int _order1, int _order2) : func(_func)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiReorderFunc func;
    int order[4];
};

struct IPPColor2GrayFunctor
{
    IPPColor2GrayFunctor(ippiColor2GrayFunc _func) : func(_func)
    {
        coeffs[0] = 0.114f;
        coeffs[1] = 0.587f;
        coeffs[2] = 0.299f;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows), coeffs) >= 0;
    }
private:
    ippiColor2GrayFunc func;
    Ipp32f coeffs[3];
};

struct IPPGray2BGRFunctor
{
    IPPGray2BGRFunctor(ippiGeneralFunc _func) : func(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        const void* srcarray[3] = { src, src, src };
        return func(srcarray, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiGeneralFunc func;
};

struct IPPGray2BGRAFunctor
{
    IPPGray2BGRAFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _depth) : func1(_func1), func2(_func2), depth(_depth){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        const void* srcarray[3] = { src, src, src };
        Mat temp(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(srcarray, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        int order[4] = {0, 1, 2, 3};
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc func1;
    ippiReorderFunc func2;
    int depth;
};

struct IPPReorderGeneralFunctor
{
    IPPReorderGeneralFunctor(ippiReorderFunc _func1, ippiGeneralFunc _func2, int _order0, int _order1, int _order2, int _depth) : func1(_func1), func2(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(src, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows), order) < 0)
            return false;
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiReorderFunc func1;
    ippiGeneralFunc func2;
    int order[4];
    int depth;
};

struct IPPGeneralReorderFunctor
{
    IPPGeneralReorderFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _order0, int _order1, int _order2, int _depth) : func1(_func1), func2(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(src, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc func1;
    ippiReorderFunc func2;
    int order[4];
    int depth;
};
#endif

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

template<typename _Tp> struct RGB2RGB
{
    typedef _Tp channel_type;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) : srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bidx = blueIdx;
        if( dcn == 3 )
        {
            n *= 3;
            for( int i = 0; i < n; i += 3, src += scn )
            {
                _Tp t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
                dst[i] = t0; dst[i+1] = t1; dst[i+2] = t2;
            }
        }
        else if( scn == 3 )
        {
            n *= 3;
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i += 3, dst += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2];
                dst[bidx] = t0; dst[1] = t1; dst[bidx^2] = t2; dst[3] = alpha;
            }
        }
        else
        {
            n *= 4;
            for( int i = 0; i < n; i += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2], t3 = src[i+3];
                dst[i] = t2; dst[i+1] = t1; dst[i+2] = t0; dst[i+3] = t3;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};

/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

struct RGB5x52RGB
{
    typedef uchar channel_type;

    RGB5x52RGB(int _dstcn, int _blueIdx, int _greenBits)
        : dstcn(_dstcn), blueIdx(_blueIdx), greenBits(_greenBits) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        if( greenBits == 6 )
            for( int i = 0; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 3) & ~3);
                dst[bidx ^ 2] = (uchar)((t >> 8) & ~7);
                if( dcn == 4 )
                    dst[3] = 255;
            }
        else
            for( int i = 0; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 2) & ~7);
                dst[bidx ^ 2] = (uchar)((t >> 7) & ~7);
                if( dcn == 4 )
                    dst[3] = t & 0x8000 ? 255 : 0;
            }
    }

    int dstcn, blueIdx, greenBits;
};


struct RGB2RGB5x5
{
    typedef uchar channel_type;

    RGB2RGB5x5(int _srccn, int _blueIdx, int _greenBits)
        : srccn(_srccn), blueIdx(_blueIdx), greenBits(_greenBits) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        if( greenBits == 6 )
            for( int i = 0; i < n; i++, src += scn )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
        else if( scn == 3 )
            for( int i = 0; i < n; i++, src += 3 )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|((src[bidx^2]&~7) << 7));
            }
        else
            for( int i = 0; i < n; i++, src += 4 )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|
                    ((src[bidx^2]&~7) << 7)|(src[3] ? 0x8000 : 0));
            }
    }

    int srccn, blueIdx, greenBits;
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////

template<typename _Tp>
struct Gray2RGB
{
    typedef _Tp channel_type;

    Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        if( dstcn == 3 )
            for( int i = 0; i < n; i++, dst += 3 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
            }
        else
        {
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i++, dst += 4 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
                dst[3] = alpha;
            }
        }
    }

    int dstcn;
};


struct Gray2RGB5x5
{
    typedef uchar channel_type;

    Gray2RGB5x5(int _greenBits) : greenBits(_greenBits) {}
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if( greenBits == 6 )
            for( int i = 0; i < n; i++ )
            {
                int t = src[i];
                ((ushort*)dst)[i] = (ushort)((t >> 3)|((t & ~3) << 3)|((t & ~7) << 8));
            }
        else
            for( int i = 0; i < n; i++ )
            {
                int t = src[i] >> 3;
                ((ushort*)dst)[i] = (ushort)(t|(t << 5)|(t << 10));
            }
    }
    int greenBits;
};


#undef R2Y
#undef G2Y
#undef B2Y

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899,
    G2Y = 9617,
    B2Y = 1868,
    BLOCK_SIZE = 256
};


struct RGB5x52Gray
{
    typedef uchar channel_type;

    RGB5x52Gray(int _greenBits) : greenBits(_greenBits) {}
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if( greenBits == 6 )
            for( int i = 0; i < n; i++ )
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 3) & 0xfc)*G2Y +
                                           ((t >> 8) & 0xf8)*R2Y, yuv_shift);
            }
        else
            for( int i = 0; i < n; i++ )
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 2) & 0xf8)*G2Y +
                                           ((t >> 7) & 0xf8)*R2Y, yuv_shift);
            }
    }
    int greenBits;
};


template<typename _Tp> struct RGB2Gray
{
    typedef _Tp channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { 0.299f, 0.587f, 0.114f };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = saturate_cast<_Tp>(src[0]*cb + src[1]*cg + src[2]*cr);
    }
    int srccn;
    float coeffs[3];
};


template<> struct RGB2Gray<uchar>
{
    typedef uchar channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { R2Y, G2Y, B2Y };
        if(!coeffs) coeffs = coeffs0;

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs[blueIdx^2], dg = coeffs[1], dr = coeffs[blueIdx];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
    }
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn;
        const int* _tab = tab;
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (uchar)((_tab[src[0]] + _tab[src[1]+256] + _tab[src[2]+512]) >> yuv_shift);
    }
    int srccn;
    int tab[256*3];
};


template<> struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }
    int srccn;
    int coeffs[3];
};


///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

template<typename _Tp> struct RGB2YCrCb_f
{
    typedef _Tp channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, const float* _coeffs) : srccn(_srccn), blueIdx(_blueIdx)
    {
        static const float coeffs0[] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp Y = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Cr = saturate_cast<_Tp>((src[bidx^2] - Y)*C3 + delta);
            _Tp Cb = saturate_cast<_Tp>((src[bidx] - Y)*C4 + delta);
            dst[i] = Y; dst[i+1] = Cr; dst[i+2] = Cb;
        }
    }
    int srccn, blueIdx;
    float coeffs[5];
};


template<typename _Tp> struct RGB2YCrCb_i
{
    typedef _Tp channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, const int* _coeffs)
        : srccn(_srccn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] = {R2Y, G2Y, B2Y, 11682, 9241};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<_Tp>::half()*(1 << yuv_shift);
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<_Tp>(Y);
            dst[i+1] = saturate_cast<_Tp>(Cr);
            dst[i+2] = saturate_cast<_Tp>(Cb);
        }
    }
    int srccn, blueIdx;
    int coeffs[5];
};


template<typename _Tp> struct YCrCb2RGB_f
{
    typedef _Tp channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
        : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const float coeffs0[] = {1.403f, -0.714f, -0.344f, 1.773f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 4*sizeof(coeffs[0]));
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1];
            _Tp Cb = src[i+2];

            _Tp b = saturate_cast<_Tp>(Y + (Cb - delta)*C3);
            _Tp g = saturate_cast<_Tp>(Y + (Cb - delta)*C2 + (Cr - delta)*C1);
            _Tp r = saturate_cast<_Tp>(Y + (Cr - delta)*C0);

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[4];
};


template<typename _Tp> struct YCrCb2RGB_i
{
    typedef _Tp channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
        : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] = {22987, -11698, -5636, 29049};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 4*sizeof(coeffs[0]));
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1];
            _Tp Cb = src[i+2];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<_Tp>(b);
            dst[1] = saturate_cast<_Tp>(g);
            dst[bidx^2] = saturate_cast<_Tp>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[4];
};


////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float XYZ2sRGB_D65[] =
{
    3.240479f, -1.53715f, -0.498535f,
    -0.969256f, 1.875991f, 0.041556f,
    0.055648f, -0.204043f, 1.057311f
};

template<typename _Tp> struct RGB2XYZ_f
{
    typedef _Tp channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        memcpy(coeffs, _coeffs ? _coeffs : sRGB2XYZ_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp X = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Y = saturate_cast<_Tp>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            _Tp Z = saturate_cast<_Tp>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }
    int srccn;
    float coeffs[9];
};


template<typename _Tp> struct RGB2XYZ_i
{
    typedef _Tp channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] =
        {
            1689,    1465,    739,
            871,     2929,    296,
            79,      488,     3892
        };
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<_Tp>(X); dst[i+1] = saturate_cast<_Tp>(Y);
            dst[i+2] = saturate_cast<_Tp>(Z);
        }
    }
    int srccn;
    int coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_f
{
    typedef _Tp channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        memcpy(coeffs, _coeffs ? _coeffs : XYZ2sRGB_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp B = saturate_cast<_Tp>(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2);
            _Tp G = saturate_cast<_Tp>(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5);
            _Tp R = saturate_cast<_Tp>(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_i
{
    typedef _Tp channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] =
        {
            13273,  -6296,  -2042,
            -3970,   7684,    170,
              228,   -836,   4331
        };
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<_Tp>(B); dst[1] = saturate_cast<_Tp>(G);
            dst[2] = saturate_cast<_Tp>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////


struct RGB2HSV_b
{
    typedef uchar channel_type;

    RGB2HSV_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    {
        CV_Assert( hrange == 180 || hrange == 256 );
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized = false;

        int hr = hrange;
        const int* hdiv_table = hr == 180 ? hdiv_table180 : hdiv_table256;
        n *= 3;

        if( !initialized )
        {
            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
            for( i = 1; i < 256; i++ )
            {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
                hdiv_table256[i] = saturate_cast<int>((256 << hsv_shift)/(6.*i));
            }
            initialized = true;
        }

        for( i = 0; i < n; i += 3, src += scn )
        {
            int b = src[bidx], g = src[1], r = src[bidx^2];
            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            diff = v - vmin;
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += h < 0 ? hr : 0;

            dst[i] = saturate_cast<uchar>(h);
            dst[i+1] = (uchar)s;
            dst[i+2] = (uchar)v;
        }
    }

    int srccn, blueIdx, hrange;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;

            dst[i] = h*hscale;
            dst[i+1] = s;
            dst[i+2] = v;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float h = src[i], s = src[i+1], v = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = cvFloor(h);
                h -= sector;
                if( (unsigned)sector >= 6u )
                {
                    sector = 0;
                    h = 0.f;
                }

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HSV2RGB_f cvt;
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            if( vmax < g ) vmax = g;
            if( vmax < b ) vmax = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = vmax - vmin;
            l = (vmax + vmin)*0.5f;

            if( diff > FLT_EPSILON )
            {
                s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
                diff = 60.f/diff;

                if( vmax == r )
                    h = (g - b)*diff;
                else if( vmax == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0.f ) h += 360.f;
            }

            dst[i] = h*hscale;
            dst[i+1] = l;
            dst[i+2] = s;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct RGB2HLS_b
{
    typedef uchar channel_type;

    RGB2HLS_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), cvt(3, _blueIdx, (float)_hrange) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = src[1]*(1.f/255.f);
                buf[j+2] = src[2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*255.f);
            }
        }
    }

    int srccn;
    RGB2HLS_f cvt;
};


struct HLS2RGB_f
{
    typedef float channel_type;

    HLS2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float h = src[i], l = src[i+1], s = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = l;
            else
            {
                static const int sector_data[][3]=
                {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;

                float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
                float p1 = 2*l - p2;

                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                assert( 0 <= h && h < 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1)*(1-h);
                tab[3] = p1 + (p2 - p1)*h;

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HLS2RGB_b
{
    typedef uchar channel_type;

    HLS2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HLS2RGB_f cvt;
};


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

static const float D65[] = { 0.950456f, 1.f, 1.088754f };

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static float LabCbrtTab[LAB_CBRT_TAB_SIZE*4];
static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

static float sRGBGammaTab[GAMMA_TAB_SIZE*4], sRGBInvGammaTab[GAMMA_TAB_SIZE*4];
static const float GammaTabScale = (float)GAMMA_TAB_SIZE;

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        float f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1], scale = 1.f/LabCbrtTabScale;
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            float x = i*scale;
            f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);
        }
        splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

        scale = 1.f/GammaTabScale;
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            float x = i*scale;
            g[i] = x <= 0.04045f ? x*(1.f/12.92f) : (float)pow((double)(x + 0.055)*(1./1.055), 2.4);
            ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*pow((double)x, 1./2.4) - 0.055);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*(x <= 0.04045f ? x*(1.f/12.92f) : (float)pow((double)(x + 0.055)*(1./1.055), 2.4)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }

        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            float x = i*(1.f/(255.f*(1 << gamma_shift)));
            LabCbrtTab_b[i] = saturate_cast<ushort>((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));
        }
        initialized = true;
    }
}

struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        static volatile int _3 = 3;
        initLabTabs();

        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!_whitept) _whitept = D65;
        float scale[] =
        {
            (1 << lab_shift)/_whitept[0],
            (float)(1 << lab_shift),
            (1 << lab_shift)/_whitept[2]
        };

        for( int i = 0; i < _3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = cvRound(_coeffs[i*3]*scale[i]);
            coeffs[i*3+1] = cvRound(_coeffs[i*3+1]*scale[i]);
            coeffs[i*3+blueIdx] = cvRound(_coeffs[i*3+2]*scale[i]);

            CV_Assert( coeffs[i] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift) );
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[i] = saturate_cast<uchar>(L);
            dst[i+1] = saturate_cast<uchar>(a);
            dst[i+2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
};


#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int _3 = 3;
        initLabTabs();

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        float scale[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

        for( int i = 0; i < _3; i++ )
        {
            int j = i * 3;
            coeffs[j + (blueIdx ^ 2)] = _coeffs[j] * scale[i];
            coeffs[j + 1] = _coeffs[j + 1] * scale[i];
            coeffs[j + blueIdx] = _coeffs[j + 2] * scale[i];

            CV_Assert( coeffs[j] >= 0 && coeffs[j + 1] >= 0 && coeffs[j + 2] >= 0 &&
                       coeffs[j] + coeffs[j + 1] + coeffs[j + 2] < 1.5f*LabCbrtTabScale );
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        static const float _1_3 = 1.0f / 3.0f;
        static const float _a = 16.0f / 116.0f;
        for (i = 0; i < n; i += 3, src += scn )
        {
            float R = clip(src[0]);
            float G = clip(src[1]);
            float B = clip(src[2]);

//            CV_Assert(R >= 0.0f && R <= 1.0f);
//            CV_Assert(G >= 0.0f && G <= 1.0f);
//            CV_Assert(B >= 0.0f && B <= 1.0f);

            if (gammaTab)
            {
                R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float FX = X > 0.008856f ? pow(X, _1_3) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? pow(Y, _1_3) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? pow(Z, _1_3) : (7.787f * Z + _a);

            float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
            float a = 500.f * (FX - FY);
            float b = 200.f * (FY - FZ);

            dst[i] = L;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
};

struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueInd(blueIdx)
    {
        initLabTabs();

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i]*_whitept[i];
            coeffs[i+3] = _coeffs[i+3]*_whitept[i];
            coeffs[i+blueIdx*3] = _coeffs[i+6]*_whitept[i];
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        static const float lThresh = 0.008856f * 903.3f;
        static const float fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
        for (i = 0; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];


            float x = fxz[0], z = fxz[1];
            float ro = clip(C0 * x + C1 * y + C2 * z);
            float go = clip(C3 * x + C4 * y + C5 * z);
            float bo = clip(C6 * x + C7 * y + C8 * z);

//            CV_Assert(ro >= 0.0f && ro <= 1.0f);
//            CV_Assert(go >= 0.0f && go <= 1.0f);
//            CV_Assert(bo >= 0.0f && bo <= 1.0f);

            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    int blueInd;
};

#undef clip

struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb ) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Lab2RGB_f cvt;
};


///////////////////////////////////// RGB <-> L*u*v* /////////////////////////////////////

struct RGB2Luv_f
{
    typedef float channel_type;

    RGB2Luv_f( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int i;
        initLabTabs();

        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!whitept) whitept = D65;

        for( i = 0; i < 3; i++ )
        {
            coeffs[i*3] = _coeffs[i*3];
            coeffs[i*3+1] = _coeffs[i*3+1];
            coeffs[i*3+2] = _coeffs[i*3+2];
            if( blueIdx == 0 )
                std::swap(coeffs[i*3], coeffs[i*3+2]);
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 1.5f );
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d;
        vn = 9*whitept[1]*d;

        CV_Assert(whitept[1] == 1.f);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float _un = 13*un, _vn = 13*vn;
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float R = src[0], G = src[1], B = src[2];
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f*L - 16.f;

            float d = (4*13) / std::max(X + 15 * Y + 3 * Z, FLT_EPSILON);
            float u = L*(X*d - _un);
            float v = L*((9*0.25f)*Y*d - _vn);

            dst[i] = L; dst[i+1] = u; dst[i+2] = v;
        }
    }

    int srccn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct Luv2RGB_f
{
    typedef float channel_type;

    Luv2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb)
    {
        initLabTabs();

        if(!_coeffs) _coeffs = XYZ2sRGB_D65;
        if(!whitept) whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i];
            coeffs[i+3] = _coeffs[i+3];
            coeffs[i+blueIdx*3] = _coeffs[i+6];
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d;
        vn = 9*whitept[1]*d;

        CV_Assert(whitept[1] == 1.f);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        float _un = un, _vn = vn;
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float L = src[i], u = src[i+1], v = src[i+2], d, X, Y, Z;
            Y = (L + 16.f) * (1.f/116.f);
            Y = Y*Y*Y;
            d = (1.f/13.f)/L;
            u = u*d + _un;
            v = v*d + _vn;
            float iv = 1.f/v;
            X = 2.25f * u * Y * iv ;
            Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

            float R = X*C0 + Y*C1 + Z*C2;
            float G = X*C3 + Y*C4 + Z*C5;
            float B = X*C6 + Y*C7 + Z*C8;

            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = R; dst[1] = G; dst[2] = B;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct RGB2Luv_b
{
    typedef uchar channel_type;

    RGB2Luv_b( int _srccn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : srccn(_srccn), cvt(3, blueIdx, _coeffs, _whitept, _srgb) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = (float)(src[1]*(1.f/255.f));
                buf[j+2] = (float)(src[2]*(1.f/255.f));
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]*2.55f);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*0.72033898305084743f + 96.525423728813564f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*0.9732824427480916f + 136.259541984732824f);
            }
        }
    }

    int srccn;
    RGB2Luv_f cvt;
};


struct Luv2RGB_b
{
    typedef uchar channel_type;

    Luv2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb ) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1]*1.388235294117647f - 134.f);
                buf[j+2] = (float)(src[j+2]*1.027450980392157f - 140.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Luv2RGB_f cvt;
};


//////////////////////////// Bayer Pattern -> RGB conversion /////////////////////////////

template<typename T>
class SIMDBayerStubInterpolator_
{
public:
    int bayer2Gray(const T*, int, T*, int, int, int, int) const
    {
        return 0;
    }

    int bayer2RGB(const T*, int, T*, int, int) const
    {
        return 0;
    }
};

#if CV_SSE2
class SIMDBayerInterpolator_8u
{
public:
    SIMDBayerInterpolator_8u()
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int bayer2Gray(const uchar* bayer, int bayer_step, uchar* dst,
                   int width, int bcoeff, int gcoeff, int rcoeff) const
    {
        if( !use_simd )
            return 0;

        __m128i _b2y = _mm_set1_epi16((short)(rcoeff*2));
        __m128i _g2y = _mm_set1_epi16((short)(gcoeff*2));
        __m128i _r2y = _mm_set1_epi16((short)(bcoeff*2));
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 14 )
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)bayer);
            __m128i r1 = _mm_loadu_si128((const __m128i*)(bayer+bayer_step));
            __m128i r2 = _mm_loadu_si128((const __m128i*)(bayer+bayer_step*2));

            __m128i b1 = _mm_add_epi16(_mm_srli_epi16(_mm_slli_epi16(r0, 8), 7),
                                       _mm_srli_epi16(_mm_slli_epi16(r2, 8), 7));
            __m128i b0 = _mm_add_epi16(b1, _mm_srli_si128(b1, 2));
            b1 = _mm_slli_epi16(_mm_srli_si128(b1, 2), 1);

            __m128i g0 = _mm_add_epi16(_mm_srli_epi16(r0, 7), _mm_srli_epi16(r2, 7));
            __m128i g1 = _mm_srli_epi16(_mm_slli_epi16(r1, 8), 7);
            g0 = _mm_add_epi16(g0, _mm_add_epi16(g1, _mm_srli_si128(g1, 2)));
            g1 = _mm_slli_epi16(_mm_srli_si128(g1, 2), 2);

            r0 = _mm_srli_epi16(r1, 8);
            r1 = _mm_slli_epi16(_mm_add_epi16(r0, _mm_srli_si128(r0, 2)), 2);
            r0 = _mm_slli_epi16(r0, 3);

            g0 = _mm_add_epi16(_mm_mulhi_epi16(b0, _b2y), _mm_mulhi_epi16(g0, _g2y));
            g1 = _mm_add_epi16(_mm_mulhi_epi16(b1, _b2y), _mm_mulhi_epi16(g1, _g2y));
            g0 = _mm_add_epi16(g0, _mm_mulhi_epi16(r0, _r2y));
            g1 = _mm_add_epi16(g1, _mm_mulhi_epi16(r1, _r2y));
            g0 = _mm_srli_epi16(g0, 2);
            g1 = _mm_srli_epi16(g1, 2);
            g0 = _mm_packus_epi16(g0, g0);
            g1 = _mm_packus_epi16(g1, g1);
            g0 = _mm_unpacklo_epi8(g0, g1);
            _mm_storeu_si128((__m128i*)dst, g0);
        }

        return (int)(bayer - (bayer_end - width));
    }

    int bayer2RGB(const uchar* bayer, int bayer_step, uchar* dst, int width, int blue) const
    {
        if( !use_simd )
            return 0;
        /*
         B G B G | B G B G | B G B G | B G B G
         G R G R | G R G R | G R G R | G R G R
         B G B G | B G B G | B G B G | B G B G
         */
        __m128i delta1 = _mm_set1_epi16(1), delta2 = _mm_set1_epi16(2);
        __m128i mask = _mm_set1_epi16(blue < 0 ? -1 : 0), z = _mm_setzero_si128();
        __m128i masklo = _mm_set1_epi16(0x00ff);
        const uchar* bayer_end = bayer + width;

        for( ; bayer <= bayer_end - 18; bayer += 14, dst += 42 )
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)bayer);
            __m128i r1 = _mm_loadu_si128((const __m128i*)(bayer+bayer_step));
            __m128i r2 = _mm_loadu_si128((const __m128i*)(bayer+bayer_step*2));

            __m128i b1 = _mm_add_epi16(_mm_and_si128(r0, masklo), _mm_and_si128(r2, masklo));
            __m128i b0 = _mm_add_epi16(b1, _mm_srli_si128(b1, 2));
            b1 = _mm_srli_si128(b1, 2);
            b1 = _mm_srli_epi16(_mm_add_epi16(b1, delta1), 1);
            b0 = _mm_srli_epi16(_mm_add_epi16(b0, delta2), 2);
            b0 = _mm_packus_epi16(b0, b1);

            __m128i g0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_srli_epi16(r2, 8));
            __m128i g1 = _mm_and_si128(r1, masklo);
            g0 = _mm_add_epi16(g0, _mm_add_epi16(g1, _mm_srli_si128(g1, 2)));
            g1 = _mm_srli_si128(g1, 2);
            g0 = _mm_srli_epi16(_mm_add_epi16(g0, delta2), 2);
            g0 = _mm_packus_epi16(g0, g1);

            r0 = _mm_srli_epi16(r1, 8);
            r1 = _mm_add_epi16(r0, _mm_srli_si128(r0, 2));
            r1 = _mm_srli_epi16(_mm_add_epi16(r1, delta1), 1);
            r0 = _mm_packus_epi16(r0, r1);

            b1 = _mm_and_si128(_mm_xor_si128(b0, r0), mask);
            b0 = _mm_xor_si128(b0, b1);
            r0 = _mm_xor_si128(r0, b1);

            // b1 g1 b1 g1 ...
            b1 = _mm_unpackhi_epi8(b0, g0);
            // b0 g0 b2 g2 b4 g4 ....
            b0 = _mm_unpacklo_epi8(b0, g0);

            // r1 0 r3 0 ...
            r1 = _mm_unpackhi_epi8(r0, z);
            // r0 0 r2 0 r4 0 ...
            r0 = _mm_unpacklo_epi8(r0, z);

            // 0 b0 g0 r0 0 b2 g2 r2 0 ...
            g0 = _mm_slli_si128(_mm_unpacklo_epi16(b0, r0), 1);
            // 0 b8 g8 r8 0 b10 g10 r10 0 ...
            g1 = _mm_slli_si128(_mm_unpackhi_epi16(b0, r0), 1);

            // b1 g1 r1 0 b3 g3 r3 ....
            r0 = _mm_unpacklo_epi16(b1, r1);
            // b9 g9 r9 0 ...
            r1 = _mm_unpackhi_epi16(b1, r1);

            b0 = _mm_srli_si128(_mm_unpacklo_epi32(g0, r0), 1);
            b1 = _mm_srli_si128(_mm_unpackhi_epi32(g0, r0), 1);

            _mm_storel_epi64((__m128i*)(dst-1+0), b0);
            _mm_storel_epi64((__m128i*)(dst-1+6*1), _mm_srli_si128(b0, 8));
            _mm_storel_epi64((__m128i*)(dst-1+6*2), b1);
            _mm_storel_epi64((__m128i*)(dst-1+6*3), _mm_srli_si128(b1, 8));

            g0 = _mm_srli_si128(_mm_unpacklo_epi32(g1, r1), 1);
            g1 = _mm_srli_si128(_mm_unpackhi_epi32(g1, r1), 1);

            _mm_storel_epi64((__m128i*)(dst-1+6*4), g0);
            _mm_storel_epi64((__m128i*)(dst-1+6*5), _mm_srli_si128(g0, 8));

            _mm_storel_epi64((__m128i*)(dst-1+6*6), g1);
        }

        return (int)(bayer - (bayer_end - width));
    }

    bool use_simd;
};
#else
typedef SIMDBayerStubInterpolator_<uchar> SIMDBayerInterpolator_8u;
#endif

template<typename T, class SIMDInterpolator>
static void Bayer2Gray_( const Mat& srcmat, Mat& dstmat, int code )
{
    SIMDInterpolator vecOp;
    const int R2Y = 4899;
    const int G2Y = 9617;
    const int B2Y = 1868;
    const int SHIFT = 14;

    const T* bayer0 = (const T*)srcmat.data;
    int bayer_step = (int)(srcmat.step/sizeof(T));
    T* dst0 = (T*)dstmat.data;
    int dst_step = (int)(dstmat.step/sizeof(T));
    Size size = srcmat.size();
    int bcoeff = B2Y, rcoeff = R2Y;
    int start_with_green = code == CV_BayerGB2GRAY || code == CV_BayerGR2GRAY;
    bool brow = true;

    if( code != CV_BayerBG2GRAY && code != CV_BayerGB2GRAY )
    {
        brow = false;
        std::swap(bcoeff, rcoeff);
    }

    dst0 += dst_step + 1;
    size.height -= 2;
    size.width -= 2;

    for( ; size.height-- > 0; bayer0 += bayer_step, dst0 += dst_step )
    {
        unsigned t0, t1, t2;
        const T* bayer = bayer0;
        T* dst = dst0;
        const T* bayer_end = bayer + size.width;

        if( size.width <= 0 )
        {
            dst[-1] = dst[size.width] = 0;
            continue;
        }

        if( start_with_green )
        {
            t0 = (bayer[1] + bayer[bayer_step*2+1])*rcoeff;
            t1 = (bayer[bayer_step] + bayer[bayer_step+2])*bcoeff;
            t2 = bayer[bayer_step+1]*(2*G2Y);

            dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
            bayer++;
            dst++;
        }

        int delta = vecOp.bayer2Gray(bayer, bayer_step, dst, size.width, bcoeff, G2Y, rcoeff);
        bayer += delta;
        dst += delta;

        for( ; bayer <= bayer_end - 2; bayer += 2, dst += 2 )
        {
            t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
            t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
            t2 = bayer[bayer_step+1]*(4*bcoeff);
            dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);

            t0 = (bayer[2] + bayer[bayer_step*2+2])*rcoeff;
            t1 = (bayer[bayer_step+1] + bayer[bayer_step+3])*bcoeff;
            t2 = bayer[bayer_step+2]*(2*G2Y);
            dst[1] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+1);
        }

        if( bayer < bayer_end )
        {
            t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] + bayer[bayer_step*2+2])*rcoeff;
            t1 = (bayer[1] + bayer[bayer_step] + bayer[bayer_step+2] + bayer[bayer_step*2+1])*G2Y;
            t2 = bayer[bayer_step+1]*(4*bcoeff);
            dst[0] = (T)CV_DESCALE(t0 + t1 + t2, SHIFT+2);
            bayer++;
            dst++;
        }

        dst0[-1] = dst0[0];
        dst0[size.width] = dst0[size.width-1];

        brow = !brow;
        std::swap(bcoeff, rcoeff);
        start_with_green = !start_with_green;
    }

    size = dstmat.size();
    dst0 = (T*)dstmat.data;
    if( size.height > 2 )
        for( int i = 0; i < size.width; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width; i++ )
        {
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
        }
}

template<typename T, class SIMDInterpolator>
static void Bayer2RGB_( const Mat& srcmat, Mat& dstmat, int code )
{
    SIMDInterpolator vecOp;
    const T* bayer0 = (const T*)srcmat.data;
    int bayer_step = (int)(srcmat.step/sizeof(T));
    T* dst0 = (T*)dstmat.data;
    int dst_step = (int)(dstmat.step/sizeof(T));
    Size size = srcmat.size();
    int blue = code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ? -1 : 1;
    int start_with_green = code == CV_BayerGB2BGR || code == CV_BayerGR2BGR;

    dst0 += dst_step + 3 + 1;
    size.height -= 2;
    size.width -= 2;

    for( ; size.height-- > 0; bayer0 += bayer_step, dst0 += dst_step )
    {
        int t0, t1;
        const T* bayer = bayer0;
        T* dst = dst0;
        const T* bayer_end = bayer + size.width;

        if( size.width <= 0 )
        {
            dst[-4] = dst[-3] = dst[-2] = dst[size.width*3-1] =
            dst[size.width*3] = dst[size.width*3+1] = 0;
            continue;
        }

        if( start_with_green )
        {
            t0 = (bayer[1] + bayer[bayer_step*2+1] + 1) >> 1;
            t1 = (bayer[bayer_step] + bayer[bayer_step+2] + 1) >> 1;
            dst[-blue] = (T)t0;
            dst[0] = bayer[bayer_step+1];
            dst[blue] = (T)t1;
            bayer++;
            dst += 3;
        }

        int delta = vecOp.bayer2RGB(bayer, bayer_step, dst, size.width, blue);
        bayer += delta;
        dst += delta*3;

        if( blue > 0 )
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[-1] = (T)t0;
                dst[0] = (T)t1;
                dst[1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[2] = (T)t0;
                dst[3] = bayer[bayer_step+2];
                dst[4] = (T)t1;
            }
        }
        else
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[1] = (T)t0;
                dst[0] = (T)t1;
                dst[-1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[4] = (T)t0;
                dst[3] = bayer[bayer_step+2];
                dst[2] = (T)t1;
            }
        }

        if( bayer < bayer_end )
        {
            t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                  bayer[bayer_step*2+2] + 2) >> 2;
            t1 = (bayer[1] + bayer[bayer_step] +
                  bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
            dst[-blue] = (T)t0;
            dst[0] = (T)t1;
            dst[blue] = bayer[bayer_step+1];
            bayer++;
            dst += 3;
        }

        dst0[-4] = dst0[-1];
        dst0[-3] = dst0[0];
        dst0[-2] = dst0[1];
        dst0[size.width*3-1] = dst0[size.width*3-4];
        dst0[size.width*3] = dst0[size.width*3-3];
        dst0[size.width*3+1] = dst0[size.width*3-2];

        blue = -blue;
        start_with_green = !start_with_green;
    }

    size = dstmat.size();
    dst0 = (T*)dstmat.data;
    if( size.height > 2 )
        for( int i = 0; i < size.width*3; i++ )
        {
            dst0[i] = dst0[i + dst_step];
            dst0[i + (size.height-1)*dst_step] = dst0[i + (size.height-2)*dst_step];
        }
    else
        for( int i = 0; i < size.width*3; i++ )
        {
            dst0[i] = dst0[i + (size.height-1)*dst_step] = 0;
        }
}


/////////////////// Demosaicing using Variable Number of Gradients ///////////////////////

static void Bayer2RGB_VNG_8u( const Mat& srcmat, Mat& dstmat, int code )
{
    const uchar* bayer = srcmat.data;
    int bstep = (int)srcmat.step;
    uchar* dst = dstmat.data;
    int dststep = (int)dstmat.step;
    Size size = srcmat.size();

    int blueIdx = code == CV_BayerBG2BGR_VNG || code == CV_BayerGB2BGR_VNG ? 0 : 2;
    bool greenCell0 = code != CV_BayerBG2BGR_VNG && code != CV_BayerRG2BGR_VNG;

    // for too small images use the simple interpolation algorithm
    if( MIN(size.width, size.height) < 8 )
    {
        Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>( srcmat, dstmat, code );
        return;
    }

    const int brows = 3, bcn = 7;
    int N = size.width, N2 = N*2, N3 = N*3, N4 = N*4, N5 = N*5, N6 = N*6, N7 = N*7;
    int i, bufstep = N7*bcn;
    cv::AutoBuffer<ushort> _buf(bufstep*brows);
    ushort* buf = (ushort*)_buf;

    bayer += bstep*2;

#if CV_SSE2
    bool haveSSE = cv::checkHardwareSupport(CV_CPU_SSE2);
    #define _mm_absdiff_epu16(a,b) _mm_adds_epu16(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a))
#endif

    for( int y = 2; y < size.height - 4; y++ )
    {
        uchar* dstrow = dst + dststep*y + 6;
        const uchar* srow;

        for( int dy = (y == 2 ? -1 : 1); dy <= 1; dy++ )
        {
            ushort* brow = buf + ((y + dy - 1)%brows)*bufstep + 1;
            srow = bayer + (y+dy)*bstep + 1;

            for( i = 0; i < bcn; i++ )
                brow[N*i-1] = brow[(N-2) + N*i] = 0;

            i = 1;

#if CV_SSE2
            if( haveSSE )
            {
                __m128i z = _mm_setzero_si128();
                for( ; i <= N-9; i += 8, srow += 8, brow += 8 )
                {
                    __m128i s1, s2, s3, s4, s6, s7, s8, s9;

                    s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1-bstep)),z);
                    s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep)),z);
                    s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1-bstep)),z);

                    s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1)),z);
                    s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1)),z);

                    s7 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1+bstep)),z);
                    s8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep)),z);
                    s9 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1+bstep)),z);

                    __m128i b0, b1, b2, b3, b4, b5, b6;

                    b0 = _mm_adds_epu16(_mm_slli_epi16(_mm_absdiff_epu16(s2,s8),1),
                                        _mm_adds_epu16(_mm_absdiff_epu16(s1, s7),
                                                       _mm_absdiff_epu16(s3, s9)));
                    b1 = _mm_adds_epu16(_mm_slli_epi16(_mm_absdiff_epu16(s4,s6),1),
                                        _mm_adds_epu16(_mm_absdiff_epu16(s1, s3),
                                                       _mm_absdiff_epu16(s7, s9)));
                    b2 = _mm_slli_epi16(_mm_absdiff_epu16(s3,s7),1);
                    b3 = _mm_slli_epi16(_mm_absdiff_epu16(s1,s9),1);

                    _mm_storeu_si128((__m128i*)brow, b0);
                    _mm_storeu_si128((__m128i*)(brow + N), b1);
                    _mm_storeu_si128((__m128i*)(brow + N2), b2);
                    _mm_storeu_si128((__m128i*)(brow + N3), b3);

                    b4 = _mm_adds_epu16(b2,_mm_adds_epu16(_mm_absdiff_epu16(s2, s4),
                                                          _mm_absdiff_epu16(s6, s8)));
                    b5 = _mm_adds_epu16(b3,_mm_adds_epu16(_mm_absdiff_epu16(s2, s6),
                                                          _mm_absdiff_epu16(s4, s8)));
                    b6 = _mm_adds_epu16(_mm_adds_epu16(s2, s4), _mm_adds_epu16(s6, s8));
                    b6 = _mm_srli_epi16(b6, 1);

                    _mm_storeu_si128((__m128i*)(brow + N4), b4);
                    _mm_storeu_si128((__m128i*)(brow + N5), b5);
                    _mm_storeu_si128((__m128i*)(brow + N6), b6);
                }
            }
#endif

            for( ; i < N-1; i++, srow++, brow++ )
            {
                brow[0] = (ushort)(std::abs(srow[-1-bstep] - srow[-1+bstep]) +
                                   std::abs(srow[-bstep] - srow[+bstep])*2 +
                                   std::abs(srow[1-bstep] - srow[1+bstep]));
                brow[N] = (ushort)(std::abs(srow[-1-bstep] - srow[1-bstep]) +
                                   std::abs(srow[-1] - srow[1])*2 +
                                   std::abs(srow[-1+bstep] - srow[1+bstep]));
                brow[N2] = (ushort)(std::abs(srow[+1-bstep] - srow[-1+bstep])*2);
                brow[N3] = (ushort)(std::abs(srow[-1-bstep] - srow[1+bstep])*2);
                brow[N4] = (ushort)(brow[N2] + std::abs(srow[-bstep] - srow[-1]) +
                                    std::abs(srow[+bstep] - srow[1]));
                brow[N5] = (ushort)(brow[N3] + std::abs(srow[-bstep] - srow[1]) +
                                    std::abs(srow[+bstep] - srow[-1]));
                brow[N6] = (ushort)((srow[-bstep] + srow[-1] + srow[1] + srow[+bstep])>>1);
            }
        }

        const ushort* brow0 = buf + ((y - 2) % brows)*bufstep + 2;
        const ushort* brow1 = buf + ((y - 1) % brows)*bufstep + 2;
        const ushort* brow2 = buf + (y % brows)*bufstep + 2;
        static const float scale[] = { 0.f, 0.5f, 0.25f, 0.1666666666667f, 0.125f, 0.1f, 0.08333333333f, 0.0714286f, 0.0625f };
        srow = bayer + y*bstep + 2;
        bool greenCell = greenCell0;

        i = 2;
#if CV_SSE2
        int limit = !haveSSE ? N-2 : greenCell ? std::min(3, N-2) : 2;
#else
        int limit = N - 2;
#endif

        do
        {
            for( ; i < limit; i++, srow++, brow0++, brow1++, brow2++, dstrow += 3 )
            {
                int gradN = brow0[0] + brow1[0];
                int gradS = brow1[0] + brow2[0];
                int gradW = brow1[N-1] + brow1[N];
                int gradE = brow1[N] + brow1[N+1];
                int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                int R, G, B;

                if( !greenCell )
                {
                    int gradNE = brow0[N4+1] + brow1[N4];
                    int gradSW = brow1[N4] + brow2[N4-1];
                    int gradNW = brow0[N5-1] + brow1[N5];
                    int gradSE = brow1[N5] + brow2[N5+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2] + srow[0];
                        Gs += srow[-bstep]*2;
                        Bs += srow[-bstep-1] + srow[-bstep+1];
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2] + srow[0];
                        Gs += srow[bstep]*2;
                        Bs += srow[bstep-1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-2] + srow[0];
                        Gs += srow[-1]*2;
                        Bs += srow[-bstep-1] + srow[bstep-1];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[2] + srow[0];
                        Gs += srow[1]*2;
                        Bs += srow[-bstep+1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+2] + srow[0];
                        Gs += brow0[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-2] + srow[0];
                        Gs += brow2[N6-1];
                        Bs += srow[bstep-1]*2;
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-2] + srow[0];
                        Gs += brow0[N6-1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+2] + srow[0];
                        Gs += brow2[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    R = srow[0];
                    G = R + cvRound((Gs - Rs)*scale[ng]);
                    B = R + cvRound((Bs - Rs)*scale[ng]);
                }
                else
                {
                    int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                    int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                    int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                    int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];

                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + MAX(maxGrad/2, 1);

                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-bstep*2+1];
                        Gs += srow[-bstep*2] + srow[0];
                        Bs += srow[-bstep]*2;
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2-1] + srow[bstep*2+1];
                        Gs += srow[bstep*2] + srow[0];
                        Bs += srow[bstep]*2;
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-1]*2;
                        Gs += srow[-2] + srow[0];
                        Bs += srow[-bstep-2]+srow[bstep-2];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[1]*2;
                        Gs += srow[2] + srow[0];
                        Bs += srow[-bstep+2]+srow[bstep+2];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+1] + srow[1];
                        Gs += srow[-bstep+1]*2;
                        Bs += srow[-bstep] + srow[-bstep+2];
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-1] + srow[-1];
                        Gs += srow[bstep-1]*2;
                        Bs += srow[bstep] + srow[bstep-2];
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-1];
                        Gs += srow[-bstep-1]*2;
                        Bs += srow[-bstep-2]+srow[-bstep];
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+1] + srow[1];
                        Gs += srow[bstep+1]*2;
                        Bs += srow[bstep+2]+srow[bstep];
                        ng++;
                    }
                    G = srow[0];
                    R = G + cvRound((Rs - Gs)*scale[ng]);
                    B = G + cvRound((Bs - Gs)*scale[ng]);
                }
                dstrow[blueIdx] = CV_CAST_8U(B);
                dstrow[1] = CV_CAST_8U(G);
                dstrow[blueIdx^2] = CV_CAST_8U(R);
                greenCell = !greenCell;
            }

#if CV_SSE2
            if( !haveSSE )
                break;

            __m128i emask    = _mm_set1_epi32(0x0000ffff),
                    omask    = _mm_set1_epi32(0xffff0000),
                    z        = _mm_setzero_si128(),
                    one      = _mm_set1_epi16(1);
            __m128 _0_5      = _mm_set1_ps(0.5f);

            #define _mm_merge_epi16(a, b) _mm_or_si128(_mm_and_si128(a, emask), _mm_and_si128(b, omask)) //(aA_aA_aA_aA) * (bB_bB_bB_bB) => (bA_bA_bA_bA)
            #define _mm_cvtloepi16_ps(a)  _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(a,a), 16))   //(1,2,3,4,5,6,7,8) => (1f,2f,3f,4f)
            #define _mm_cvthiepi16_ps(a)  _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(a,a), 16))   //(1,2,3,4,5,6,7,8) => (5f,6f,7f,8f)
            #define _mm_loadl_u8_s16(ptr, offset) _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)((ptr) + (offset))), z) //load 8 uchars to 8 shorts

            // process 8 pixels at once
            for( ; i <= N - 10; i += 8, srow += 8, brow0 += 8, brow1 += 8, brow2 += 8 )
            {
                //int gradN = brow0[0] + brow1[0];
                __m128i gradN = _mm_adds_epi16(_mm_loadu_si128((__m128i*)brow0), _mm_loadu_si128((__m128i*)brow1));

                //int gradS = brow1[0] + brow2[0];
                __m128i gradS = _mm_adds_epi16(_mm_loadu_si128((__m128i*)brow1), _mm_loadu_si128((__m128i*)brow2));

                //int gradW = brow1[N-1] + brow1[N];
                __m128i gradW = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N-1)), _mm_loadu_si128((__m128i*)(brow1+N)));

                //int gradE = brow1[N+1] + brow1[N];
                __m128i gradE = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N+1)), _mm_loadu_si128((__m128i*)(brow1+N)));

                //int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                //int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                __m128i minGrad = _mm_min_epi16(_mm_min_epi16(gradN, gradS), _mm_min_epi16(gradW, gradE));
                __m128i maxGrad = _mm_max_epi16(_mm_max_epi16(gradN, gradS), _mm_max_epi16(gradW, gradE));

                __m128i grad0, grad1;

                //int gradNE = brow0[N4+1] + brow1[N4];
                //int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                grad0 = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow0+N4+1)), _mm_loadu_si128((__m128i*)(brow1+N4)));
                grad1 = _mm_adds_epi16( _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow0+N2)), _mm_loadu_si128((__m128i*)(brow0+N2+1))),
                                        _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N2)), _mm_loadu_si128((__m128i*)(brow1+N2+1))));
                __m128i gradNE = _mm_merge_epi16(grad0, grad1);

                //int gradSW = brow1[N4] + brow2[N4-1];
                //int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                grad0 = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow2+N4-1)), _mm_loadu_si128((__m128i*)(brow1+N4)));
                grad1 = _mm_adds_epi16(_mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow2+N2)), _mm_loadu_si128((__m128i*)(brow2+N2-1))),
                                       _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N2)), _mm_loadu_si128((__m128i*)(brow1+N2-1))));
                __m128i gradSW = _mm_merge_epi16(grad0, grad1);

                minGrad = _mm_min_epi16(_mm_min_epi16(minGrad, gradNE), gradSW);
                maxGrad = _mm_max_epi16(_mm_max_epi16(maxGrad, gradNE), gradSW);

                //int gradNW = brow0[N5-1] + brow1[N5];
                //int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                grad0 = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow0+N5-1)), _mm_loadu_si128((__m128i*)(brow1+N5)));
                grad1 = _mm_adds_epi16(_mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow0+N3)), _mm_loadu_si128((__m128i*)(brow0+N3-1))),
                                       _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N3)), _mm_loadu_si128((__m128i*)(brow1+N3-1))));
                __m128i gradNW = _mm_merge_epi16(grad0, grad1);

                //int gradSE = brow1[N5] + brow2[N5+1];
                //int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];
                grad0 = _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow2+N5+1)), _mm_loadu_si128((__m128i*)(brow1+N5)));
                grad1 = _mm_adds_epi16(_mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow2+N3)), _mm_loadu_si128((__m128i*)(brow2+N3+1))),
                                       _mm_adds_epi16(_mm_loadu_si128((__m128i*)(brow1+N3)), _mm_loadu_si128((__m128i*)(brow1+N3+1))));
                __m128i gradSE = _mm_merge_epi16(grad0, grad1);

                minGrad = _mm_min_epi16(_mm_min_epi16(minGrad, gradNW), gradSE);
                maxGrad = _mm_max_epi16(_mm_max_epi16(maxGrad, gradNW), gradSE);

                //int T = minGrad + maxGrad/2;
                __m128i T = _mm_adds_epi16(_mm_max_epi16(_mm_srli_epi16(maxGrad, 1), one), minGrad);

                __m128i RGs = z, GRs = z, Bs = z, ng = z;

                __m128i x0  = _mm_loadl_u8_s16(srow, +0          );
                __m128i x1  = _mm_loadl_u8_s16(srow, -1 - bstep  );
                __m128i x2  = _mm_loadl_u8_s16(srow, -1 - bstep*2);
                __m128i x3  = _mm_loadl_u8_s16(srow,    - bstep  );
                __m128i x4  = _mm_loadl_u8_s16(srow, +1 - bstep*2);
                __m128i x5  = _mm_loadl_u8_s16(srow, +1 - bstep  );
                __m128i x6  = _mm_loadl_u8_s16(srow, +2 - bstep  );
                __m128i x7  = _mm_loadl_u8_s16(srow, +1          );
                __m128i x8  = _mm_loadl_u8_s16(srow, +2 + bstep  );
                __m128i x9  = _mm_loadl_u8_s16(srow, +1 + bstep  );
                __m128i x10 = _mm_loadl_u8_s16(srow, +1 + bstep*2);
                __m128i x11 = _mm_loadl_u8_s16(srow,    + bstep  );
                __m128i x12 = _mm_loadl_u8_s16(srow, -1 + bstep*2);
                __m128i x13 = _mm_loadl_u8_s16(srow, -1 + bstep  );
                __m128i x14 = _mm_loadl_u8_s16(srow, -2 + bstep  );
                __m128i x15 = _mm_loadl_u8_s16(srow, -1          );
                __m128i x16 = _mm_loadl_u8_s16(srow, -2 - bstep  );

                __m128i t0, t1, mask;

                // gradN ***********************************************
                mask = _mm_cmpgt_epi16(T, gradN); // mask = T>gradN
                ng = _mm_sub_epi16(ng, mask);     // ng += (T>gradN)

                t0 = _mm_slli_epi16(x3, 1);                                 // srow[-bstep]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, -bstep*2), x0);  // srow[-bstep*2] + srow[0]

                // RGs += (srow[-bstep*2] + srow[0]) * (T>gradN)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(t1, mask));
                // GRs += {srow[-bstep]*2; (srow[-bstep*2-1] + srow[-bstep*2+1])} * (T>gradN)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(t0, _mm_adds_epi16(x2,x4)), mask));
                // Bs  += {(srow[-bstep-1]+srow[-bstep+1]); srow[-bstep]*2 } * (T>gradN)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epi16(x1,x5), t0), mask));

                // gradNE **********************************************
                mask = _mm_cmpgt_epi16(T, gradNE); // mask = T>gradNE
                ng = _mm_sub_epi16(ng, mask);      // ng += (T>gradNE)

                t0 = _mm_slli_epi16(x5, 1);                                    // srow[-bstep+1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, -bstep*2+2), x0);   // srow[-bstep*2+2] + srow[0]

                // RGs += {(srow[-bstep*2+2] + srow[0]); srow[-bstep+1]*2} * (T>gradNE)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                // GRs += {brow0[N6+1]; (srow[-bstep*2+1] + srow[1])} * (T>gradNE)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow0+N6+1)), _mm_adds_epi16(x4,x7)), mask));
                // Bs  += {srow[-bstep+1]*2; (srow[-bstep] + srow[-bstep+2])}  * (T>gradNE)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(t0,_mm_adds_epi16(x3,x6)), mask));

                // gradE ***********************************************
                mask = _mm_cmpgt_epi16(T, gradE);  // mask = T>gradE
                ng = _mm_sub_epi16(ng, mask);      // ng += (T>gradE)

                t0 = _mm_slli_epi16(x7, 1);                         // srow[1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, 2), x0); // srow[2] + srow[0]

                // RGs += (srow[2] + srow[0]) * (T>gradE)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(t1, mask));
                // GRs += (srow[1]*2) * (T>gradE)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(t0, mask));
                // Bs  += {(srow[-bstep+1]+srow[bstep+1]); (srow[-bstep+2]+srow[bstep+2])} * (T>gradE)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epi16(x5,x9), _mm_adds_epi16(x6,x8)), mask));

                // gradSE **********************************************
                mask = _mm_cmpgt_epi16(T, gradSE);  // mask = T>gradSE
                ng = _mm_sub_epi16(ng, mask);       // ng += (T>gradSE)

                t0 = _mm_slli_epi16(x9, 1);                                 // srow[bstep+1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, bstep*2+2), x0); // srow[bstep*2+2] + srow[0]

                // RGs += {(srow[bstep*2+2] + srow[0]); srow[bstep+1]*2} * (T>gradSE)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                // GRs += {brow2[N6+1]; (srow[1]+srow[bstep*2+1])} * (T>gradSE)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow2+N6+1)), _mm_adds_epi16(x7,x10)), mask));
                // Bs  += {srow[-bstep+1]*2; (srow[bstep+2]+srow[bstep])} * (T>gradSE)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_slli_epi16(x5, 1), _mm_adds_epi16(x8,x11)), mask));

                // gradS ***********************************************
                mask = _mm_cmpgt_epi16(T, gradS);  // mask = T>gradS
                ng = _mm_sub_epi16(ng, mask);      // ng += (T>gradS)

                t0 = _mm_slli_epi16(x11, 1);                             // srow[bstep]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow,bstep*2), x0); // srow[bstep*2]+srow[0]

                // RGs += (srow[bstep*2]+srow[0]) * (T>gradS)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(t1, mask));
                // GRs += {srow[bstep]*2; (srow[bstep*2+1]+srow[bstep*2-1])} * (T>gradS)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(t0, _mm_adds_epi16(x10,x12)), mask));
                // Bs  += {(srow[bstep+1]+srow[bstep-1]); srow[bstep]*2} * (T>gradS)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epi16(x9,x13), t0), mask));

                // gradSW **********************************************
                mask = _mm_cmpgt_epi16(T, gradSW);  // mask = T>gradSW
                ng = _mm_sub_epi16(ng, mask);       // ng += (T>gradSW)

                t0 = _mm_slli_epi16(x13, 1);                                // srow[bstep-1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, bstep*2-2), x0); // srow[bstep*2-2]+srow[0]

                // RGs += {(srow[bstep*2-2]+srow[0]); srow[bstep-1]*2} * (T>gradSW)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                // GRs += {brow2[N6-1]; (srow[bstep*2-1]+srow[-1])} * (T>gradSW)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow2+N6-1)), _mm_adds_epi16(x12,x15)), mask));
                // Bs  += {srow[bstep-1]*2; (srow[bstep]+srow[bstep-2])} * (T>gradSW)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(t0,_mm_adds_epi16(x11,x14)), mask));

                // gradW ***********************************************
                mask = _mm_cmpgt_epi16(T, gradW);  // mask = T>gradW
                ng = _mm_sub_epi16(ng, mask);      // ng += (T>gradW)

                t0 = _mm_slli_epi16(x15, 1);                         // srow[-1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow, -2), x0); // srow[-2]+srow[0]

                // RGs += (srow[-2]+srow[0]) * (T>gradW)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(t1, mask));
                // GRs += (srow[-1]*2) * (T>gradW)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(t0, mask));
                // Bs  += {(srow[-bstep-1]+srow[bstep-1]); (srow[bstep-2]+srow[-bstep-2])} * (T>gradW)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epi16(x1,x13), _mm_adds_epi16(x14,x16)), mask));

                // gradNW **********************************************
                mask = _mm_cmpgt_epi16(T, gradNW);  // mask = T>gradNW
                ng = _mm_sub_epi16(ng, mask);       // ng += (T>gradNW)

                t0 = _mm_slli_epi16(x1, 1);                                 // srow[-bstep-1]*2
                t1 = _mm_adds_epi16(_mm_loadl_u8_s16(srow,-bstep*2-2), x0); // srow[-bstep*2-2]+srow[0]

                // RGs += {(srow[-bstep*2-2]+srow[0]); srow[-bstep-1]*2} * (T>gradNW)
                RGs = _mm_adds_epi16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                // GRs += {brow0[N6-1]; (srow[-bstep*2-1]+srow[-1])} * (T>gradNW)
                GRs = _mm_adds_epi16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow0+N6-1)), _mm_adds_epi16(x2,x15)), mask));
                // Bs  += {srow[-bstep-1]*2; (srow[-bstep]+srow[-bstep-2])} * (T>gradNW)
                Bs  = _mm_adds_epi16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_slli_epi16(x5, 1),_mm_adds_epi16(x3,x16)), mask));

                __m128 ngf0 = _mm_div_ps(_0_5, _mm_cvtloepi16_ps(ng));
                __m128 ngf1 = _mm_div_ps(_0_5, _mm_cvthiepi16_ps(ng));

                // now interpolate r, g & b
                t0 = _mm_subs_epi16(GRs, RGs);
                t1 = _mm_subs_epi16(Bs, RGs);

                t0 = _mm_add_epi16(x0, _mm_packs_epi32(
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtloepi16_ps(t0), ngf0)),
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvthiepi16_ps(t0), ngf1))));

                t1 = _mm_add_epi16(x0, _mm_packs_epi32(
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtloepi16_ps(t1), ngf0)),
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvthiepi16_ps(t1), ngf1))));

                x1 = _mm_merge_epi16(x0, t0);
                x2 = _mm_merge_epi16(t0, x0);

                uchar R[8], G[8], B[8];

                _mm_storel_epi64(blueIdx ? (__m128i*)B : (__m128i*)R, _mm_packus_epi16(x1, z));
                _mm_storel_epi64((__m128i*)G, _mm_packus_epi16(x2, z));
                _mm_storel_epi64(blueIdx ? (__m128i*)R : (__m128i*)B, _mm_packus_epi16(t1, z));

                for( int j = 0; j < 8; j++, dstrow += 3 )
                {
                    dstrow[0] = B[j]; dstrow[1] = G[j]; dstrow[2] = R[j];
                }
            }
#endif

            limit = N - 2;
        }
        while( i < N - 2 );

        for( i = 0; i < 6; i++ )
        {
            dst[dststep*y + 5 - i] = dst[dststep*y + 8 - i];
            dst[dststep*y + (N - 2)*3 + i] = dst[dststep*y + (N - 3)*3 + i];
        }

        greenCell0 = !greenCell0;
        blueIdx ^= 2;
    }

    for( i = 0; i < size.width*3; i++ )
    {
        dst[i] = dst[i + dststep] = dst[i + dststep*2];
        dst[i + dststep*(size.height-4)] =
        dst[i + dststep*(size.height-3)] =
        dst[i + dststep*(size.height-2)] =
        dst[i + dststep*(size.height-1)] = dst[i + dststep*(size.height-5)];
    }
}

///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
const int ITUR_BT_601_CRY =  269484;
const int ITUR_BT_601_CGY =  528482;
const int ITUR_BT_601_CBY =  102760;
const int ITUR_BT_601_CRU = -155188;
const int ITUR_BT_601_CGU = -305135;
const int ITUR_BT_601_CBU =  460324;
const int ITUR_BT_601_CGV = -385875;
const int ITUR_BT_601_CBV = -74448;

template<int bIdx, int uIdx>
struct YUV420sp2RGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *muv;
    int width, stride;

    YUV420sp2RGB888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _uv)
        : dst(_dst), my1(_y1), muv(_uv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

#ifdef HAVE_TEGRA_OPTIMIZATION
        if(tegra::cvtYUV4202RGB(bIdx, uIdx, 3, y1, uv, stride, dst->ptr<uchar>(rangeBegin), dst->step, rangeEnd - rangeBegin, dst->cols))
            return;
#endif

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 6, row2 += 6)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx>
struct YUV420sp2RGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *muv;
    int width, stride;

    YUV420sp2RGBA8888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _uv)
        : dst(_dst), my1(_y1), muv(_uv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

#ifdef HAVE_TEGRA_OPTIMIZATION
        if(tegra::cvtYUV4202RGB(bIdx, uIdx, 4, y1, uv, stride, dst->ptr<uchar>(rangeBegin), dst->step, rangeEnd - rangeBegin, dst->cols))
            return;
#endif

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 8, row2 += 8)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *mu, *mv;
    int width, stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGB888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst(_dst), my1(_y1), mu(_u), mv(_v), width(_dst->cols), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        const int rangeBegin = range.start * 2;
        const int rangeEnd = range.end * 2;

        int uvsteps[2] = {width/2, stride - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 6, row2 += 6)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *mu, *mv;
    int width, stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGBA8888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst(_dst), my1(_y1), mu(_u), mv(_v), width(_dst->cols), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        int uvsteps[2] = {width/2, stride - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION (320*240)

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGB(Mat& _dst, int _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGB888Invoker<bIdx, uIdx> converter(&_dst, _stride, _y1,  _uv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGBA(Mat& _dst, int _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGBA8888Invoker<bIdx, uIdx> converter(&_dst, _stride, _y1,  _uv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx>
inline void cvtYUV420p2RGB(Mat& _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGB888Invoker<bIdx> converter(&_dst, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx>
inline void cvtYUV420p2RGBA(Mat& _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGBA8888Invoker<bIdx> converter(&_dst, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

///////////////////////////////////// RGB -> YUV420p /////////////////////////////////////

template<int bIdx>
struct RGB888toYUV420pInvoker: public ParallelLoopBody
{
    RGB888toYUV420pInvoker( const Mat& src, Mat* dst, const int uIdx )
        : src_(src),
          dst_(dst),
          uIdx_(uIdx) { }

    void operator()(const Range& rowRange) const
    {
        const int w = src_.cols;
        const int h = src_.rows;

        const int cn = src_.channels();
        for( int i = rowRange.start; i < rowRange.end; i++ )
        {
            const uchar* row0 = src_.ptr<uchar>(2 * i);
            const uchar* row1 = src_.ptr<uchar>(2 * i + 1);

            uchar* y = dst_->ptr<uchar>(2*i);
            uchar* u = dst_->ptr<uchar>(h + i/2) + (i % 2) * (w/2);
            uchar* v = dst_->ptr<uchar>(h + (i + h/2)/2) + ((i + h/2) % 2) * (w/2);
            if( uIdx_ == 2 ) std::swap(u, v);

            for( int j = 0, k = 0; j < w * cn; j += 2 * cn, k++ )
            {
                int r00 = row0[2-bIdx + j];      int g00 = row0[1 + j];      int b00 = row0[bIdx + j];
                int r01 = row0[2-bIdx + cn + j]; int g01 = row0[1 + cn + j]; int b01 = row0[bIdx + cn + j];
                int r10 = row1[2-bIdx + j];      int g10 = row1[1 + j];      int b10 = row1[bIdx + j];
                int r11 = row1[2-bIdx + cn + j]; int g11 = row1[1 + cn + j]; int b11 = row1[bIdx + cn + j];

                const int shifted16 = (16 << ITUR_BT_601_SHIFT);
                const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
                int y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
                int y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
                int y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
                int y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

                y[2*k + 0]            = saturate_cast<uchar>(y00 >> ITUR_BT_601_SHIFT);
                y[2*k + 1]            = saturate_cast<uchar>(y01 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_->step + 0] = saturate_cast<uchar>(y10 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_->step + 1] = saturate_cast<uchar>(y11 >> ITUR_BT_601_SHIFT);

                const int shifted128 = (128 << ITUR_BT_601_SHIFT);
                int u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
                int v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;

                u[k] = saturate_cast<uchar>(u00 >> ITUR_BT_601_SHIFT);
                v[k] = saturate_cast<uchar>(v00 >> ITUR_BT_601_SHIFT);
            }
        }
    }

    static bool isFit( const Mat& src )
    {
        return (src.total() >= 320*240);
    }

private:
    RGB888toYUV420pInvoker& operator=(const RGB888toYUV420pInvoker&);

    const Mat& src_;
    Mat* const dst_;
    const int uIdx_;
};

template<int bIdx, int uIdx>
static void cvtRGBtoYUV420p(const Mat& src, Mat& dst)
{
    RGB888toYUV420pInvoker<bIdx> colorConverter(src, &dst, uIdx);
    if( RGB888toYUV420pInvoker<bIdx>::isFit(src) )
        parallel_for_(Range(0, src.rows/2), colorConverter);
    else
        colorConverter(Range(0, src.rows/2));
}

///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* src;
    int width, stride;

    YUV422toRGB888Invoker(Mat* _dst, int _stride, const uchar* _yuv)
        : dst(_dst), src(_yuv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src + rangeBegin * stride;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += stride)
        {
            uchar* row = dst->ptr<uchar>(j);

            for (int i = 0; i < 2 * width; i += 4, row += 6)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* src;
    int width, stride;

    YUV422toRGBA8888Invoker(Mat* _dst, int _stride, const uchar* _yuv)
        : dst(_dst), src(_yuv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src + rangeBegin * stride;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += stride)
        {
            uchar* row = dst->ptr<uchar>(j);

            for (int i = 0; i < 2 * width; i += 4, row += 8)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row[3]      = uchar(0xff);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGB(Mat& _dst, int _stride, const uchar* _yuv)
{
    YUV422toRGB888Invoker<bIdx, uIdx, yIdx> converter(&_dst, _stride, _yuv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, _dst.rows), converter);
    else
        converter(Range(0, _dst.rows));
}

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGBA(Mat& _dst, int _stride, const uchar* _yuv)
{
    YUV422toRGBA8888Invoker<bIdx, uIdx, yIdx> converter(&_dst, _stride, _yuv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, _dst.rows), converter);
    else
        converter(Range(0, _dst.rows));
}

/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

template<typename _Tp>
struct RGBA2mRGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val  = ColorChannel<_Tp>::max();
        _Tp half_val = ColorChannel<_Tp>::half();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;

            *dst++ = (v0 * v3 + half_val) / max_val;
            *dst++ = (v1 * v3 + half_val) / max_val;
            *dst++ = (v2 * v3 + half_val) / max_val;
            *dst++ = v3;
        }
    }
};


template<typename _Tp>
struct mRGBA2RGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val = ColorChannel<_Tp>::max();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;
            _Tp v3_half = v3 / 2;

            *dst++ = (v3==0)? 0 : (v0 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v1 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v2 * max_val + v3_half) / v3;
            *dst++ = v3;
        }
    }
};

}//namespace cv

//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////

void cv::cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    Mat src = _src.getMat(), dst;
    Size sz = src.size();
    int scn = src.channels(), depth = src.depth(), bidx;

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32F );

    switch( code )
    {
        case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
        case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
            CV_Assert( scn == 3 || scn == 4 );
            dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
            bidx = code == CV_BGR2BGRA || code == CV_BGRA2BGR ? 0 : 2;

            _dst.create( sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2BGRA || code == CV_RGB2RGBA)
            {
                if ( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 0, 1, 2)) )
                    return;
            }
            else if( code == CV_BGRA2BGR )
            {
                if ( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiCopyAC4C3RTab[depth])) )
                    return;
            }
            else if( code == CV_BGR2RGBA )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGBA2BGR )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC4C3RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGB2BGR )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderFunctor(ippiSwapChannelsC3RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGBA2BGRA )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderFunctor(ippiSwapChannelsC4RTab[depth], 2, 1, 0)) )
                    return;
            }
#endif

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtBGR2RGB(src, dst, bidx))
#endif
                    CvtColorLoop(src, dst, RGB2RGB<uchar>(scn, dcn, bidx));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2RGB<ushort>(scn, dcn, bidx));
            else
                CvtColorLoop(src, dst, RGB2RGB<float>(scn, dcn, bidx));
            break;

        case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
        case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
            if(code == CV_BGR2BGR565 || code == CV_BGRA2BGR565 || code == CV_RGB2BGR565  || code == CV_RGBA2BGR565)
                if(tegra::cvtRGB2RGB565(src, dst, code == CV_RGB2BGR565 || code == CV_RGBA2BGR565 ? 0 : 2))
                    break;
#endif

            CvtColorLoop(src, dst, RGB2RGB5x5(scn,
                      code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
                      code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2,
                      code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                      code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5 // green bits
                                              ));
            break;

        case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
        case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
            if(dcn <= 0) dcn = (code==CV_BGR5652BGRA || code==CV_BGR5552BGRA || code==CV_BGR5652RGBA || code==CV_BGR5552RGBA) ? 4 : 3;
            CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            CvtColorLoop(src, dst, RGB5x52RGB(dcn,
                      code == CV_BGR5652BGR || code == CV_BGR5552BGR ||
                      code == CV_BGR5652BGRA || code == CV_BGR5552BGRA ? 0 : 2, // blue idx
                      code == CV_BGR5652BGR || code == CV_BGR5652RGB ||
                      code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5 // green bits
                      ));
            break;

        case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 1));
            dst = _dst.getMat();
/*
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPColor2GrayFunctor(ippiColor2GrayC3Tab[depth])) )
                    return;
            }
            else if( code == CV_RGB2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiRGB2GrayC3Tab[depth])) )
                    return;
            }
            else if( code == CV_BGRA2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPColor2GrayFunctor(ippiColor2GrayC4Tab[depth])) )
                    return;
            }
            else if( code == CV_RGBA2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiRGB2GrayC4Tab[depth])) )
                    return;
            }
#endif
*/
            bidx = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtRGB2Gray(src, dst, bidx))
#endif
                CvtColorLoop(src, dst, RGB2Gray<uchar>(scn, bidx, 0));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2Gray<ushort>(scn, bidx, 0));
            else
                CvtColorLoop(src, dst, RGB2Gray<float>(scn, bidx, 0));
            break;

        case CV_BGR5652GRAY: case CV_BGR5552GRAY:
            CV_Assert( scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_8UC1);
            dst = _dst.getMat();

            CvtColorLoop(src, dst, RGB5x52Gray(code == CV_BGR5652GRAY ? 6 : 5));
            break;

        case CV_GRAY2BGR: case CV_GRAY2BGRA:
            if( dcn <= 0 ) dcn = (code==CV_GRAY2BGRA) ? 4 : 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4));
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_GRAY2BGR )
            {
                if( CvtColorIPPLoop(src, dst, IPPGray2BGRFunctor(ippiCopyP3C3RTab[depth])) )
                    return;
            }
            else if( code == CV_GRAY2BGRA )
            {
                if( CvtColorIPPLoop(src, dst, IPPGray2BGRAFunctor(ippiCopyP3C3RTab[depth], ippiSwapChannelsC3C4RTab[depth], depth)) )
                    return;
            }
#endif


            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtGray2RGB(src, dst))
#endif
                CvtColorLoop(src, dst, Gray2RGB<uchar>(dcn));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, Gray2RGB<ushort>(dcn));
            else
                CvtColorLoop(src, dst, Gray2RGB<float>(dcn));
            break;

        case CV_GRAY2BGR565: case CV_GRAY2BGR555:
            CV_Assert( scn == 1 && depth == CV_8U );
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();

            CvtColorLoop(src, dst, Gray2RGB5x5(code == CV_GRAY2BGR565 ? 6 : 5));
            break;

        case CV_BGR2YCrCb: case CV_RGB2YCrCb:
        case CV_BGR2YUV: case CV_RGB2YUV:
            {
            CV_Assert( scn == 3 || scn == 4 );
            bidx = code == CV_BGR2YCrCb || code == CV_BGR2YUV ? 0 : 2;
            static const float yuv_f[] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
            static const int yuv_i[] = { B2Y, G2Y, R2Y, 8061, 14369 };
            const float* coeffs_f = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_f;
            const int* coeffs_i = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_i;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if((code == CV_RGB2YCrCb || code == CV_BGR2YCrCb) && tegra::cvtRGB2YCrCb(src, dst, bidx))
                    break;
#endif
                CvtColorLoop(src, dst, RGB2YCrCb_i<uchar>(scn, bidx, coeffs_i));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2YCrCb_i<ushort>(scn, bidx, coeffs_i));
            else
                CvtColorLoop(src, dst, RGB2YCrCb_f<float>(scn, bidx, coeffs_f));
            }
            break;

        case CV_YCrCb2BGR: case CV_YCrCb2RGB:
        case CV_YUV2BGR: case CV_YUV2RGB:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            bidx = code == CV_YCrCb2BGR || code == CV_YUV2BGR ? 0 : 2;
            static const float yuv_f[] = { 2.032f, -0.395f, -0.581f, 1.140f };
            static const int yuv_i[] = { 33292, -6472, -9519, 18678 };
            const float* coeffs_f = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_f;
            const int* coeffs_i = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_i;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            if( depth == CV_8U )
                CvtColorLoop(src, dst, YCrCb2RGB_i<uchar>(dcn, bidx, coeffs_i));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, YCrCb2RGB_i<ushort>(dcn, bidx, coeffs_i));
            else
                CvtColorLoop(src, dst, YCrCb2RGB_f<float>(dcn, bidx, coeffs_f));
            }
            break;

        case CV_BGR2XYZ: case CV_RGB2XYZ:
            CV_Assert( scn == 3 || scn == 4 );
            bidx = code == CV_BGR2XYZ ? 0 : 2;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2XYZ && scn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_BGR2XYZ && scn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_RGB2XYZ && scn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2XYZTab[depth])) )
                    return;
            }
            else if( code == CV_RGB2XYZ && scn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 0, 1, 2, depth)) )
                    return;
            }
#endif

            if( depth == CV_8U )
                CvtColorLoop(src, dst, RGB2XYZ_i<uchar>(scn, bidx, 0));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2XYZ_i<ushort>(scn, bidx, 0));
            else
                CvtColorLoop(src, dst, RGB2XYZ_f<float>(scn, bidx, 0));
            break;

        case CV_XYZ2BGR: case CV_XYZ2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            bidx = code == CV_XYZ2BGR ? 0 : 2;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_XYZ2BGR && dcn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_XYZ2BGR && dcn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            if( code == CV_XYZ2RGB && dcn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiXYZ2RGBTab[depth])) )
                    return;
            }
            else if( code == CV_XYZ2RGB && dcn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return;
            }
#endif

            if( depth == CV_8U )
                CvtColorLoop(src, dst, XYZ2RGB_i<uchar>(dcn, bidx, 0));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, XYZ2RGB_i<ushort>(dcn, bidx, 0));
            else
                CvtColorLoop(src, dst, XYZ2RGB_f<float>(dcn, bidx, 0));
            break;

        case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
        case CV_BGR2HLS: case CV_RGB2HLS: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
            {
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_BGR2HSV || code == CV_BGR2HLS ||
                code == CV_BGR2HSV_FULL || code == CV_BGR2HLS_FULL ? 0 : 2;
            int hrange = depth == CV_32F ? 360 : code == CV_BGR2HSV || code == CV_RGB2HSV ||
                code == CV_BGR2HLS || code == CV_RGB2HLS ? 180 : 256;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( depth == CV_8U || depth == CV_16U )
            {
                if( code == CV_BGR2HSV_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_BGR2HSV_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_RGB2HSV_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2HSVTab[depth])) )
                        return;
                }
                else if( code == CV_RGB2HSV_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                else if( code == CV_BGR2HLS_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_BGR2HLS_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_RGB2HLS_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2HLSTab[depth])) )
                        return;
                }
                else if( code == CV_RGB2HLS_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
#endif

            if( code == CV_BGR2HSV || code == CV_RGB2HSV ||
                code == CV_BGR2HSV_FULL || code == CV_RGB2HSV_FULL )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(tegra::cvtRGB2HSV(src, dst, bidx, hrange))
                    break;
#endif
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2HSV_b(scn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, RGB2HSV_f(scn, bidx, (float)hrange));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2HLS_b(scn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, RGB2HLS_f(scn, bidx, (float)hrange));
            }
            }
            break;

        case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
        case CV_HLS2BGR: case CV_HLS2RGB: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_HSV2BGR || code == CV_HLS2BGR ||
                code == CV_HSV2BGR_FULL || code == CV_HLS2BGR_FULL ? 0 : 2;
            int hrange = depth == CV_32F ? 360 : code == CV_HSV2BGR || code == CV_HSV2RGB ||
                code == CV_HLS2BGR || code == CV_HLS2RGB ? 180 : 255;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( depth == CV_8U || depth == CV_16U )
            {
                if( code == CV_HSV2BGR_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HSV2BGR_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HSV2RGB_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiHSV2RGBTab[depth])) )
                        return;
                }
                else if( code == CV_HSV2RGB_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                else if( code == CV_HLS2BGR_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HLS2BGR_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HLS2RGB_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiHLS2RGBTab[depth])) )
                        return;
                }
                else if( code == CV_HLS2RGB_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
#endif

            if( code == CV_HSV2BGR || code == CV_HSV2RGB ||
                code == CV_HSV2BGR_FULL || code == CV_HSV2RGB_FULL )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, HSV2RGB_b(dcn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, HSV2RGB_f(dcn, bidx, (float)hrange));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, HLS2RGB_b(dcn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, HLS2RGB_f(dcn, bidx, (float)hrange));
            }
            }
            break;

        case CV_BGR2Lab: case CV_RGB2Lab: case CV_LBGR2Lab: case CV_LRGB2Lab:
        case CV_BGR2Luv: case CV_RGB2Luv: case CV_LBGR2Luv: case CV_LRGB2Luv:
            {
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_BGR2Lab || code == CV_BGR2Luv ||
                   code == CV_LBGR2Lab || code == CV_LBGR2Luv ? 0 : 2;
            bool srgb = code == CV_BGR2Lab || code == CV_RGB2Lab ||
                        code == CV_BGR2Luv || code == CV_RGB2Luv;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

            if( code == CV_BGR2Lab || code == CV_RGB2Lab ||
                code == CV_LBGR2Lab || code == CV_LRGB2Lab )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2Lab_b(scn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, RGB2Lab_f(scn, bidx, 0, 0, srgb));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2Luv_b(scn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, RGB2Luv_f(scn, bidx, 0, 0, srgb));
            }
            }
            break;

        case CV_Lab2BGR: case CV_Lab2RGB: case CV_Lab2LBGR: case CV_Lab2LRGB:
        case CV_Luv2BGR: case CV_Luv2RGB: case CV_Luv2LBGR: case CV_Luv2LRGB:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_Lab2BGR || code == CV_Luv2BGR ||
                   code == CV_Lab2LBGR || code == CV_Luv2LBGR ? 0 : 2;
            bool srgb = code == CV_Lab2BGR || code == CV_Lab2RGB ||
                    code == CV_Luv2BGR || code == CV_Luv2RGB;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            if( code == CV_Lab2BGR || code == CV_Lab2RGB ||
                code == CV_Lab2LBGR || code == CV_Lab2LRGB )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, Lab2RGB_b(dcn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, Lab2RGB_f(dcn, bidx, 0, 0, srgb));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, Luv2RGB_b(dcn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, Luv2RGB_f(dcn, bidx, 0, 0, srgb));
            }
            }
            break;

        case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
            if(dcn <= 0) dcn = 1;
            CV_Assert( scn == 1 && dcn == 1 );

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            if( depth == CV_8U )
                Bayer2Gray_<uchar, SIMDBayerInterpolator_8u>(src, dst, code);
            else if( depth == CV_16U )
                Bayer2Gray_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst, code);
            else
                CV_Error(CV_StsUnsupportedFormat, "Bayer->Gray demosaicing only supports 8u and 16u types");
            break;

        case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
            {
                if (dcn <= 0)
                    dcn = 3;
                CV_Assert( scn == 1 && dcn == 3 );

                _dst.create(sz, CV_MAKE_TYPE(depth, dcn));
                Mat dst_ = _dst.getMat();

                if( code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ||
                    code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
                {
                    if( depth == CV_8U )
                        Bayer2RGB_<uchar, SIMDBayerInterpolator_8u>(src, dst_, code);
                    else if( depth == CV_16U )
                        Bayer2RGB_<ushort, SIMDBayerStubInterpolator_<ushort> >(src, dst_, code);
                    else
                        CV_Error(CV_StsUnsupportedFormat, "Bayer->RGB demosaicing only supports 8u and 16u types");
                }
                else
                {
                    CV_Assert( depth == CV_8U );
                    Bayer2RGB_VNG_8u(src, dst_, code);
                }
            }
            break;
        case CV_YUV2BGR_NV21:  case CV_YUV2RGB_NV21:  case CV_YUV2BGR_NV12:  case CV_YUV2RGB_NV12:
        case CV_YUV2BGRA_NV21: case CV_YUV2RGBA_NV21: case CV_YUV2BGRA_NV12: case CV_YUV2RGBA_NV12:
            {
                // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
                // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples

                if (dcn <= 0) dcn = (code==CV_YUV420sp2BGRA || code==CV_YUV420sp2RGBA || code==CV_YUV2BGRA_NV12 || code==CV_YUV2RGBA_NV12) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_NV21 || code==CV_YUV2BGRA_NV21 || code==CV_YUV2BGR_NV12 || code==CV_YUV2BGRA_NV12) ? 0 : 2;
                const int uIdx = (code==CV_YUV2BGR_NV21 || code==CV_YUV2BGRA_NV21 || code==CV_YUV2RGB_NV21 || code==CV_YUV2RGBA_NV21) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                int srcstep = (int)src.step;
                const uchar* y = src.ptr();
                const uchar* uv = y + srcstep * dstSz.height;

                switch(dcn*100 + bIdx * 10 + uIdx)
                {
                    case 300: cvtYUV420sp2RGB<0, 0> (dst, srcstep, y, uv); break;
                    case 301: cvtYUV420sp2RGB<0, 1> (dst, srcstep, y, uv); break;
                    case 320: cvtYUV420sp2RGB<2, 0> (dst, srcstep, y, uv); break;
                    case 321: cvtYUV420sp2RGB<2, 1> (dst, srcstep, y, uv); break;
                    case 400: cvtYUV420sp2RGBA<0, 0>(dst, srcstep, y, uv); break;
                    case 401: cvtYUV420sp2RGBA<0, 1>(dst, srcstep, y, uv); break;
                    case 420: cvtYUV420sp2RGBA<2, 0>(dst, srcstep, y, uv); break;
                    case 421: cvtYUV420sp2RGBA<2, 1>(dst, srcstep, y, uv); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2BGR_YV12: case CV_YUV2RGB_YV12: case CV_YUV2BGRA_YV12: case CV_YUV2RGBA_YV12:
        case CV_YUV2BGR_IYUV: case CV_YUV2RGB_IYUV: case CV_YUV2BGRA_IYUV: case CV_YUV2RGBA_IYUV:
            {
                //http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
                //http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes

                if (dcn <= 0) dcn = (code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12 || code==CV_YUV2RGBA_IYUV || code==CV_YUV2BGRA_IYUV) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_YV12 || code==CV_YUV2BGRA_YV12 || code==CV_YUV2BGR_IYUV || code==CV_YUV2BGRA_IYUV) ? 0 : 2;
                const int uIdx  = (code==CV_YUV2BGR_YV12 || code==CV_YUV2RGB_YV12 || code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                int srcstep = (int)src.step;
                const uchar* y = src.ptr();
                const uchar* u = y + srcstep * dstSz.height;
                const uchar* v = y + srcstep * (dstSz.height + dstSz.height/4) + (dstSz.width/2) * ((dstSz.height % 4)/2);

                int ustepIdx = 0;
                int vstepIdx = dstSz.height % 4 == 2 ? 1 : 0;

                if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); };

                switch(dcn*10 + bIdx)
                {
                    case 30: cvtYUV420p2RGB<0>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 32: cvtYUV420p2RGB<2>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 40: cvtYUV420p2RGBA<0>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 42: cvtYUV420p2RGBA<2>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2GRAY_420:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                src(Range(0, dstSz.height), Range::all()).copyTo(dst);
            }
            break;
        case CV_RGB2YUV_YV12: case CV_BGR2YUV_YV12: case CV_RGBA2YUV_YV12: case CV_BGRA2YUV_YV12:
        case CV_RGB2YUV_IYUV: case CV_BGR2YUV_IYUV: case CV_RGBA2YUV_IYUV: case CV_BGRA2YUV_IYUV:
            {
                if (dcn <= 0) dcn = 1;
                const int bIdx = (code == CV_BGR2YUV_IYUV || code == CV_BGRA2YUV_IYUV || code == CV_BGR2YUV_YV12 || code == CV_BGRA2YUV_YV12) ? 0 : 2;
                const int uIdx = (code == CV_BGR2YUV_IYUV || code == CV_BGRA2YUV_IYUV || code == CV_RGB2YUV_IYUV || code == CV_RGBA2YUV_IYUV) ? 1 : 2;

                CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );

                Size dstSz(sz.width, sz.height / 2 * 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                switch(bIdx + uIdx*10)
                {
                    case 10: cvtRGBtoYUV420p<0, 1>(src, dst); break;
                    case 12: cvtRGBtoYUV420p<2, 1>(src, dst); break;
                    case 20: cvtRGBtoYUV420p<0, 2>(src, dst); break;
                    case 22: cvtRGBtoYUV420p<2, 2>(src, dst); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2RGB_UYVY: case CV_YUV2BGR_UYVY: case CV_YUV2RGBA_UYVY: case CV_YUV2BGRA_UYVY:
        case CV_YUV2RGB_YUY2: case CV_YUV2BGR_YUY2: case CV_YUV2RGB_YVYU: case CV_YUV2BGR_YVYU:
        case CV_YUV2RGBA_YUY2: case CV_YUV2BGRA_YUY2: case CV_YUV2RGBA_YVYU: case CV_YUV2BGRA_YVYU:
            {
                //http://www.fourcc.org/yuv.php#UYVY
                //http://www.fourcc.org/yuv.php#YUY2
                //http://www.fourcc.org/yuv.php#YVYU

                if (dcn <= 0) dcn = (code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY || code==CV_YUV2RGBA_YUY2 || code==CV_YUV2BGRA_YUY2 || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_UYVY || code==CV_YUV2BGRA_UYVY || code==CV_YUV2BGR_YUY2 || code==CV_YUV2BGRA_YUY2 || code==CV_YUV2BGR_YVYU || code==CV_YUV2BGRA_YVYU) ? 0 : 2;
                const int ycn  = (code==CV_YUV2RGB_UYVY || code==CV_YUV2BGR_UYVY || code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY) ? 1 : 0;
                const int uIdx = (code==CV_YUV2RGB_YVYU || code==CV_YUV2BGR_YVYU || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( scn == 2 && depth == CV_8U );

                _dst.create(sz, CV_8UC(dcn));
                dst = _dst.getMat();

                switch(dcn*1000 + bIdx*100 + uIdx*10 + ycn)
                {
                    case 3000: cvtYUV422toRGB<0,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3001: cvtYUV422toRGB<0,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3010: cvtYUV422toRGB<0,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3011: cvtYUV422toRGB<0,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3200: cvtYUV422toRGB<2,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3201: cvtYUV422toRGB<2,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3210: cvtYUV422toRGB<2,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3211: cvtYUV422toRGB<2,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4000: cvtYUV422toRGBA<0,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4001: cvtYUV422toRGBA<0,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4010: cvtYUV422toRGBA<0,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4011: cvtYUV422toRGBA<0,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4200: cvtYUV422toRGBA<2,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4201: cvtYUV422toRGBA<2,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4210: cvtYUV422toRGBA<2,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4211: cvtYUV422toRGBA<2,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2GRAY_UYVY: case CV_YUV2GRAY_YUY2:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( scn == 2 && depth == CV_8U );

                extractChannel(_src, _dst, code == CV_YUV2GRAY_UYVY ? 1 : 0);
            }
            break;
        case CV_RGBA2mRGBA:
            {
                if (dcn <= 0) dcn = 4;
                CV_Assert( scn == 4 && dcn == 4 );

                _dst.create(sz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                if( depth == CV_8U )
                {
                    CvtColorLoop(src, dst, RGBA2mRGBA<uchar>());
                } else {
                    CV_Error( CV_StsBadArg, "Unsupported image depth" );
                }
            }
            break;
        case CV_mRGBA2RGBA:
            {
                if (dcn <= 0) dcn = 4;
                CV_Assert( scn == 4 && dcn == 4 );

                _dst.create(sz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                if( depth == CV_8U )
                {
                    CvtColorLoop(src, dst, mRGBA2RGBA<uchar>());
                } else {
                    CV_Error( CV_StsBadArg, "Unsupported image depth" );
                }
            }
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }
}

CV_IMPL void
cvCvtColor( const CvArr* srcarr, CvArr* dstarr, int code )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    CV_Assert( src.depth() == dst.depth() );

    cv::cvtColor(src, dst, code, dst.channels());
    CV_Assert( dst.data == dst0.data );
}


/* End of file. */
