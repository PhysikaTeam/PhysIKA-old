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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::gemm(const GpuMat&, const GpuMat&, double, const GpuMat&, double, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::transpose(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::flip(const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::LUT(const GpuMat&, const Mat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitude(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::magnitudeSqr(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::phase(const GpuMat&, const GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }
void cv::gpu::cartToPolar(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }
void cv::gpu::polarToCart(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, bool, Stream&) { throw_nogpu(); }
void cv::gpu::normalize(const GpuMat&, GpuMat&, double, double, int, int, const GpuMat&) { throw_nogpu(); }
void cv::gpu::normalize(const GpuMat&, GpuMat&, double, double, int, int, const GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// gemm

void cv::gpu::gemm(const GpuMat& src1, const GpuMat& src2, double alpha, const GpuMat& src3, double beta, GpuMat& dst, int flags, Stream& stream)
{
#ifndef HAVE_CUBLAS
    (void)src1;
    (void)src2;
    (void)alpha;
    (void)src3;
    (void)beta;
    (void)dst;
    (void)flags;
    (void)stream;
    CV_Error(CV_StsNotImplemented, "The library was build without CUBLAS");
#else
    // CUBLAS works with column-major matrices

    CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_32FC2 || src1.type() == CV_64FC1 || src1.type() == CV_64FC2);
    CV_Assert(src2.type() == src1.type() && (src3.empty() || src3.type() == src1.type()));

    if (src1.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");
    }

    bool tr1 = (flags & GEMM_1_T) != 0;
    bool tr2 = (flags & GEMM_2_T) != 0;
    bool tr3 = (flags & GEMM_3_T) != 0;

    if (src1.type() == CV_64FC2)
    {
        if (tr1 || tr2 || tr3)
            CV_Error(CV_StsNotImplemented, "transpose operation doesn't implemented for CV_64FC2 type");
    }

    Size src1Size = tr1 ? Size(src1.rows, src1.cols) : src1.size();
    Size src2Size = tr2 ? Size(src2.rows, src2.cols) : src2.size();
    Size src3Size = tr3 ? Size(src3.rows, src3.cols) : src3.size();
    Size dstSize(src2Size.width, src1Size.height);

    CV_Assert(src1Size.width == src2Size.height);
    CV_Assert(src3.empty() || src3Size == dstSize);

    dst.create(dstSize, src1.type());

    if (beta != 0)
    {
        if (src3.empty())
        {
            if (stream)
                stream.enqueueMemSet(dst, Scalar::all(0));
            else
                dst.setTo(Scalar::all(0));
        }
        else
        {
            if (tr3)
            {
                transpose(src3, dst, stream);
            }
            else
            {
                if (stream)
                    stream.enqueueCopy(src3, dst);
                else
                    src3.copyTo(dst);
            }
        }
    }

    cublasHandle_t handle;
    cublasSafeCall( cublasCreate_v2(&handle) );

    cublasSafeCall( cublasSetStream_v2(handle, StreamAccessor::getStream(stream)) );

    cublasSafeCall( cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST) );

    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);

    const cuComplex alphacf = make_cuComplex(alphaf, 0);
    const cuComplex betacf = make_cuComplex(betaf, 0);

    const cuDoubleComplex alphac = make_cuDoubleComplex(alpha, 0);
    const cuDoubleComplex betac = make_cuDoubleComplex(beta, 0);

    cublasOperation_t transa = tr2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = tr1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    switch (src1.type())
    {
    case CV_32FC1:
        cublasSafeCall( cublasSgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphaf,
            src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
            src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
            &betaf,
            dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
        break;

    case CV_64FC1:
        cublasSafeCall( cublasDgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alpha,
            src2.ptr<double>(), static_cast<int>(src2.step / sizeof(double)),
            src1.ptr<double>(), static_cast<int>(src1.step / sizeof(double)),
            &beta,
            dst.ptr<double>(), static_cast<int>(dst.step / sizeof(double))) );
        break;

    case CV_32FC2:
        cublasSafeCall( cublasCgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphacf,
            src2.ptr<cuComplex>(), static_cast<int>(src2.step / sizeof(cuComplex)),
            src1.ptr<cuComplex>(), static_cast<int>(src1.step / sizeof(cuComplex)),
            &betacf,
            dst.ptr<cuComplex>(), static_cast<int>(dst.step / sizeof(cuComplex))) );
        break;

    case CV_64FC2:
        cublasSafeCall( cublasZgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphac,
            src2.ptr<cuDoubleComplex>(), static_cast<int>(src2.step / sizeof(cuDoubleComplex)),
            src1.ptr<cuDoubleComplex>(), static_cast<int>(src1.step / sizeof(cuDoubleComplex)),
            &betac,
            dst.ptr<cuDoubleComplex>(), static_cast<int>(dst.step / sizeof(cuDoubleComplex))) );
        break;
    }

    cublasSafeCall( cublasDestroy_v2(handle) );
#endif
}

////////////////////////////////////////////////////////////////////////
// transpose

void cv::gpu::transpose(const GpuMat& src, GpuMat& dst, Stream& s)
{
    CV_Assert(src.elemSize() == 1 || src.elemSize() == 4 || src.elemSize() == 8);

    dst.create( src.cols, src.rows, src.type() );

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src.elemSize() == 1)
    {
        NppStreamHandler h(stream);

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiTranspose_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step),
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );
    }
    else if (src.elemSize() == 4)
    {
        NppStStreamHandler h(stream);

        NcvSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        ncvSafeCall( nppiStTranspose_32u_C1R(const_cast<Ncv32u*>(src.ptr<Ncv32u>()), static_cast<int>(src.step),
            dst.ptr<Ncv32u>(), static_cast<int>(dst.step), sz) );
    }
    else // if (src.elemSize() == 8)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(CV_StsUnsupportedFormat, "The device doesn't support double");

        NppStStreamHandler h(stream);

        NcvSize32u sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        ncvSafeCall( nppiStTranspose_64u_C1R(const_cast<Ncv64u*>(src.ptr<Ncv64u>()), static_cast<int>(src.step),
            dst.ptr<Ncv64u>(), static_cast<int>(dst.step), sz) );
    }

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////
// flip

namespace
{
    template<int DEPTH> struct NppTypeTraits;
    template<> struct NppTypeTraits<CV_8U>  { typedef Npp8u npp_t; };
    template<> struct NppTypeTraits<CV_8S>  { typedef Npp8s npp_t; };
    template<> struct NppTypeTraits<CV_16U> { typedef Npp16u npp_t; };
    template<> struct NppTypeTraits<CV_16S> { typedef Npp16s npp_t; };
    template<> struct NppTypeTraits<CV_32S> { typedef Npp32s npp_t; };
    template<> struct NppTypeTraits<CV_32F> { typedef Npp32f npp_t; };
    template<> struct NppTypeTraits<CV_64F> { typedef Npp64f npp_t; };

    template <int DEPTH> struct NppMirrorFunc
    {
        typedef typename NppTypeTraits<DEPTH>::npp_t npp_t;

        typedef NppStatus (*func_t)(const npp_t* pSrc, int nSrcStep, npp_t* pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
    };

    template <int DEPTH, typename NppMirrorFunc<DEPTH>::func_t func> struct NppMirror
    {
        typedef typename NppMirrorFunc<DEPTH>::npp_t npp_t;

        static void call(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream)
        {
            NppStreamHandler h(stream);

            NppiSize sz;
            sz.width  = src.cols;
            sz.height = src.rows;

            nppSafeCall( func(src.ptr<npp_t>(), static_cast<int>(src.step),
                dst.ptr<npp_t>(), static_cast<int>(dst.step), sz,
                (flipCode == 0 ? NPP_HORIZONTAL_AXIS : (flipCode > 0 ? NPP_VERTICAL_AXIS : NPP_BOTH_AXIS))) );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    };
}

void cv::gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode, Stream& stream)
{
    typedef void (*func_t)(const GpuMat& src, GpuMat& dst, int flipCode, cudaStream_t stream);
    static const func_t funcs[6][4] =
    {
        {NppMirror<CV_8U, nppiMirror_8u_C1R>::call, 0, NppMirror<CV_8U, nppiMirror_8u_C3R>::call, NppMirror<CV_8U, nppiMirror_8u_C4R>::call},
        {0,0,0,0},
        {NppMirror<CV_16U, nppiMirror_16u_C1R>::call, 0, NppMirror<CV_16U, nppiMirror_16u_C3R>::call, NppMirror<CV_16U, nppiMirror_16u_C4R>::call},
        {0,0,0,0},
        {NppMirror<CV_32S, nppiMirror_32s_C1R>::call, 0, NppMirror<CV_32S, nppiMirror_32s_C3R>::call, NppMirror<CV_32S, nppiMirror_32s_C4R>::call},
        {NppMirror<CV_32F, nppiMirror_32f_C1R>::call, 0, NppMirror<CV_32F, nppiMirror_32f_C3R>::call, NppMirror<CV_32F, nppiMirror_32f_C4R>::call}
    };

    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32S || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);

    dst.create(src.size(), src.type());

    funcs[src.depth()][src.channels() - 1](src, dst, flipCode, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// LUT

namespace arithm
{
    void lut(PtrStepSzb src, uchar* lut, int lut_cn, PtrStepSzb dst, bool cc30, cudaStream_t stream);
}

void cv::gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& s)
{
    const int cn = src.channels();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 );
    CV_Assert( lut.depth() == CV_8U );
    CV_Assert( lut.channels() == 1 || lut.channels() == cn );
    CV_Assert( lut.rows * lut.cols == 256 && lut.isContinuous() );

    dst.create(src.size(), CV_MAKE_TYPE(lut.depth(), cn));

    GpuMat d_lut;
    d_lut.upload(Mat(1, 256, lut.type(), lut.data));

    int lut_cn = d_lut.channels();
    bool cc30 = deviceSupports(FEATURE_SET_COMPUTE_30);
    cudaStream_t stream = StreamAccessor::getStream(s);

    if (lut_cn == 1)
    {
        arithm::lut(src.reshape(1), d_lut.data, lut_cn, dst.reshape(1), cc30, stream);
    }
    else if (lut_cn == 3)
    {
        arithm::lut(src, d_lut.data, lut_cn, dst, cc30, stream);
    }
}

////////////////////////////////////////////////////////////////////////
// NPP magnitide

namespace
{
    typedef NppStatus (*nppMagnitude_t)(const Npp32fc* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    inline void npp_magnitude(const GpuMat& src, GpuMat& dst, nppMagnitude_t func, cudaStream_t stream)
    {
        CV_Assert(src.type() == CV_32FC2);

        dst.create(src.size(), CV_32FC1);

        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        nppSafeCall( func(src.ptr<Npp32fc>(), static_cast<int>(src.step), dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

void cv::gpu::magnitude(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    npp_magnitude(src, dst, nppiMagnitude_32fc32f_C1R, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    npp_magnitude(src, dst, nppiMagnitudeSqr_32fc32f_C1R, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// Polar <-> Cart

namespace cv { namespace gpu { namespace device
{
    namespace mathfunc
    {
        void cartToPolar_gpu(PtrStepSzf x, PtrStepSzf y, PtrStepSzf mag, bool magSqr, PtrStepSzf angle, bool angleInDegrees, cudaStream_t stream);
        void polarToCart_gpu(PtrStepSzf mag, PtrStepSzf angle, PtrStepSzf x, PtrStepSzf y, bool angleInDegrees, cudaStream_t stream);
    }
}}}

namespace
{
    inline void cartToPolar_caller(const GpuMat& x, const GpuMat& y, GpuMat* mag, bool magSqr, GpuMat* angle, bool angleInDegrees, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device::mathfunc;

        CV_Assert(x.size() == y.size() && x.type() == y.type());
        CV_Assert(x.depth() == CV_32F);

        if (mag)
            mag->create(x.size(), x.type());
        if (angle)
            angle->create(x.size(), x.type());

        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);
        GpuMat mag1cn = mag ? mag->reshape(1) : GpuMat();
        GpuMat angle1cn = angle ? angle->reshape(1) : GpuMat();

        cartToPolar_gpu(x1cn, y1cn, mag1cn, magSqr, angle1cn, angleInDegrees, stream);
    }

    inline void polarToCart_caller(const GpuMat& mag, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device::mathfunc;

        CV_Assert((mag.empty() || mag.size() == angle.size()) && mag.type() == angle.type());
        CV_Assert(mag.depth() == CV_32F);

        x.create(mag.size(), mag.type());
        y.create(mag.size(), mag.type());

        GpuMat mag1cn = mag.reshape(1);
        GpuMat angle1cn = angle.reshape(1);
        GpuMat x1cn = x.reshape(1);
        GpuMat y1cn = y.reshape(1);

        polarToCart_gpu(mag1cn, angle1cn, x1cn, y1cn, angleInDegrees, stream);
    }
}

void cv::gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& dst, Stream& stream)
{
    cartToPolar_caller(x, y, &dst, false, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& dst, Stream& stream)
{
    cartToPolar_caller(x, y, &dst, true, 0, false, StreamAccessor::getStream(stream));
}

void cv::gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, Stream& stream)
{
    cartToPolar_caller(x, y, 0, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& mag, GpuMat& angle, bool angleInDegrees, Stream& stream)
{
    cartToPolar_caller(x, y, &mag, false, &angle, angleInDegrees, StreamAccessor::getStream(stream));
}

void cv::gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, Stream& stream)
{
    polarToCart_caller(magnitude, angle, x, y, angleInDegrees, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// normalize

void cv::gpu::normalize(const GpuMat& src, GpuMat& dst, double a, double b, int norm_type, int dtype, const GpuMat& mask)
{
    GpuMat norm_buf;
    GpuMat cvt_buf;
    normalize(src, dst, a, b, norm_type, dtype, mask, norm_buf, cvt_buf);
}

void cv::gpu::normalize(const GpuMat& src, GpuMat& dst, double a, double b, int norm_type, int dtype, const GpuMat& mask, GpuMat& norm_buf, GpuMat& cvt_buf)
{
    double scale = 1, shift = 0;
    if (norm_type == NORM_MINMAX)
    {
        double smin = 0, smax = 0;
        double dmin = std::min(a, b), dmax = std::max(a, b);
        minMax(src, &smin, &smax, mask, norm_buf);
        scale = (dmax - dmin) * (smax - smin > numeric_limits<double>::epsilon() ? 1.0 / (smax - smin) : 0.0);
        shift = dmin - smin * scale;
    }
    else if (norm_type == NORM_L2 || norm_type == NORM_L1 || norm_type == NORM_INF)
    {
        scale = norm(src, norm_type, mask, norm_buf);
        scale = scale > numeric_limits<double>::epsilon() ? a / scale : 0.0;
        shift = 0;
    }
    else
    {
        CV_Error(CV_StsBadArg, "Unknown/unsupported norm type");
    }

    if (mask.empty())
    {
        src.convertTo(dst, dtype, scale, shift);
    }
    else
    {
        src.convertTo(cvt_buf, dtype, scale, shift);
        cvt_buf.copyTo(dst, mask);
    }
}

#endif /* !defined (HAVE_CUDA) */
