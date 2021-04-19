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

#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/type_traits.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"

namespace cv { namespace gpu { namespace device
{
    void writeScalar(const uchar*);
    void writeScalar(const schar*);
    void writeScalar(const ushort*);
    void writeScalar(const short int*);
    void writeScalar(const int*);
    void writeScalar(const float*);
    void writeScalar(const double*);
    void copyToWithMask_gpu(PtrStepSzb src, PtrStepSzb dst, size_t elemSize1, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream);
    void convert_gpu(PtrStepSzb, int, PtrStepSzb, int, double, double, cudaStream_t);
}}}

namespace cv { namespace gpu { namespace device
{
    template <typename T> struct shift_and_sizeof;
    template <> struct shift_and_sizeof<signed char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<unsigned char> { enum { shift = 0 }; };
    template <> struct shift_and_sizeof<short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<unsigned short> { enum { shift = 1 }; };
    template <> struct shift_and_sizeof<int> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<float> { enum { shift = 2 }; };
    template <> struct shift_and_sizeof<double> { enum { shift = 3 }; };

    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// CopyTo /////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template <typename T> void copyToWithMask(PtrStepSzb src, PtrStepSzb dst, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream)
    {
        if (colorMask)
            cv::gpu::device::transform((PtrStepSz<T>)src, (PtrStepSz<T>)dst, identity<T>(), SingleMask(mask), stream);
        else
            cv::gpu::device::transform((PtrStepSz<T>)src, (PtrStepSz<T>)dst, identity<T>(), SingleMaskChannels(mask, cn), stream);
    }

    void copyToWithMask_gpu(PtrStepSzb src, PtrStepSzb dst, size_t elemSize1, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, int cn, PtrStepSzb mask, bool colorMask, cudaStream_t stream);

        static func_t tab[] =
        {
            0,
            copyToWithMask<unsigned char>,
            copyToWithMask<unsigned short>,
            0,
            copyToWithMask<int>,
            0,
            0,
            0,
            copyToWithMask<double>
        };

        tab[elemSize1](src, dst, cn, mask, colorMask, stream);
    }

    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// SetTo //////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void set_to_without_mask(PtrStepSz<T> mat, typename TypeVec<T, 4>::vec_type val, int channels)
    {
        const int y = blockIdx.x * blockDim.y + threadIdx.y;

        if (y < mat.rows)
        {
            const T vals[] = {
                val.x, val.y, val.z, val.w
            };

            T* row = mat.ptr(y);

            for (int x = threadIdx.x; x < mat.cols * channels; x += 32)
            {
                row[x] = vals[x % channels];
            }
        }
    }

    template<typename T>
    __global__ void set_to_with_mask(PtrStepSz<T> mat, const PtrStepb mask, typename TypeVec<T, 4>::vec_type val, int channels)
    {
        const int y = blockIdx.x * blockDim.y + threadIdx.y;

        if (y < mat.rows)
        {
            const T vals[] = {
                val.x, val.y, val.z, val.w
            };

            T* row = mat.ptr(y);
            const uchar* mask_row = mask.ptr(y);

            for (int x = threadIdx.x; x < mat.cols * channels; x += 32)
            {
                if (mask_row[x / channels])
                {
                    row[x] = vals[x % channels];
                }
            }
        }
    }

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, PtrStepSzb mask, int channels, cudaStream_t stream)
    {
        typedef typename TypeVec<T, 4>::vec_type vec_type;

        dim3 block(32, 8);
        dim3 grid(divUp(mat.rows, block.y));

        set_to_with_mask<T><<<grid, block, 0, stream>>>(PtrStepSz<T>(mat), mask, VecTraits<vec_type>::make(scalar), channels);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set_to_gpu<uchar >(PtrStepSzb mat, const uchar*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<schar >(PtrStepSzb mat, const schar*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<ushort>(PtrStepSzb mat, const ushort* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<short >(PtrStepSzb mat, const short*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<int   >(PtrStepSzb mat, const int*    scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<float >(PtrStepSzb mat, const float*  scalar, PtrStepSzb mask, int channels, cudaStream_t stream);
    template void set_to_gpu<double>(PtrStepSzb mat, const double* scalar, PtrStepSzb mask, int channels, cudaStream_t stream);

    template <typename T>
    void set_to_gpu(PtrStepSzb mat, const T* scalar, int channels, cudaStream_t stream)
    {
        typedef typename TypeVec<T, 4>::vec_type vec_type;

        dim3 block(32, 8);
        dim3 grid(divUp(mat.rows, block.y));

        set_to_without_mask<T><<<grid, block, 0, stream>>>(PtrStepSz<T>(mat), VecTraits<vec_type>::make(scalar), channels);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall ( cudaDeviceSynchronize() );
    }

    template void set_to_gpu<uchar >(PtrStepSzb mat, const uchar*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<schar >(PtrStepSzb mat, const schar*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<ushort>(PtrStepSzb mat, const ushort* scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<short >(PtrStepSzb mat, const short*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<int   >(PtrStepSzb mat, const int*    scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<float >(PtrStepSzb mat, const float*  scalar, int channels, cudaStream_t stream);
    template void set_to_gpu<double>(PtrStepSzb mat, const double* scalar, int channels, cudaStream_t stream);

    ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////// ConvertTo ////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    template <typename T, typename D, typename S> struct Convertor : unary_function<T, D>
    {
        Convertor(S alpha_, S beta_) : alpha(alpha_), beta(beta_) {}

        __device__ __forceinline__ D operator()(typename TypeTraits<T>::ParameterType src) const
        {
            return saturate_cast<D>(alpha * src + beta);
        }

        S alpha, beta;
    };

    namespace detail
    {
        template <size_t src_size, size_t dst_size, typename F> struct ConvertTraitsDispatcher : DefaultTransformFunctorTraits<F>
        {
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 1, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 8 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<1, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };

        template <typename F> struct ConvertTraitsDispatcher<2, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<2, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraitsDispatcher<4, 2, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 4 };
        };
        template <typename F> struct ConvertTraitsDispatcher<4, 4, F> : DefaultTransformFunctorTraits<F>
        {
            enum { smart_block_dim_y = 8 };
            enum { smart_shift = 2 };
        };

        template <typename F> struct ConvertTraits : ConvertTraitsDispatcher<sizeof(typename F::argument_type), sizeof(typename F::result_type), F>
        {
        };
    }

    template <typename T, typename D, typename S> struct TransformFunctorTraits< Convertor<T, D, S> > : detail::ConvertTraits< Convertor<T, D, S> >
    {
    };

    template<typename T, typename D, typename S>
    void cvt_(PtrStepSzb src, PtrStepSzb dst, double alpha, double beta, cudaStream_t stream)
    {
        cudaSafeCall( cudaSetDoubleForDevice(&alpha) );
        cudaSafeCall( cudaSetDoubleForDevice(&beta) );
        Convertor<T, D, S> op(static_cast<S>(alpha), static_cast<S>(beta));
        cv::gpu::device::transform((PtrStepSz<T>)src, (PtrStepSz<D>)dst, op, WithOutMask(), stream);
    }

#if defined  __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wmissing-declarations"
#endif

    void convert_gpu(PtrStepSzb src, int sdepth, PtrStepSzb dst, int ddepth, double alpha, double beta, cudaStream_t stream)
    {
        typedef void (*caller_t)(PtrStepSzb src, PtrStepSzb dst, double alpha, double beta, cudaStream_t stream);

        static const caller_t tab[7][7] =
        {
            {
                cvt_<uchar, uchar, float>,
                cvt_<uchar, schar, float>,
                cvt_<uchar, ushort, float>,
                cvt_<uchar, short, float>,
                cvt_<uchar, int, float>,
                cvt_<uchar, float, float>,
                cvt_<uchar, double, double>
            },
            {
                cvt_<schar, uchar, float>,
                cvt_<schar, schar, float>,
                cvt_<schar, ushort, float>,
                cvt_<schar, short, float>,
                cvt_<schar, int, float>,
                cvt_<schar, float, float>,
                cvt_<schar, double, double>
            },
            {
                cvt_<ushort, uchar, float>,
                cvt_<ushort, schar, float>,
                cvt_<ushort, ushort, float>,
                cvt_<ushort, short, float>,
                cvt_<ushort, int, float>,
                cvt_<ushort, float, float>,
                cvt_<ushort, double, double>
            },
            {
                cvt_<short, uchar, float>,
                cvt_<short, schar, float>,
                cvt_<short, ushort, float>,
                cvt_<short, short, float>,
                cvt_<short, int, float>,
                cvt_<short, float, float>,
                cvt_<short, double, double>
            },
            {
                cvt_<int, uchar, float>,
                cvt_<int, schar, float>,
                cvt_<int, ushort, float>,
                cvt_<int, short, float>,
                cvt_<int, int, double>,
                cvt_<int, float, double>,
                cvt_<int, double, double>
            },
            {
                cvt_<float, uchar, float>,
                cvt_<float, schar, float>,
                cvt_<float, ushort, float>,
                cvt_<float, short, float>,
                cvt_<float, int, float>,
                cvt_<float, float, float>,
                cvt_<float, double, double>
            },
            {
                cvt_<double, uchar, double>,
                cvt_<double, schar, double>,
                cvt_<double, ushort, double>,
                cvt_<double, short, double>,
                cvt_<double, int, double>,
                cvt_<double, float, double>,
                cvt_<double, double, double>
            }
        };

        caller_t func = tab[sdepth][ddepth];
        func(src, dst, alpha, beta, stream);
    }

#if defined __clang__
# pragma clang diagnostic pop
#endif
}}} // namespace cv { namespace gpu { namespace device
