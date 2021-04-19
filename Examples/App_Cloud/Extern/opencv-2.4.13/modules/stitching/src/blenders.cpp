/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

using namespace std;

namespace cv {
namespace detail {

static const float WEIGHT_EPS = 1e-5f;

Ptr<Blender> Blender::createDefault(int type, bool try_gpu)
{
    if (type == NO)
        return new Blender();
    if (type == FEATHER)
        return new FeatherBlender();
    if (type == MULTI_BAND)
        return new MultiBandBlender(try_gpu);
    CV_Error(CV_StsBadArg, "unsupported blending method");
    return NULL;
}


void Blender::prepare(const vector<Point> &corners, const vector<Size> &sizes)
{
    prepare(resultRoi(corners, sizes));
}


void Blender::prepare(Rect dst_roi)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;
}


void Blender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(dy + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}


void Blender::blend(Mat &dst, Mat &dst_mask)
{
    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
    dst_.release();
    dst_mask_.release();
}


void FeatherBlender::prepare(Rect dst_roi)
{
    Blender::prepare(dst_roi);
    dst_weight_map_.create(dst_roi.size(), CV_32F);
    dst_weight_map_.setTo(0);
}


void FeatherBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    createWeightMap(mask, sharpness_, weight_map_);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short>* src_row = img.ptr<Point3_<short> >(y);
        Point3_<short>* dst_row = dst_.ptr<Point3_<short> >(dy + y);
        const float* weight_row = weight_map_.ptr<float>(y);
        float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
            dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
            dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
}


void FeatherBlender::blend(Mat &dst, Mat &dst_mask)
{
    normalizeUsingWeightMap(dst_weight_map_, dst_);
    dst_mask_ = dst_weight_map_ > WEIGHT_EPS;
    Blender::blend(dst, dst_mask);
}


Rect FeatherBlender::createWeightMaps(const vector<Mat> &masks, const vector<Point> &corners,
                                      vector<Mat> &weight_maps)
{
    weight_maps.resize(masks.size());
    for (size_t i = 0; i < masks.size(); ++i)
        createWeightMap(masks[i], sharpness_, weight_maps[i]);

    Rect dst_roi = resultRoi(corners, masks);
    Mat weights_sum(dst_roi.size(), CV_32F);
    weights_sum.setTo(0);

    for (size_t i = 0; i < weight_maps.size(); ++i)
    {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y,
                 weight_maps[i].cols, weight_maps[i].rows);
        weights_sum(roi) += weight_maps[i];
    }

    for (size_t i = 0; i < weight_maps.size(); ++i)
    {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y,
                 weight_maps[i].cols, weight_maps[i].rows);
        Mat tmp = weights_sum(roi);
        tmp.setTo(1, tmp < numeric_limits<float>::epsilon());
        divide(weight_maps[i], tmp, weight_maps[i]);
    }

    return dst_roi;
}


MultiBandBlender::MultiBandBlender(int try_gpu, int num_bands, int weight_type)
{
    setNumBands(num_bands);
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    can_use_gpu_ = try_gpu && gpu::getCudaEnabledDeviceCount();
#else
    (void)try_gpu;
    can_use_gpu_ = false;
#endif
    CV_Assert(weight_type == CV_32F || weight_type == CV_16S);
    weight_type_ = weight_type;
}


void MultiBandBlender::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(max(dst_roi.width, dst_roi.height));
    num_bands_ = min(actual_num_bands_, static_cast<int>(ceil(log(max_len) / log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    Blender::prepare(dst_roi);

    dst_pyr_laplace_.resize(num_bands_ + 1);
    dst_pyr_laplace_[0] = dst_;

    dst_band_weights_.resize(num_bands_ + 1);
    dst_band_weights_[0].create(dst_roi.size(), weight_type_);
    dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i)
    {
        dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
                                   (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
        dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                    (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
        dst_pyr_laplace_[i].setTo(Scalar::all(0));
        dst_band_weights_[i].setTo(0);
    }
}


void MultiBandBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    Point tl_new(max(dst_roi_.x, tl.x - gap),
                 max(dst_roi_.y, tl.y - gap));
    Point br_new(min(dst_roi_.br().x, tl.x + img.cols + gap),
                 min(dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = max(br_new.y - dst_roi_.br().y, 0);
    int dx = max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    Mat img_with_border;
    copyMakeBorder(img, img_with_border, top, bottom, left, right,
                   BORDER_REFLECT);
    vector<Mat> src_pyr_laplace;
    if (can_use_gpu_ && img_with_border.depth() == CV_16S)
        createLaplacePyrGpu(img_with_border, num_bands_, src_pyr_laplace);
    else
        createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);

    // Create the weight map Gaussian pyramid
    Mat weight_map;
    vector<Mat> weight_pyr_gauss(num_bands_ + 1);

    if(weight_type_ == CV_32F)
    {
        mask.convertTo(weight_map, CV_32F, 1./255.);
    }
    else// weight_type_ == CV_16S
    {
        mask.convertTo(weight_map, CV_16S);
        add(weight_map, 1, weight_map, mask != 0);
    }

    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    if(weight_type_ == CV_32F)
    {
        for (int i = 0; i <= num_bands_; ++i)
        {
            for (int y = y_tl; y < y_br; ++y)
            {
                int y_ = y - y_tl;
                const Point3_<short>* src_row = src_pyr_laplace[i].ptr<Point3_<short> >(y_);
                Point3_<short>* dst_row = dst_pyr_laplace_[i].ptr<Point3_<short> >(y);
                const float* weight_row = weight_pyr_gauss[i].ptr<float>(y_);
                float* dst_weight_row = dst_band_weights_[i].ptr<float>(y);

                for (int x = x_tl; x < x_br; ++x)
                {
                    int x_ = x - x_tl;
                    dst_row[x].x += static_cast<short>(src_row[x_].x * weight_row[x_]);
                    dst_row[x].y += static_cast<short>(src_row[x_].y * weight_row[x_]);
                    dst_row[x].z += static_cast<short>(src_row[x_].z * weight_row[x_]);
                    dst_weight_row[x] += weight_row[x_];
                }
            }
            x_tl /= 2; y_tl /= 2;
            x_br /= 2; y_br /= 2;
        }
    }
    else// weight_type_ == CV_16S
    {
        for (int i = 0; i <= num_bands_; ++i)
        {
            for (int y = y_tl; y < y_br; ++y)
            {
                int y_ = y - y_tl;
                const Point3_<short>* src_row = src_pyr_laplace[i].ptr<Point3_<short> >(y_);
                Point3_<short>* dst_row = dst_pyr_laplace_[i].ptr<Point3_<short> >(y);
                const short* weight_row = weight_pyr_gauss[i].ptr<short>(y_);
                short* dst_weight_row = dst_band_weights_[i].ptr<short>(y);

                for (int x = x_tl; x < x_br; ++x)
                {
                    int x_ = x - x_tl;
                    dst_row[x].x += short((src_row[x_].x * weight_row[x_]) >> 8);
                    dst_row[x].y += short((src_row[x_].y * weight_row[x_]) >> 8);
                    dst_row[x].z += short((src_row[x_].z * weight_row[x_]) >> 8);
                    dst_weight_row[x] += weight_row[x_];
                }
            }
            x_tl /= 2; y_tl /= 2;
            x_br /= 2; y_br /= 2;
        }
    }
}


void MultiBandBlender::blend(Mat &dst, Mat &dst_mask)
{
    for (int i = 0; i <= num_bands_; ++i)
        normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);

    if (can_use_gpu_)
        restoreImageFromLaplacePyrGpu(dst_pyr_laplace_);
    else
        restoreImageFromLaplacePyr(dst_pyr_laplace_);

    dst_ = dst_pyr_laplace_[0];
    dst_ = dst_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    dst_mask_ = dst_band_weights_[0] > WEIGHT_EPS;
    dst_mask_ = dst_mask_(Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    dst_pyr_laplace_.clear();
    dst_band_weights_.clear();

    Blender::blend(dst, dst_mask);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void normalizeUsingWeightMap(const Mat& weight, Mat& src)
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if(tegra::normalizeUsingWeightMap(weight, src))
        return;
#endif
    CV_Assert(src.type() == CV_16SC3);

    if(weight.type() == CV_32FC1)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            Point3_<short> *row = src.ptr<Point3_<short> >(y);
            const float *weight_row = weight.ptr<float>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
                row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
                row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
            }
        }
    }
    else
    {
        CV_Assert(weight.type() == CV_16SC1);

        for (int y = 0; y < src.rows; ++y)
        {
            const short *weight_row = weight.ptr<short>(y);
            Point3_<short> *row = src.ptr<Point3_<short> >(y);

            for (int x = 0; x < src.cols; ++x)
            {
                int w = weight_row[x] + 1;
                row[x].x = static_cast<short>((row[x].x << 8) / w);
                row[x].y = static_cast<short>((row[x].y << 8) / w);
                row[x].z = static_cast<short>((row[x].z << 8) / w);
            }
        }
    }
}


void createWeightMap(const Mat &mask, float sharpness, Mat &weight)
{
    CV_Assert(mask.type() == CV_8U);
    distanceTransform(mask, weight, CV_DIST_L1, 3);
    threshold(weight * sharpness, weight, 1.f, 1.f, THRESH_TRUNC);
}


void createLaplacePyr(const Mat &img, int num_levels, vector<Mat> &pyr)
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if(tegra::createLaplacePyr(img, num_levels, pyr))
        return;
#endif

    pyr.resize(num_levels + 1);

    if(img.depth() == CV_8U)
    {
        if(num_levels == 0)
        {
            img.convertTo(pyr[0], CV_16S);
            return;
        }

        Mat downNext;
        Mat current = img;
        pyrDown(img, downNext);

        for(int i = 1; i < num_levels; ++i)
        {
            Mat lvl_up;
            Mat lvl_down;

            pyrDown(downNext, lvl_down);
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[i-1], noArray(), CV_16S);

            current = downNext;
            downNext = lvl_down;
        }

        {
            Mat lvl_up;
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[num_levels-1], noArray(), CV_16S);

            downNext.convertTo(pyr[num_levels], CV_16S);
        }
    }
    else
    {
        pyr[0] = img;
        for (int i = 0; i < num_levels; ++i)
            pyrDown(pyr[i], pyr[i + 1]);
        Mat tmp;
        for (int i = 0; i < num_levels; ++i)
        {
            pyrUp(pyr[i + 1], tmp, pyr[i].size());
            subtract(pyr[i], tmp, pyr[i]);
        }
    }
}


void createLaplacePyrGpu(const Mat &img, int num_levels, vector<Mat> &pyr)
{
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    pyr.resize(num_levels + 1);

    vector<gpu::GpuMat> gpu_pyr(num_levels + 1);
    gpu_pyr[0].upload(img);
    for (int i = 0; i < num_levels; ++i)
        gpu::pyrDown(gpu_pyr[i], gpu_pyr[i + 1]);

    gpu::GpuMat tmp;
    for (int i = 0; i < num_levels; ++i)
    {
        gpu::pyrUp(gpu_pyr[i + 1], tmp);
        gpu::subtract(gpu_pyr[i], tmp, gpu_pyr[i]);
        gpu_pyr[i].download(pyr[i]);
    }

    gpu_pyr[num_levels].download(pyr[num_levels]);
#else
    (void)img;
    (void)num_levels;
    (void)pyr;
    CV_Error(CV_StsNotImplemented, "CUDA optimization is unavailable");
#endif
}


void restoreImageFromLaplacePyr(vector<Mat> &pyr)
{
    if (pyr.empty())
        return;
    Mat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
        add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}


void restoreImageFromLaplacePyrGpu(vector<Mat> &pyr)
{
#if defined(HAVE_OPENCV_GPU) && !defined(DYNAMIC_CUDA_SUPPORT)
    if (pyr.empty())
        return;

    vector<gpu::GpuMat> gpu_pyr(pyr.size());
    for (size_t i = 0; i < pyr.size(); ++i)
        gpu_pyr[i].upload(pyr[i]);

    gpu::GpuMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        gpu::pyrUp(gpu_pyr[i], tmp);
        gpu::add(tmp, gpu_pyr[i - 1], gpu_pyr[i - 1]);
    }

    gpu_pyr[0].download(pyr[0]);
#else
    (void)pyr;
    CV_Error(CV_StsNotImplemented, "CUDA optimization is unavailable");
#endif
}

} // namespace detail
} // namespace cv
