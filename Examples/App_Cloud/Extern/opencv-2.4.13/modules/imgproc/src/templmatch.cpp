/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

namespace cv
{

void crossCorr( const Mat& img, const Mat& _templ, Mat& corr,
                Size corrsize, int ctype,
                Point anchor, double delta, int borderType )
{
    const double blockScale = 4.5;
    const int minBlockSize = 256;
    std::vector<uchar> buf;

    Mat templ = _templ;
    int depth = img.depth(), cn = img.channels();
    int tdepth = templ.depth(), tcn = templ.channels();
    int cdepth = CV_MAT_DEPTH(ctype), ccn = CV_MAT_CN(ctype);

    CV_Assert( img.dims <= 2 && templ.dims <= 2 && corr.dims <= 2 );

    if( depth != tdepth && tdepth != std::max(CV_32F, depth) )
    {
        _templ.convertTo(templ, std::max(CV_32F, depth));
        tdepth = templ.depth();
    }

    CV_Assert( depth == tdepth || tdepth == CV_32F);
    CV_Assert( corrsize.height <= img.rows + templ.rows - 1 &&
               corrsize.width <= img.cols + templ.cols - 1 );

    CV_Assert( ccn == 1 || delta == 0 );

    corr.create(corrsize, ctype);

    int maxDepth = depth > CV_8S ? CV_64F : std::max(std::max(CV_32F, tdepth), cdepth);
    Size blocksize, dftsize;

    blocksize.width = cvRound(templ.cols*blockScale);
    blocksize.width = std::max( blocksize.width, minBlockSize - templ.cols + 1 );
    blocksize.width = std::min( blocksize.width, corr.cols );
    blocksize.height = cvRound(templ.rows*blockScale);
    blocksize.height = std::max( blocksize.height, minBlockSize - templ.rows + 1 );
    blocksize.height = std::min( blocksize.height, corr.rows );

    dftsize.width = std::max(getOptimalDFTSize(blocksize.width + templ.cols - 1), 2);
    dftsize.height = getOptimalDFTSize(blocksize.height + templ.rows - 1);
    if( dftsize.width <= 0 || dftsize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "the input arrays are too big" );

    // recompute block size
    blocksize.width = dftsize.width - templ.cols + 1;
    blocksize.width = MIN( blocksize.width, corr.cols );
    blocksize.height = dftsize.height - templ.rows + 1;
    blocksize.height = MIN( blocksize.height, corr.rows );

    Mat dftTempl( dftsize.height*tcn, dftsize.width, maxDepth );
    Mat dftImg( dftsize, maxDepth );

    int i, k, bufSize = 0;
    if( tcn > 1 && tdepth != maxDepth )
        bufSize = templ.cols*templ.rows*CV_ELEM_SIZE(tdepth);

    if( cn > 1 && depth != maxDepth )
        bufSize = std::max( bufSize, (blocksize.width + templ.cols - 1)*
            (blocksize.height + templ.rows - 1)*CV_ELEM_SIZE(depth));

    if( (ccn > 1 || cn > 1) && cdepth != maxDepth )
        bufSize = std::max( bufSize, blocksize.width*blocksize.height*CV_ELEM_SIZE(cdepth));

    buf.resize(bufSize);

    // compute DFT of each template plane
    for( k = 0; k < tcn; k++ )
    {
        int yofs = k*dftsize.height;
        Mat src = templ;
        Mat dst(dftTempl, Rect(0, yofs, dftsize.width, dftsize.height));
        Mat dst1(dftTempl, Rect(0, yofs, templ.cols, templ.rows));

        if( tcn > 1 )
        {
            src = tdepth == maxDepth ? dst1 : Mat(templ.size(), tdepth, &buf[0]);
            int pairs[] = {k, 0};
            mixChannels(&templ, 1, &src, 1, pairs, 1);
        }

        if( dst1.data != src.data )
            src.convertTo(dst1, dst1.depth());

        if( dst.cols > templ.cols )
        {
            Mat part(dst, Range(0, templ.rows), Range(templ.cols, dst.cols));
            part = Scalar::all(0);
        }
        dft(dst, dst, 0, templ.rows);
    }

    int tileCountX = (corr.cols + blocksize.width - 1)/blocksize.width;
    int tileCountY = (corr.rows + blocksize.height - 1)/blocksize.height;
    int tileCount = tileCountX * tileCountY;

    Size wholeSize = img.size();
    Point roiofs(0,0);
    Mat img0 = img;

    if( !(borderType & BORDER_ISOLATED) )
    {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(roiofs.y, wholeSize.height-img.rows-roiofs.y,
                       roiofs.x, wholeSize.width-img.cols-roiofs.x);
    }
    borderType |= BORDER_ISOLATED;

    // calculate correlation by blocks
    for( i = 0; i < tileCount; i++ )
    {
        int x = (i%tileCountX)*blocksize.width;
        int y = (i/tileCountX)*blocksize.height;

        Size bsz(std::min(blocksize.width, corr.cols - x),
                 std::min(blocksize.height, corr.rows - y));
        Size dsz(bsz.width + templ.cols - 1, bsz.height + templ.rows - 1);
        int x0 = x - anchor.x + roiofs.x, y0 = y - anchor.y + roiofs.y;
        int x1 = std::max(0, x0), y1 = std::max(0, y0);
        int x2 = std::min(img0.cols, x0 + dsz.width);
        int y2 = std::min(img0.rows, y0 + dsz.height);
        Mat src0(img0, Range(y1, y2), Range(x1, x2));
        Mat dst(dftImg, Rect(0, 0, dsz.width, dsz.height));
        Mat dst1(dftImg, Rect(x1-x0, y1-y0, x2-x1, y2-y1));
        Mat cdst(corr, Rect(x, y, bsz.width, bsz.height));

        for( k = 0; k < cn; k++ )
        {
            Mat src = src0;
            dftImg = Scalar::all(0);

            if( cn > 1 )
            {
                src = depth == maxDepth ? dst1 : Mat(y2-y1, x2-x1, depth, &buf[0]);
                int pairs[] = {k, 0};
                mixChannels(&src0, 1, &src, 1, pairs, 1);
            }

            if( dst1.data != src.data )
                src.convertTo(dst1, dst1.depth());

            if( x2 - x1 < dsz.width || y2 - y1 < dsz.height )
                copyMakeBorder(dst1, dst, y1-y0, dst.rows-dst1.rows-(y1-y0),
                               x1-x0, dst.cols-dst1.cols-(x1-x0), borderType);

            dft( dftImg, dftImg, 0, dsz.height );
            Mat dftTempl1(dftTempl, Rect(0, tcn > 1 ? k*dftsize.height : 0,
                                         dftsize.width, dftsize.height));
            mulSpectrums(dftImg, dftTempl1, dftImg, 0, true);
            dft( dftImg, dftImg, DFT_INVERSE + DFT_SCALE, bsz.height );

            src = dftImg(Rect(0, 0, bsz.width, bsz.height));

            if( ccn > 1 )
            {
                if( cdepth != maxDepth )
                {
                    Mat plane(bsz, cdepth, &buf[0]);
                    src.convertTo(plane, cdepth, 1, delta);
                    src = plane;
                }
                int pairs[] = {0, k};
                mixChannels(&src, 1, &cdst, 1, pairs, 1);
            }
            else
            {
                if( k == 0 )
                    src.convertTo(cdst, cdepth, 1, delta);
                else
                {
                    if( maxDepth != cdepth )
                    {
                        Mat plane(bsz, cdepth, &buf[0]);
                        src.convertTo(plane, cdepth);
                        src = plane;
                    }
                    add(src, cdst, cdst);
                }
            }
        }
    }
}

}

/*****************************************************************************************/

void cv::matchTemplate( InputArray _img, InputArray _templ, OutputArray _result, int method )
{
    CV_Assert( CV_TM_SQDIFF <= method && method <= CV_TM_CCOEFF_NORMED );

    int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
                  method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    bool isNormed = method == CV_TM_CCORR_NORMED ||
                    method == CV_TM_SQDIFF_NORMED ||
                    method == CV_TM_CCOEFF_NORMED;

    Mat img = _img.getMat(), templ = _templ.getMat();
    if( img.rows < templ.rows || img.cols < templ.cols )
        std::swap(img, templ);

    CV_Assert( (img.depth() == CV_8U || img.depth() == CV_32F) &&
               img.type() == templ.type() );

    CV_Assert( img.rows >= templ.rows && img.cols >= templ.cols);

    Size corrSize(img.cols - templ.cols + 1, img.rows - templ.rows + 1);
    _result.create(corrSize, CV_32F);
    Mat result = _result.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::matchTemplate(img, templ, result, method))
        return;
#endif

    int cn = img.channels();
    crossCorr( img, templ, result, result.size(), result.type(), Point(0,0), 0, 0);

    if( method == CV_TM_CCORR )
        return;

    double invArea = 1./((double)templ.rows * templ.cols);

    Mat sum, sqsum;
    Scalar templMean, templSdv;
    double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
    double templNorm = 0, templSum2 = 0;

    if( method == CV_TM_CCOEFF )
    {
        integral(img, sum, CV_64F);
        templMean = mean(templ);
    }
    else
    {
        integral(img, sum, sqsum, CV_64F);
        meanStdDev( templ, templMean, templSdv );

        templNorm = CV_SQR(templSdv[0]) + CV_SQR(templSdv[1]) +
                    CV_SQR(templSdv[2]) + CV_SQR(templSdv[3]);

        if( templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED )
        {
            result = Scalar::all(1);
            return;
        }

        templSum2 = templNorm +
                     CV_SQR(templMean[0]) + CV_SQR(templMean[1]) +
                     CV_SQR(templMean[2]) + CV_SQR(templMean[3]);

        if( numType != 1 )
        {
            templMean = Scalar::all(0);
            templNorm = templSum2;
        }

        templSum2 /= invArea;
        templNorm = sqrt(templNorm);
        templNorm /= sqrt(invArea); // care of accuracy here

        q0 = (double*)sqsum.data;
        q1 = q0 + templ.cols*cn;
        q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
        q3 = q2 + templ.cols*cn;
    }

    double* p0 = (double*)sum.data;
    double* p1 = p0 + templ.cols*cn;
    double* p2 = (double*)(sum.data + templ.rows*sum.step);
    double* p3 = p2 + templ.cols*cn;

    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

    int i, j, k;

    for( i = 0; i < result.rows; i++ )
    {
        float* rrow = (float*)(result.data + i*result.step);
        int idx = i * sumstep;
        int idx2 = i * sqstep;

        for( j = 0; j < result.cols; j++, idx += cn, idx2 += cn )
        {
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;

            if( numType == 1 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = p0[idx+k] - p1[idx+k] - p2[idx+k] + p3[idx+k];
                    wndMean2 += CV_SQR(t);
                    num -= t*templMean[k];
                }

                wndMean2 *= invArea;
            }

            if( isNormed || numType == 2 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = q0[idx2+k] - q1[idx2+k] - q2[idx2+k] + q3[idx2+k];
                    wndSum2 += t;
                }

                if( numType == 2 )
                {
                    num = wndSum2 - 2*num + templSum2;
                    num = MAX(num, 0.);
                }
            }

            if( isNormed )
            {
                t = sqrt(MAX(wndSum2 - wndMean2,0))*templNorm;
                if( fabs(num) < t )
                    num /= t;
                else if( fabs(num) < t*1.125 )
                    num = num > 0 ? 1 : -1;
                else
                    num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }

            rrow[j] = (float)num;
        }
    }
}


CV_IMPL void
cvMatchTemplate( const CvArr* _img, const CvArr* _templ, CvArr* _result, int method )
{
    cv::Mat img = cv::cvarrToMat(_img), templ = cv::cvarrToMat(_templ),
        result = cv::cvarrToMat(_result);
    CV_Assert( result.size() == cv::Size(std::abs(img.cols - templ.cols) + 1,
                                         std::abs(img.rows - templ.rows) + 1) &&
              result.type() == CV_32F );
    matchTemplate(img, templ, result, method);
}

/* End of file. */
