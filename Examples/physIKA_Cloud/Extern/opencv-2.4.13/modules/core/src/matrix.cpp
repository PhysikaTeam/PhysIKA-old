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
#include "opencv2/core/gpumat.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/core/opengl_interop_deprecated.hpp"

/****************************************************************************************\
*                           [scaled] Identity matrix initialization                      *
\****************************************************************************************/

namespace cv {

void swap( Mat& a, Mat& b )
{
    std::swap(a.flags, b.flags);
    std::swap(a.dims, b.dims);
    std::swap(a.rows, b.rows);
    std::swap(a.cols, b.cols);
    std::swap(a.data, b.data);
    std::swap(a.refcount, b.refcount);
    std::swap(a.datastart, b.datastart);
    std::swap(a.dataend, b.dataend);
    std::swap(a.datalimit, b.datalimit);
    std::swap(a.allocator, b.allocator);

    std::swap(a.size.p, b.size.p);
    std::swap(a.step.p, b.step.p);
    std::swap(a.step.buf[0], b.step.buf[0]);
    std::swap(a.step.buf[1], b.step.buf[1]);

    if( a.step.p == b.step.buf )
    {
        a.step.p = a.step.buf;
        a.size.p = &a.rows;
    }

    if( b.step.p == a.step.buf )
    {
        b.step.p = b.step.buf;
        b.size.p = &b.rows;
    }
}


static inline void setSize( Mat& m, int _dims, const int* _sz,
                            const size_t* _steps, bool autoSteps=false )
{
    CV_Assert( 0 <= _dims && _dims <= CV_MAX_DIM );
    if( m.dims != _dims )
    {
        if( m.step.p != m.step.buf )
        {
            fastFree(m.step.p);
            m.step.p = m.step.buf;
            m.size.p = &m.rows;
        }
        if( _dims > 2 )
        {
            m.step.p = (size_t*)fastMalloc(_dims*sizeof(m.step.p[0]) + (_dims+1)*sizeof(m.size.p[0]));
            m.size.p = (int*)(m.step.p + _dims) + 1;
            m.size.p[-1] = _dims;
            m.rows = m.cols = -1;
        }
    }

    m.dims = _dims;
    if( !_sz )
        return;

    size_t esz = CV_ELEM_SIZE(m.flags), total = esz;
    int i;
    for( i = _dims-1; i >= 0; i-- )
    {
        int s = _sz[i];
        CV_Assert( s >= 0 );
        m.size.p[i] = s;

        if( _steps )
            m.step.p[i] = i < _dims-1 ? _steps[i] : esz;
        else if( autoSteps )
        {
            m.step.p[i] = total;
            int64 total1 = (int64)total*s;
            if( (uint64)total1 != (size_t)total1 )
                CV_Error( CV_StsOutOfRange, "The total matrix size does not fit to \"size_t\" type" );
            total = (size_t)total1;
        }
    }

    if( _dims == 1 )
    {
        m.dims = 2;
        m.cols = 1;
        m.step[1] = esz;
    }
}

static void updateContinuityFlag(Mat& m)
{
    int i, j;
    for( i = 0; i < m.dims; i++ )
    {
        if( m.size[i] > 1 )
            break;
    }

    for( j = m.dims-1; j > i; j-- )
    {
        if( m.step[j]*m.size[j] < m.step[j-1] )
            break;
    }

    uint64 t = (uint64)m.step[0]*m.size[0];
    if( j <= i && t == (size_t)t )
        m.flags |= Mat::CONTINUOUS_FLAG;
    else
        m.flags &= ~Mat::CONTINUOUS_FLAG;
}

static void finalizeHdr(Mat& m)
{
    updateContinuityFlag(m);
    int d = m.dims;
    if( d > 2 )
        m.rows = m.cols = -1;
    if( m.data )
    {
        m.datalimit = m.datastart + m.size[0]*m.step[0];
        if( m.size[0] > 0 )
        {
            m.dataend = m.data + m.size[d-1]*m.step[d-1];
            for( int i = 0; i < d-1; i++ )
                m.dataend += (m.size[i] - 1)*m.step[i];
        }
        else
            m.dataend = m.datalimit;
    }
    else
        m.dataend = m.datalimit = 0;
}


void Mat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert(0 <= d && d <= CV_MAX_DIM && _sizes);
    _type = CV_MAT_TYPE(_type);

    if( data && (d == dims || (d == 1 && dims <= 2)) && _type == type() )
    {
        if( d == 2 && rows == _sizes[0] && cols == _sizes[1] )
            return;
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d && (d > 1 || size[1] == 1))
            return;
    }

    release();
    if( d == 0 )
        return;
    flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
    setSize(*this, d, _sizes, 0, true);

    if( total() > 0 )
    {
#ifdef HAVE_TGPU
        if( !allocator || allocator == tegra::getAllocator() ) allocator = tegra::getAllocator(d, _sizes, _type);
#endif
        if( !allocator )
        {
            size_t totalsize = alignSize(step.p[0]*size.p[0], (int)sizeof(*refcount));
            data = datastart = (uchar*)fastMalloc(totalsize + (int)sizeof(*refcount));
            refcount = (int*)(data + totalsize);
            *refcount = 1;
        }
        else
        {
#ifdef HAVE_TGPU
           try
            {
                allocator->allocate(dims, size, _type, refcount, datastart, data, step.p);
                CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
            }catch(...)
            {
                allocator = 0;
                size_t totalSize = alignSize(step.p[0]*size.p[0], (int)sizeof(*refcount));
                data = datastart = (uchar*)fastMalloc(totalSize + (int)sizeof(*refcount));
                refcount = (int*)(data + totalSize);
                *refcount = 1;
            }
#else
            allocator->allocate(dims, size, _type, refcount, datastart, data, step.p);
            CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
#endif
        }
    }

    finalizeHdr(*this);
}

void Mat::copySize(const Mat& m)
{
    setSize(*this, m.dims, 0, 0);
    for( int i = 0; i < dims; i++ )
    {
        size[i] = m.size[i];
        step[i] = m.step[i];
    }
}

void Mat::deallocate()
{
    if( allocator )
        allocator->deallocate(refcount, datastart, data);
    else
    {
        CV_DbgAssert(refcount != 0);
        fastFree(datastart);
    }
}


Mat::Mat(const Mat& m, const Range& _rowRange, const Range& _colRange) : size(&rows)
{
    initEmpty();
    CV_Assert( m.dims >= 2 );
    if( m.dims > 2 )
    {
        AutoBuffer<Range> rs(m.dims);
        rs[0] = _rowRange;
        rs[1] = _colRange;
        for( int i = 2; i < m.dims; i++ )
            rs[i] = Range::all();
        *this = m(rs);
        return;
    }

    *this = m;
    if( _rowRange != Range::all() && _rowRange != Range(0,rows) )
    {
        CV_Assert( 0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows );
        rows = _rowRange.size();
        data += step*_rowRange.start;
        flags |= SUBMATRIX_FLAG;
    }

    if( _colRange != Range::all() && _colRange != Range(0,cols) )
    {
        CV_Assert( 0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols );
        cols = _colRange.size();
        data += _colRange.start*elemSize();
        flags &= cols < m.cols ? ~CONTINUOUS_FLAG : -1;
        flags |= SUBMATRIX_FLAG;
    }

    if( rows == 1 )
        flags |= CONTINUOUS_FLAG;

    if( rows <= 0 || cols <= 0 )
    {
        release();
        rows = cols = 0;
    }
}


Mat::Mat(const Mat& m, const Rect& roi)
    : flags(m.flags), dims(2), rows(roi.height), cols(roi.width),
    data(m.data + roi.y*m.step[0]), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit),
    allocator(m.allocator), size(&rows)
{
    CV_Assert( m.dims <= 2 );
    flags &= roi.width < m.cols ? ~CONTINUOUS_FLAG : -1;
    flags |= roi.height == 1 ? CONTINUOUS_FLAG : 0;

    size_t esz = CV_ELEM_SIZE(flags);
    data += roi.x*esz;
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
              0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( refcount )
        CV_XADD(refcount, 1);
    if( roi.width < m.cols || roi.height < m.rows )
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0]; step[1] = esz;

    if( rows <= 0 || cols <= 0 )
    {
        release();
        rows = cols = 0;
    }
}


Mat::Mat(int _dims, const int* _sizes, int _type, void* _data, const size_t* _steps) : size(&rows)
{
    initEmpty();
    flags |= CV_MAT_TYPE(_type);
    data = datastart = (uchar*)_data;
    setSize(*this, _dims, _sizes, _steps, true);
    finalizeHdr(*this);
}


Mat::Mat(const Mat& m, const Range* ranges) : size(&rows)
{
    initEmpty();
    int i, d = m.dims;

    CV_Assert(ranges);
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        CV_Assert( r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]) );
    }
    *this = m;
    for( i = 0; i < d; i++ )
    {
        Range r = ranges[i];
        if( r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            data += r.start*step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}


Mat::Mat(const CvMatND* m, bool copyData) : size(&rows)
{
    initEmpty();
    if( !m )
        return;
    data = datastart = m->data.ptr;
    flags |= CV_MAT_TYPE(m->type);
    int _sizes[CV_MAX_DIM];
    size_t _steps[CV_MAX_DIM];

    int i, d = m->dims;
    for( i = 0; i < d; i++ )
    {
        _sizes[i] = m->dim[i].size;
        _steps[i] = m->dim[i].step;
    }

    setSize(*this, d, _sizes, _steps);
    finalizeHdr(*this);

    if( copyData )
    {
        Mat temp(*this);
        temp.copyTo(*this);
    }
}


Mat Mat::diag(int d) const
{
    CV_Assert( dims <= 2 );
    Mat m = *this;
    size_t esz = elemSize();
    int len;

    if( d >= 0 )
    {
        len = std::min(cols - d, rows);
        m.data += esz*d;
    }
    else
    {
        len = std::min(rows + d, cols);
        m.data -= step[0]*d;
    }
    CV_DbgAssert( len > 0 );

    m.size[0] = m.rows = len;
    m.size[1] = m.cols = 1;
    m.step[0] += (len > 1 ? esz : 0);

    if( m.rows > 1 )
        m.flags &= ~CONTINUOUS_FLAG;
    else
        m.flags |= CONTINUOUS_FLAG;

    if( size() != Size(1,1) )
        m.flags |= SUBMATRIX_FLAG;

    return m;
}


Mat::Mat(const CvMat* m, bool copyData) : size(&rows)
{
    initEmpty();

    if( !m )
        return;

    if( !copyData )
    {
        flags = MAGIC_VAL + (m->type & (CV_MAT_TYPE_MASK|CV_MAT_CONT_FLAG));
        dims = 2;
        rows = m->rows;
        cols = m->cols;
        data = datastart = m->data.ptr;
        size_t esz = CV_ELEM_SIZE(m->type), minstep = cols*esz, _step = m->step;
        if( _step == 0 )
            _step = minstep;
        datalimit = datastart + _step*rows;
        dataend = datalimit - _step + minstep;
        step[0] = _step; step[1] = esz;
    }
    else
    {
        data = datastart = dataend = 0;
        Mat(m->rows, m->cols, m->type, m->data.ptr, m->step).copyTo(*this);
    }
}


Mat::Mat(const IplImage* img, bool copyData) : size(&rows)
{
    initEmpty();

    if( !img )
        return;

    dims = 2;
    CV_DbgAssert(CV_IS_IMAGE(img) && img->imageData != 0);

    int imgdepth = IPL2CV_DEPTH(img->depth);
    size_t esz;
    step[0] = img->widthStep;

    if(!img->roi)
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL);
        flags = MAGIC_VAL + CV_MAKETYPE(imgdepth, img->nChannels);
        rows = img->height; cols = img->width;
        datastart = data = (uchar*)img->imageData;
        esz = CV_ELEM_SIZE(flags);
    }
    else
    {
        CV_Assert(img->dataOrder == IPL_DATA_ORDER_PIXEL || img->roi->coi != 0);
        bool selectedPlane = img->roi->coi && img->dataOrder == IPL_DATA_ORDER_PLANE;
        flags = MAGIC_VAL + CV_MAKETYPE(imgdepth, selectedPlane ? 1 : img->nChannels);
        rows = img->roi->height; cols = img->roi->width;
        esz = CV_ELEM_SIZE(flags);
        data = datastart = (uchar*)img->imageData +
            (selectedPlane ? (img->roi->coi - 1)*step*img->height : 0) +
            img->roi->yOffset*step[0] + img->roi->xOffset*esz;
    }
    datalimit = datastart + step.p[0]*rows;
    dataend = datastart + step.p[0]*(rows-1) + esz*cols;
    flags |= (cols*esz == step.p[0] || rows == 1 ? CONTINUOUS_FLAG : 0);
    step[1] = esz;

    if( copyData )
    {
        Mat m = *this;
        release();
        if( !img->roi || !img->roi->coi ||
            img->dataOrder == IPL_DATA_ORDER_PLANE)
            m.copyTo(*this);
        else
        {
            int ch[] = {img->roi->coi - 1, 0};
            create(m.rows, m.cols, m.type());
            mixChannels(&m, 1, this, 1, ch, 1);
        }
    }
}


Mat::operator IplImage() const
{
    CV_Assert( dims <= 2 );
    IplImage img;
    cvInitImageHeader(&img, size(), cvIplDepth(flags), channels());
    cvSetData(&img, data, (int)step[0]);
    return img;
}


void Mat::pop_back(size_t nelems)
{
    CV_Assert( nelems <= (size_t)size.p[0] );

    if( isSubmatrix() )
        *this = rowRange(0, size.p[0] - (int)nelems);
    else
    {
        size.p[0] -= (int)nelems;
        dataend -= nelems*step.p[0];
        /*if( size.p[0] <= 1 )
        {
            if( dims <= 2 )
                flags |= CONTINUOUS_FLAG;
            else
                updateContinuityFlag(*this);
        }*/
    }
}


void Mat::push_back_(const void* elem)
{
    int r = size.p[0];
    if( isSubmatrix() || dataend + step.p[0] > datalimit )
        reserve( std::max(r + 1, (r*3+1)/2) );

    size_t esz = elemSize();
    memcpy(data + r*step.p[0], elem, esz);
    size.p[0] = r + 1;
    dataend += step.p[0];
    if( esz < step.p[0] )
        flags &= ~CONTINUOUS_FLAG;
}

void Mat::reserve(size_t nelems)
{
    const size_t MIN_SIZE = 64;

    CV_Assert( (int)nelems >= 0 );
    if( !isSubmatrix() && data + step.p[0]*nelems <= datalimit )
        return;

    int r = size.p[0];

    if( (size_t)r >= nelems )
        return;

    size.p[0] = std::max((int)nelems, 1);
    size_t newsize = total()*elemSize();

    if( newsize < MIN_SIZE )
        size.p[0] = (int)((MIN_SIZE + newsize - 1)*nelems/newsize);

    Mat m(dims, size.p, type());
    size.p[0] = r;
    if( r > 0 )
    {
        Mat mpart = m.rowRange(0, r);
        copyTo(mpart);
    }

    *this = m;
    size.p[0] = r;
    dataend = data + step.p[0]*r;
}


void Mat::resize(size_t nelems)
{
    int saveRows = size.p[0];
    if( saveRows == (int)nelems )
        return;
    CV_Assert( (int)nelems >= 0 );

    if( isSubmatrix() || data + step.p[0]*nelems > datalimit )
        reserve(nelems);

    size.p[0] = (int)nelems;
    dataend += (size.p[0] - saveRows)*step.p[0];

    //updateContinuityFlag(*this);
}


void Mat::resize(size_t nelems, const Scalar& s)
{
    int saveRows = size.p[0];
    resize(nelems);

    if( size.p[0] > saveRows )
    {
        Mat part = rowRange(saveRows, size.p[0]);
        part = s;
    }
}

void Mat::push_back(const Mat& elems)
{
    int r = size.p[0], delta = elems.size.p[0];
    if( delta == 0 )
        return;
    if( this == &elems )
    {
        Mat tmp = elems;
        push_back(tmp);
        return;
    }
    if( !data )
    {
        *this = elems.clone();
        return;
    }

    size.p[0] = elems.size.p[0];
    bool eq = size == elems.size;
    size.p[0] = r;
    if( !eq )
        CV_Error(CV_StsUnmatchedSizes, "");
    if( type() != elems.type() )
        CV_Error(CV_StsUnmatchedFormats, "");

    if( isSubmatrix() || dataend + step.p[0]*delta > datalimit )
        reserve( std::max(r + delta, (r*3+1)/2) );

    size.p[0] += delta;
    dataend += step.p[0]*delta;

    //updateContinuityFlag(*this);

    if( isContinuous() && elems.isContinuous() )
        memcpy(data + r*step.p[0], elems.data, elems.total()*elems.elemSize());
    else
    {
        Mat part = rowRange(r, r + delta);
        elems.copyTo(part);
    }
}


Mat cvarrToMat(const CvArr* arr, bool copyData,
               bool /*allowND*/, int coiMode)
{
    if( !arr )
        return Mat();
    if( CV_IS_MAT(arr) )
        return Mat((const CvMat*)arr, copyData );
    if( CV_IS_MATND(arr) )
        return Mat((const CvMatND*)arr, copyData );
    if( CV_IS_IMAGE(arr) )
    {
        const IplImage* iplimg = (const IplImage*)arr;
        if( coiMode == 0 && iplimg->roi && iplimg->roi->coi > 0 )
            CV_Error(CV_BadCOI, "COI is not supported by the function");
        return Mat(iplimg, copyData);
    }
    if( CV_IS_SEQ(arr) )
    {
        CvSeq* seq = (CvSeq*)arr;
        CV_Assert(seq->total > 0 && CV_ELEM_SIZE(seq->flags) == seq->elem_size);
        if(!copyData && seq->first->next == seq->first)
            return Mat(seq->total, 1, CV_MAT_TYPE(seq->flags), seq->first->data);
        Mat buf(seq->total, 1, CV_MAT_TYPE(seq->flags));
        cvCvtSeqToArray(seq, buf.data, CV_WHOLE_SEQ);
        return buf;
    }
    CV_Error(CV_StsBadArg, "Unknown array type");
    return Mat();
}

void Mat::locateROI( Size& wholeSize, Point& ofs ) const
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = data - datastart, delta2 = dataend - datastart;

    if( delta1 == 0 )
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1/step[0]);
        ofs.x = (int)((delta1 - step[0]*ofs.y)/esz);
        CV_DbgAssert( data == datastart + ofs.y*step[0] + ofs.x*esz );
    }
    minstep = (ofs.x + cols)*esz;
    wholeSize.height = (int)((delta2 - minstep)/step[0] + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}

Mat& Mat::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    CV_Assert( dims <= 2 && step[0] > 0 );
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    data += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    size.p[0] = rows; size.p[1] = cols;
    if( esz*cols == step[0] || rows == 1 )
        flags |= CONTINUOUS_FLAG;
    else
        flags &= ~CONTINUOUS_FLAG;
    return *this;
}

}

void cv::extractImageCOI(const CvArr* arr, OutputArray _ch, int coi)
{
    Mat mat = cvarrToMat(arr, false, true, 1);
    _ch.create(mat.dims, mat.size, mat.depth());
    Mat ch = _ch.getMat();
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(0 <= coi && coi < mat.channels());
    int _pairs[] = { coi, 0 };
    mixChannels( &mat, 1, &ch, 1, _pairs, 1 );
}

void cv::insertImageCOI(InputArray _ch, CvArr* arr, int coi)
{
    Mat ch = _ch.getMat(), mat = cvarrToMat(arr, false, true, 1);
    if(coi < 0)
    {
        CV_Assert( CV_IS_IMAGE(arr) );
        coi = cvGetImageCOI((const IplImage*)arr)-1;
    }
    CV_Assert(ch.size == mat.size && ch.depth() == mat.depth() && 0 <= coi && coi < mat.channels());
    int _pairs[] = { 0, coi };
    mixChannels( &ch, 1, &mat, 1, _pairs, 1 );
}

namespace cv
{

Mat Mat::reshape(int new_cn, int new_rows) const
{
    int cn = channels();
    Mat hdr = *this;

    if( dims > 2 && new_rows == 0 && new_cn != 0 && size[dims-1]*cn % new_cn == 0 )
    {
        hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
        hdr.step[dims-1] = CV_ELEM_SIZE(hdr.flags);
        hdr.size[dims-1] = hdr.size[dims-1]*cn / new_cn;
        return hdr;
    }

    CV_Assert( dims <= 2 );

    if( new_cn == 0 )
        new_cn = cn;

    int total_width = cols * cn;

    if( (new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = rows * total_width / new_cn;

    if( new_rows != 0 && new_rows != rows )
    {
        int total_size = total_width * rows;
        if( !isContinuous() )
            CV_Error( CV_BadStep,
            "The matrix is not continuous, thus its number of rows can not be changed" );

        if( (unsigned)new_rows > (unsigned)total_size )
            CV_Error( CV_StsOutOfRange, "Bad new number of rows" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( CV_StsBadArg, "The total number of matrix elements "
                                    "is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step[0] = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( CV_BadNumChannels,
        "The total width is not divisible by the new number of channels" );

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    hdr.step[1] = CV_ELEM_SIZE(hdr.flags);
    return hdr;
}


int Mat::checkVector(int _elemChannels, int _depth, bool _requireContinuous) const
{
    return (depth() == _depth || _depth <= 0) &&
        (isContinuous() || !_requireContinuous) &&
        ((dims == 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
                        (cols == _elemChannels && channels() == 1))) ||
        (dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
         (isContinuous() || step.p[1] == step.p[2]*size.p[2])))
    ? (int)(total()*channels()/_elemChannels) : -1;
}


void scalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
{
    int i, depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert(cn <= 4);
    switch(depth)
    {
    case CV_8U:
        {
        uchar* buf = (uchar*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<uchar>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_8S:
        {
        schar* buf = (schar*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<schar>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_16U:
        {
        ushort* buf = (ushort*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<ushort>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_16S:
        {
        short* buf = (short*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<short>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_32S:
        {
        int* buf = (int*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<int>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_32F:
        {
        float* buf = (float*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<float>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        }
        break;
    case CV_64F:
        {
        double* buf = (double*)_buf;
        for(i = 0; i < cn; i++)
            buf[i] = saturate_cast<double>(s.val[i]);
        for(; i < unroll_to; i++)
            buf[i] = buf[i-cn];
        break;
        }
    default:
        CV_Error(CV_StsUnsupportedFormat,"");
    }
}


/*************************************************************************************************\
                                        Input/Output Array
\*************************************************************************************************/

_InputArray::_InputArray() : flags(0), obj(0) {}
#ifdef OPENCV_CAN_BREAK_BINARY_COMPATIBILITY
_InputArray::~_InputArray() {}
#endif
_InputArray::_InputArray(const Mat& m) : flags(MAT), obj((void*)&m) {}
_InputArray::_InputArray(const vector<Mat>& vec) : flags(STD_VECTOR_MAT), obj((void*)&vec) {}
_InputArray::_InputArray(const double& val) : flags(FIXED_TYPE + FIXED_SIZE + MATX + CV_64F), obj((void*)&val), sz(Size(1,1)) {}
_InputArray::_InputArray(const MatExpr& expr) : flags(FIXED_TYPE + FIXED_SIZE + EXPR), obj((void*)&expr) {}
// < Deprecated
_InputArray::_InputArray(const GlBuffer&) : flags(0), obj(0) {}
_InputArray::_InputArray(const GlTexture&) : flags(0), obj(0) {}
// >
_InputArray::_InputArray(const gpu::GpuMat& d_mat) : flags(GPU_MAT), obj((void*)&d_mat) {}
_InputArray::_InputArray(const ogl::Buffer& buf) : flags(OPENGL_BUFFER), obj((void*)&buf) {}
_InputArray::_InputArray(const ogl::Texture2D& tex) : flags(OPENGL_TEXTURE), obj((void*)&tex) {}

Mat _InputArray::getMat(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        const Mat* m = (const Mat*)obj;
        if( i < 0 )
            return *m;
        return m->row(i);
    }

    if( k == EXPR )
    {
        CV_Assert( i < 0 );
        return (Mat)*((const MatExpr*)obj);
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        return Mat(sz, flags, obj);
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        int t = CV_MAT_TYPE(flags);
        const vector<uchar>& v = *(const vector<uchar>*)obj;

        return !v.empty() ? Mat(size(), t, (void*)&v[0]) : Mat();
    }

    if( k == NONE )
        return Mat();

    if( k == STD_VECTOR_VECTOR )
    {
        int t = type(i);
        const vector<vector<uchar> >& vv = *(const vector<vector<uchar> >*)obj;
        CV_Assert( 0 <= i && i < (int)vv.size() );
        const vector<uchar>& v = vv[i];

        return !v.empty() ? Mat(size(i), t, (void*)&v[0]) : Mat();
    }

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    CV_Assert( k == STD_VECTOR_MAT );
    //if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& v = *(const vector<Mat>*)obj;
        CV_Assert( 0 <= i && i < (int)v.size() );

        return v[i];
    }
}


void _InputArray::getMatVector(vector<Mat>& mv) const
{
    int k = kind();

    if( k == MAT )
    {
        const Mat& m = *(const Mat*)obj;
        int i, n = (int)m.size[0];
        mv.resize(n);

        for( i = 0; i < n; i++ )
            mv[i] = m.dims == 2 ? Mat(1, m.cols, m.type(), (void*)m.ptr(i)) :
                Mat(m.dims-1, &m.size[1], m.type(), (void*)m.ptr(i), &m.step[1]);
        return;
    }

    if( k == EXPR )
    {
        Mat m = *(const MatExpr*)obj;
        int i, n = m.size[0];
        mv.resize(n);

        for( i = 0; i < n; i++ )
            mv[i] = m.row(i);
        return;
    }

    if( k == MATX )
    {
        size_t i, n = sz.height, esz = CV_ELEM_SIZE(flags);
        mv.resize(n);

        for( i = 0; i < n; i++ )
            mv[i] = Mat(1, sz.width, CV_MAT_TYPE(flags), (uchar*)obj + esz*sz.width*i);
        return;
    }

    if( k == STD_VECTOR )
    {
        const vector<uchar>& v = *(const vector<uchar>*)obj;

        size_t i, n = v.size(), esz = CV_ELEM_SIZE(flags);
        int t = CV_MAT_DEPTH(flags), cn = CV_MAT_CN(flags);
        mv.resize(n);

        for( i = 0; i < n; i++ )
            mv[i] = Mat(1, cn, t, (void*)(&v[0] + esz*i));
        return;
    }

    if( k == NONE )
    {
        mv.clear();
        return;
    }

    if( k == STD_VECTOR_VECTOR )
    {
        const vector<vector<uchar> >& vv = *(const vector<vector<uchar> >*)obj;
        int i, n = (int)vv.size();
        int t = CV_MAT_TYPE(flags);
        mv.resize(n);

        for( i = 0; i < n; i++ )
        {
            const vector<uchar>& v = vv[i];
            mv[i] = Mat(size(i), t, (void*)&v[0]);
        }
        return;
    }

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    CV_Assert( k == STD_VECTOR_MAT );
    //if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& v = *(const vector<Mat>*)obj;
        mv.resize(v.size());
        std::copy(v.begin(), v.end(), mv.begin());
        return;
    }
}

GlBuffer _InputArray::getGlBuffer() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return GlBuffer(GlBuffer::ARRAY_BUFFER);
}

GlTexture _InputArray::getGlTexture() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return GlTexture();
}

gpu::GpuMat _InputArray::getGpuMat() const
{
    int k = kind();

    CV_Assert(k == GPU_MAT);

    const gpu::GpuMat* d_mat = (const gpu::GpuMat*)obj;
    return *d_mat;
}

ogl::Buffer _InputArray::getOGlBuffer() const
{
    int k = kind();

    CV_Assert(k == OPENGL_BUFFER);

    const ogl::Buffer* gl_buf = (const ogl::Buffer*)obj;
    return *gl_buf;
}

ogl::Texture2D _InputArray::getOGlTexture2D() const
{
    int k = kind();

    CV_Assert(k == OPENGL_TEXTURE);

    const ogl::Texture2D* gl_tex = (const ogl::Texture2D*)obj;
    return *gl_tex;
}

int _InputArray::kind() const
{
    return flags & KIND_MASK;
}

Size _InputArray::size(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->size();
    }

    if( k == EXPR )
    {
        CV_Assert( i < 0 );
        return ((const MatExpr*)obj)->size();
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        return sz;
    }

    if( k == STD_VECTOR )
    {
        CV_Assert( i < 0 );
        const vector<uchar>& v = *(const vector<uchar>*)obj;
        const vector<int>& iv = *(const vector<int>*)obj;
        size_t szb = v.size(), szi = iv.size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == NONE )
        return Size();

    if( k == STD_VECTOR_VECTOR )
    {
        const vector<vector<uchar> >& vv = *(const vector<vector<uchar> >*)obj;
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );
        const vector<vector<int> >& ivv = *(const vector<vector<int> >*)obj;

        size_t szb = vv[i].size(), szi = ivv[i].size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb/CV_ELEM_SIZE(flags)), 1);
    }

    if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& vv = *(const vector<Mat>*)obj;
        if( i < 0 )
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        CV_Assert( i < (int)vv.size() );

        return vv[i].size();
    }

    if( k == OPENGL_BUFFER )
    {
        CV_Assert( i < 0 );
        const ogl::Buffer* buf = (const ogl::Buffer*)obj;
        return buf->size();
    }

    if( k == OPENGL_TEXTURE )
    {
        CV_Assert( i < 0 );
        const ogl::Texture2D* tex = (const ogl::Texture2D*)obj;
        return tex->size();
    }

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    CV_Assert( k == GPU_MAT );
    //if( k == GPU_MAT )
    {
        CV_Assert( i < 0 );
        const gpu::GpuMat* d_mat = (const gpu::GpuMat*)obj;
        return d_mat->size();
    }
}

size_t _InputArray::total(int i) const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        return ((const Mat*)obj)->total();
    }

    if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& vv = *(const vector<Mat>*)obj;
        if( i < 0 )
            return vv.size();

        CV_Assert( i < (int)vv.size() );
        return vv[i].total();
    }

    return size(i).area();
}

int _InputArray::type(int i) const
{
    int k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->type();

    if( k == EXPR )
        return ((const MatExpr*)obj)->type();

    if( k == MATX || k == STD_VECTOR || k == STD_VECTOR_VECTOR )
        return CV_MAT_TYPE(flags);

    if( k == NONE )
        return -1;

    if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& vv = *(const vector<Mat>*)obj;
        CV_Assert( i < (int)vv.size() );

        return vv[i >= 0 ? i : 0].type();
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->type();

    CV_Assert( k == GPU_MAT );
    //if( k == GPU_MAT )
        return ((const gpu::GpuMat*)obj)->type();
}

int _InputArray::depth(int i) const
{
    return CV_MAT_DEPTH(type(i));
}

int _InputArray::channels(int i) const
{
    return CV_MAT_CN(type(i));
}

bool _InputArray::empty() const
{
    int k = kind();

    if( k == MAT )
        return ((const Mat*)obj)->empty();

    if( k == EXPR )
        return false;

    if( k == MATX )
        return false;

    if( k == STD_VECTOR )
    {
        const vector<uchar>& v = *(const vector<uchar>*)obj;
        return v.empty();
    }

    if( k == NONE )
        return true;

    if( k == STD_VECTOR_VECTOR )
    {
        const vector<vector<uchar> >& vv = *(const vector<vector<uchar> >*)obj;
        return vv.empty();
    }

    if( k == STD_VECTOR_MAT )
    {
        const vector<Mat>& vv = *(const vector<Mat>*)obj;
        return vv.empty();
    }

    if( k == OPENGL_BUFFER )
        return ((const ogl::Buffer*)obj)->empty();

    if( k == OPENGL_TEXTURE )
        return ((const ogl::Texture2D*)obj)->empty();

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    CV_Assert( k == GPU_MAT );
    //if( k == GPU_MAT )
        return ((const gpu::GpuMat*)obj)->empty();
}


_OutputArray::_OutputArray() {}
#ifdef OPENCV_CAN_BREAK_BINARY_COMPATIBILITY
_OutputArray::~_OutputArray() {}
#endif
_OutputArray::_OutputArray(Mat& m) : _InputArray(m) {}
_OutputArray::_OutputArray(vector<Mat>& vec) : _InputArray(vec) {}
_OutputArray::_OutputArray(gpu::GpuMat& d_mat) : _InputArray(d_mat) {}
_OutputArray::_OutputArray(ogl::Buffer& buf) : _InputArray(buf) {}
_OutputArray::_OutputArray(ogl::Texture2D& tex) : _InputArray(tex) {}

_OutputArray::_OutputArray(const Mat& m) : _InputArray(m) {flags |= FIXED_SIZE|FIXED_TYPE;}
_OutputArray::_OutputArray(const vector<Mat>& vec) : _InputArray(vec) {flags |= FIXED_SIZE;}
_OutputArray::_OutputArray(const gpu::GpuMat& d_mat) : _InputArray(d_mat) {flags |= FIXED_SIZE|FIXED_TYPE;}
_OutputArray::_OutputArray(const ogl::Buffer& buf) : _InputArray(buf) {flags |= FIXED_SIZE|FIXED_TYPE;}
_OutputArray::_OutputArray(const ogl::Texture2D& tex) : _InputArray(tex) {flags |= FIXED_SIZE|FIXED_TYPE;}


bool _OutputArray::fixedSize() const
{
    return (flags & FIXED_SIZE) == FIXED_SIZE;
}

bool _OutputArray::fixedType() const
{
    return (flags & FIXED_TYPE) == FIXED_TYPE;
}

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == _sz);
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((gpu::GpuMat*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((gpu::GpuMat*)obj)->type() == mtype);
        ((gpu::GpuMat*)obj)->create(_sz, mtype);
        return;
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == _sz);
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
        ((ogl::Buffer*)obj)->create(_sz, mtype);
        return;
    }
    int sizes[] = {_sz.height, _sz.width};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int rows, int cols, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if( k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((Mat*)obj)->size.operator()() == Size(cols, rows));
        CV_Assert(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(rows, cols, mtype);
        return;
    }
    if( k == GPU_MAT && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((gpu::GpuMat*)obj)->size() == Size(cols, rows));
        CV_Assert(!fixedType() || ((gpu::GpuMat*)obj)->type() == mtype);
        ((gpu::GpuMat*)obj)->create(rows, cols, mtype);
        return;
    }
    if( k == OPENGL_BUFFER && i < 0 && !allowTransposed && fixedDepthMask == 0 )
    {
        CV_Assert(!fixedSize() || ((ogl::Buffer*)obj)->size() == Size(cols, rows));
        CV_Assert(!fixedType() || ((ogl::Buffer*)obj)->type() == mtype);
        ((ogl::Buffer*)obj)->create(rows, cols, mtype);
        return;
    }
    int sizes[] = {rows, cols};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int dims, const int* sizes, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    mtype = CV_MAT_TYPE(mtype);

    if( k == MAT )
    {
        CV_Assert( i < 0 );
        Mat& m = *(Mat*)obj;
        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( dims == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(CV_MAT_TYPE(mtype) == m.type());
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == dims);
            for(int j = 0; j < dims; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }
        m.create(dims, sizes, mtype);
        return;
    }

    if( k == MATX )
    {
        CV_Assert( i < 0 );
        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0) );
        CV_Assert( dims == 2 && ((sizes[0] == sz.height && sizes[1] == sz.width) ||
                                 (allowTransposed && sizes[0] == sz.width && sizes[1] == sz.height)));
        return;
    }

    if( k == STD_VECTOR || k == STD_VECTOR_VECTOR )
    {
        CV_Assert( dims == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
        size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
        vector<uchar>* v = (vector<uchar>*)obj;

        if( k == STD_VECTOR_VECTOR )
        {
            vector<vector<uchar> >& vv = *(vector<vector<uchar> >*)obj;
            if( i < 0 )
            {
                CV_Assert(!fixedSize() || len == vv.size());
                vv.resize(len);
                return;
            }
            CV_Assert( i < (int)vv.size() );
            v = &vv[i];
        }
        else
            CV_Assert( i < 0 );

        int type0 = CV_MAT_TYPE(flags);
        CV_Assert( mtype == type0 || (CV_MAT_CN(mtype) == CV_MAT_CN(type0) && ((1 << type0) & fixedDepthMask) != 0) );

        int esz = CV_ELEM_SIZE(type0);
        CV_Assert(!fixedSize() || len == ((vector<uchar>*)v)->size() / esz);
        switch( esz )
        {
        case 1:
            ((vector<uchar>*)v)->resize(len);
            break;
        case 2:
            ((vector<Vec2b>*)v)->resize(len);
            break;
        case 3:
            ((vector<Vec3b>*)v)->resize(len);
            break;
        case 4:
            ((vector<int>*)v)->resize(len);
            break;
        case 6:
            ((vector<Vec3s>*)v)->resize(len);
            break;
        case 8:
            ((vector<Vec2i>*)v)->resize(len);
            break;
        case 12:
            ((vector<Vec3i>*)v)->resize(len);
            break;
        case 16:
            ((vector<Vec4i>*)v)->resize(len);
            break;
        case 24:
            ((vector<Vec6i>*)v)->resize(len);
            break;
        case 32:
            ((vector<Vec8i>*)v)->resize(len);
            break;
        case 36:
            ((vector<Vec<int, 9> >*)v)->resize(len);
            break;
        case 48:
            ((vector<Vec<int, 12> >*)v)->resize(len);
            break;
        case 64:
            ((vector<Vec<int, 16> >*)v)->resize(len);
            break;
        case 128:
            ((vector<Vec<int, 32> >*)v)->resize(len);
            break;
        case 256:
            ((vector<Vec<int, 64> >*)v)->resize(len);
            break;
        case 512:
            ((vector<Vec<int, 128> >*)v)->resize(len);
            break;
        default:
            CV_Error_(CV_StsBadArg, ("Vectors with element size %d are not supported. Please, modify OutputArray::create()\n", esz));
        }
        return;
    }

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    if( k == NONE )
    {
        CV_Error(CV_StsNullPtr, "create() called for the missing output array" );
        return;
    }

    CV_Assert( k == STD_VECTOR_MAT );
    //if( k == STD_VECTOR_MAT )
    {
        vector<Mat>& v = *(vector<Mat>*)obj;

        if( i < 0 )
        {
            CV_Assert( dims == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0]*sizes[1] == 0) );
            size_t len = sizes[0]*sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = v.size();

            CV_Assert(!fixedSize() || len == len0);
            v.resize(len);
            if( fixedType() )
            {
                int _type = CV_MAT_TYPE(flags);
                for( size_t j = len0; j < len; j++ )
                {
                    if( v[j].type() == _type )
                        continue;
                    CV_Assert( v[j].empty() );
                    v[j].flags = (v[j].flags & ~CV_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        CV_Assert( i < (int)v.size() );
        Mat& m = v[i];

        if( allowTransposed )
        {
            if( !m.isContinuous() )
            {
                CV_Assert(!fixedType() && !fixedSize());
                m.release();
            }

            if( dims == 2 && m.dims == 2 && m.data &&
                m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0] )
                return;
        }

        if(fixedType())
        {
            if(CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0 )
                mtype = m.type();
            else
                CV_Assert(!fixedType() || (CV_MAT_CN(mtype) == m.channels() && ((1 << CV_MAT_TYPE(flags)) & fixedDepthMask) != 0));
        }
        if(fixedSize())
        {
            CV_Assert(m.dims == dims);
            for(int j = 0; j < dims; ++j)
                CV_Assert(m.size[j] == sizes[j]);
        }

        m.create(dims, sizes, mtype);
    }
}

void _OutputArray::release() const
{
    CV_Assert(!fixedSize());

    int k = kind();

    if( k == MAT )
    {
        ((Mat*)obj)->release();
        return;
    }

    if( k == GPU_MAT )
    {
        ((gpu::GpuMat*)obj)->release();
        return;
    }

    if( k == OPENGL_BUFFER )
    {
        ((ogl::Buffer*)obj)->release();
        return;
    }

    if( k == OPENGL_TEXTURE )
    {
        ((ogl::Texture2D*)obj)->release();
        return;
    }

    if( k == NONE )
        return;

    if( k == STD_VECTOR )
    {
        create(Size(), CV_MAT_TYPE(flags));
        return;
    }

    if( k == STD_VECTOR_VECTOR )
    {
        ((vector<vector<uchar> >*)obj)->clear();
        return;
    }

    if( k == OCL_MAT )
    {
        CV_Error(CV_StsNotImplemented, "This method is not implemented for oclMat yet");
    }

    CV_Assert( k == STD_VECTOR_MAT );
    //if( k == STD_VECTOR_MAT )
    {
        ((vector<Mat>*)obj)->clear();
    }
}

void _OutputArray::clear() const
{
    int k = kind();

    if( k == MAT )
    {
        CV_Assert(!fixedSize());
        ((Mat*)obj)->resize(0);
        return;
    }

    release();
}

bool _OutputArray::needed() const
{
    return kind() != NONE;
}

Mat& _OutputArray::getMatRef(int i) const
{
    int k = kind();
    if( i < 0 )
    {
        CV_Assert( k == MAT );
        return *(Mat*)obj;
    }
    else
    {
        CV_Assert( k == STD_VECTOR_MAT );
        vector<Mat>& v = *(vector<Mat>*)obj;
        CV_Assert( i < (int)v.size() );
        return v[i];
    }
}

gpu::GpuMat& _OutputArray::getGpuMatRef() const
{
    int k = kind();
    CV_Assert( k == GPU_MAT );
    return *(gpu::GpuMat*)obj;
}

ogl::Buffer& _OutputArray::getOGlBufferRef() const
{
    int k = kind();
    CV_Assert( k == OPENGL_BUFFER );
    return *(ogl::Buffer*)obj;
}

ogl::Texture2D& _OutputArray::getOGlTexture2DRef() const
{
    int k = kind();
    CV_Assert( k == OPENGL_TEXTURE );
    return *(ogl::Texture2D*)obj;
}

static _OutputArray _none;
OutputArray noArray() { return _none; }

}

/*************************************************************************************************\
                                        Matrix Operations
\*************************************************************************************************/

void cv::hconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalCols = 0, cols = 0;
    size_t i;
    for( i = 0; i < nsrc; i++ )
    {
        CV_Assert( !src[i].empty() && src[i].dims <= 2 &&
                   src[i].rows == src[0].rows &&
                   src[i].type() == src[0].type());
        totalCols += src[i].cols;
    }
    _dst.create( src[0].rows, totalCols, src[0].type());
    Mat dst = _dst.getMat();
    for( i = 0; i < nsrc; i++ )
    {
        Mat dpart = dst(Rect(cols, 0, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        cols += src[i].cols;
    }
}

void cv::hconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    Mat src[] = {src1.getMat(), src2.getMat()};
    hconcat(src, 2, dst);
}

void cv::hconcat(InputArray _src, OutputArray dst)
{
    vector<Mat> src;
    _src.getMatVector(src);
    hconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

void cv::vconcat(const Mat* src, size_t nsrc, OutputArray _dst)
{
    if( nsrc == 0 || !src )
    {
        _dst.release();
        return;
    }

    int totalRows = 0, rows = 0;
    size_t i;
    for( i = 0; i < nsrc; i++ )
    {
        CV_Assert( !src[i].empty() && src[i].dims <= 2 &&
                  src[i].cols == src[0].cols &&
                  src[i].type() == src[0].type());
        totalRows += src[i].rows;
    }
    _dst.create( totalRows, src[0].cols, src[0].type());
    Mat dst = _dst.getMat();
    for( i = 0; i < nsrc; i++ )
    {
        Mat dpart(dst, Rect(0, rows, src[i].cols, src[i].rows));
        src[i].copyTo(dpart);
        rows += src[i].rows;
    }
}

void cv::vconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    Mat src[] = {src1.getMat(), src2.getMat()};
    vconcat(src, 2, dst);
}

void cv::vconcat(InputArray _src, OutputArray dst)
{
    vector<Mat> src;
    _src.getMatVector(src);
    vconcat(!src.empty() ? &src[0] : 0, src.size(), dst);
}

//////////////////////////////////////// set identity ////////////////////////////////////////////
void cv::setIdentity( InputOutputArray _m, const Scalar& s )
{
    Mat m = _m.getMat();
    CV_Assert( m.dims <= 2 );
    int i, j, rows = m.rows, cols = m.cols, type = m.type();

    if( type == CV_32FC1 )
    {
        float* data = (float*)m.data;
        float val = (float)s[0];
        size_t step = m.step/sizeof(data[0]);

        for( i = 0; i < rows; i++, data += step )
        {
            for( j = 0; j < cols; j++ )
                data[j] = 0;
            if( i < cols )
                data[i] = val;
        }
    }
    else if( type == CV_64FC1 )
    {
        double* data = (double*)m.data;
        double val = s[0];
        size_t step = m.step/sizeof(data[0]);

        for( i = 0; i < rows; i++, data += step )
        {
            for( j = 0; j < cols; j++ )
                data[j] = j == i ? val : 0;
        }
    }
    else
    {
        m = Scalar(0);
        m.diag() = s;
    }
}

//////////////////////////////////////////// trace ///////////////////////////////////////////

cv::Scalar cv::trace( InputArray _m )
{
    Mat m = _m.getMat();
    CV_Assert( m.dims <= 2 );
    int i, type = m.type();
    int nm = std::min(m.rows, m.cols);

    if( type == CV_32FC1 )
    {
        const float* ptr = (const float*)m.data;
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    if( type == CV_64FC1 )
    {
        const double* ptr = (const double*)m.data;
        size_t step = m.step/sizeof(ptr[0]) + 1;
        double _s = 0;
        for( i = 0; i < nm; i++ )
            _s += ptr[i*step];
        return _s;
    }

    return cv::sum(m.diag());
}

////////////////////////////////////// transpose /////////////////////////////////////////

namespace cv
{

template<typename T> static void
transpose_( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz )
{
    int i=0, j, m = sz.width, n = sz.height;

    #if CV_ENABLE_UNROLLED
    for(; i <= m - 4; i += 4 )
    {
        T* d0 = (T*)(dst + dstep*i);
        T* d1 = (T*)(dst + dstep*(i+1));
        T* d2 = (T*)(dst + dstep*(i+2));
        T* d3 = (T*)(dst + dstep*(i+3));

        for( j = 0; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
            d1[j] = s0[1]; d1[j+1] = s1[1]; d1[j+2] = s2[1]; d1[j+3] = s3[1];
            d2[j] = s0[2]; d2[j+1] = s1[2]; d2[j+2] = s2[2]; d2[j+3] = s3[2];
            d3[j] = s0[3]; d3[j+1] = s1[3]; d3[j+2] = s2[3]; d3[j+3] = s3[3];
        }

        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
        }
    }
    #endif
    for( ; i < m; i++ )
    {
        T* d0 = (T*)(dst + dstep*i);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
        }
        #endif
        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0];
        }
    }
}

template<typename T> static void
transposeI_( uchar* data, size_t step, int n )
{
    int i, j;
    for( i = 0; i < n; i++ )
    {
        T* row = (T*)(data + step*i);
        uchar* data1 = data + i*sizeof(T);
        for( j = i+1; j < n; j++ )
            std::swap( row[j], *(T*)(data1 + step*j) );
    }
}

typedef void (*TransposeFunc)( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz );
typedef void (*TransposeInplaceFunc)( uchar* data, size_t step, int n );

#define DEF_TRANSPOSE_FUNC(suffix, type) \
static void transpose_##suffix( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz ) \
{ transpose_<type>(src, sstep, dst, dstep, sz); } \
\
static void transposeI_##suffix( uchar* data, size_t step, int n ) \
{ transposeI_<type>(data, step, n); }

DEF_TRANSPOSE_FUNC(8u, uchar)
DEF_TRANSPOSE_FUNC(16u, ushort)
DEF_TRANSPOSE_FUNC(8uC3, Vec3b)
DEF_TRANSPOSE_FUNC(32s, int)
DEF_TRANSPOSE_FUNC(16uC3, Vec3s)
DEF_TRANSPOSE_FUNC(32sC2, Vec2i)
DEF_TRANSPOSE_FUNC(32sC3, Vec3i)
DEF_TRANSPOSE_FUNC(32sC4, Vec4i)
DEF_TRANSPOSE_FUNC(32sC6, Vec6i)
DEF_TRANSPOSE_FUNC(32sC8, Vec8i)

static TransposeFunc transposeTab[] =
{
    0, transpose_8u, transpose_16u, transpose_8uC3, transpose_32s, 0, transpose_16uC3, 0,
    transpose_32sC2, 0, 0, 0, transpose_32sC3, 0, 0, 0, transpose_32sC4,
    0, 0, 0, 0, 0, 0, 0, transpose_32sC6, 0, 0, 0, 0, 0, 0, 0, transpose_32sC8
};

static TransposeInplaceFunc transposeInplaceTab[] =
{
    0, transposeI_8u, transposeI_16u, transposeI_8uC3, transposeI_32s, 0, transposeI_16uC3, 0,
    transposeI_32sC2, 0, 0, 0, transposeI_32sC3, 0, 0, 0, transposeI_32sC4,
    0, 0, 0, 0, 0, 0, 0, transposeI_32sC6, 0, 0, 0, 0, 0, 0, 0, transposeI_32sC8
};

}

void cv::transpose( InputArray _src, OutputArray _dst )
{
    Mat src = _src.getMat();
    size_t esz = src.elemSize();
    CV_Assert( src.dims <= 2 && esz <= (size_t)32 );

    _dst.create(src.cols, src.rows, src.type());
    Mat dst = _dst.getMat();

    // handle the case of single-column/single-row matrices, stored in STL vectors.
    if( src.rows != dst.cols || src.cols != dst.rows )
    {
        CV_Assert( src.size() == dst.size() && (src.cols == 1 || src.rows == 1) );
        src.copyTo(dst);
        return;
    }

    if( dst.data == src.data )
    {
        TransposeInplaceFunc func = transposeInplaceTab[esz];
        CV_Assert( func != 0 );
        func( dst.data, dst.step, dst.rows );
    }
    else
    {
        TransposeFunc func = transposeTab[esz];
        CV_Assert( func != 0 );
        func( src.data, src.step, dst.data, dst.step, src.size() );
    }
}


////////////////////////////////////// completeSymm /////////////////////////////////////////

void cv::completeSymm( InputOutputArray _m, bool LtoR )
{
    Mat m = _m.getMat();
    size_t step = m.step, esz = m.elemSize();
    CV_Assert( m.dims <= 2 && m.rows == m.cols );

    int rows = m.rows;
    int j0 = 0, j1 = rows;

    uchar* data = m.data;
    for( int i = 0; i < rows; i++ )
    {
        if( !LtoR ) j1 = i; else j0 = i+1;
        for( int j = j0; j < j1; j++ )
            memcpy(data + (i*step + j*esz), data + (j*step + i*esz), esz);
    }
}


cv::Mat cv::Mat::cross(InputArray _m) const
{
    Mat m = _m.getMat();
    int tp = type(), d = CV_MAT_DEPTH(tp);
    CV_Assert( dims <= 2 && m.dims <= 2 && size() == m.size() && tp == m.type() &&
        ((rows == 3 && cols == 1) || (cols*channels() == 3 && rows == 1)));
    Mat result(rows, cols, tp);

    if( d == CV_32F )
    {
        const float *a = (const float*)data, *b = (const float*)m.data;
        float* c = (float*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }
    else if( d == CV_64F )
    {
        const double *a = (const double*)data, *b = (const double*)m.data;
        double* c = (double*)result.data;
        size_t lda = rows > 1 ? step/sizeof(a[0]) : 1;
        size_t ldb = rows > 1 ? m.step/sizeof(b[0]) : 1;

        c[0] = a[lda] * b[ldb*2] - a[lda*2] * b[ldb];
        c[1] = a[lda*2] * b[0] - a[0] * b[ldb*2];
        c[2] = a[0] * b[ldb] - a[lda] * b[0];
    }

    return result;
}


////////////////////////////////////////// reduce ////////////////////////////////////////////

namespace cv
{

template<typename T, typename ST, class Op> static void
reduceR_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    size.width *= srcmat.channels();
    AutoBuffer<WT> buffer(size.width);
    WT* buf = buffer;
    ST* dst = (ST*)dstmat.data;
    const T* src = (const T*)srcmat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    int i;
    Op op;

    for( i = 0; i < size.width; i++ )
        buf[i] = src[i];

    for( ; --size.height; )
    {
        src += srcstep;
        i = 0;
        #if CV_ENABLE_UNROLLED
        for(; i <= size.width - 4; i += 4 )
        {
            WT s0, s1;
            s0 = op(buf[i], (WT)src[i]);
            s1 = op(buf[i+1], (WT)src[i+1]);
            buf[i] = s0; buf[i+1] = s1;

            s0 = op(buf[i+2], (WT)src[i+2]);
            s1 = op(buf[i+3], (WT)src[i+3]);
            buf[i+2] = s0; buf[i+3] = s1;
        }
        #endif
        for( ; i < size.width; i++ )
            buf[i] = op(buf[i], (WT)src[i]);
    }

    for( i = 0; i < size.width; i++ )
        dst[i] = (ST)buf[i];
}


template<typename T, typename ST, class Op> static void
reduceC_( const Mat& srcmat, Mat& dstmat )
{
    typedef typename Op::rtype WT;
    Size size = srcmat.size();
    int i, k, cn = srcmat.channels();
    size.width *= cn;
    Op op;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        ST* dst = (ST*)(dstmat.data + dstmat.step*y);
        if( size.width == cn )
            for( k = 0; k < cn; k++ )
                dst[k] = src[k];
        else
        {
            for( k = 0; k < cn; k++ )
            {
                WT a0 = src[k], a1 = src[k+cn];
                for( i = 2*cn; i <= size.width - 4*cn; i += 4*cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                    a1 = op(a1, (WT)src[i+k+cn]);
                    a0 = op(a0, (WT)src[i+k+cn*2]);
                    a1 = op(a1, (WT)src[i+k+cn*3]);
                }

                for( ; i < size.width; i += cn )
                {
                    a0 = op(a0, (WT)src[i+k]);
                }
                a0 = op(a0, a1);
              dst[k] = (ST)a0;
            }
        }
    }
}

typedef void (*ReduceFunc)( const Mat& src, Mat& dst );

}

#define reduceSumR8u32s  reduceR_<uchar, int,   OpAdd<int> >
#define reduceSumR8u32f  reduceR_<uchar, float, OpAdd<int> >
#define reduceSumR8u64f  reduceR_<uchar, double,OpAdd<int> >
#define reduceSumR16u32f reduceR_<ushort,float, OpAdd<float> >
#define reduceSumR16u64f reduceR_<ushort,double,OpAdd<double> >
#define reduceSumR16s32f reduceR_<short, float, OpAdd<float> >
#define reduceSumR16s64f reduceR_<short, double,OpAdd<double> >
#define reduceSumR32f32f reduceR_<float, float, OpAdd<float> >
#define reduceSumR32f64f reduceR_<float, double,OpAdd<double> >
#define reduceSumR64f64f reduceR_<double,double,OpAdd<double> >

#define reduceMaxR8u  reduceR_<uchar, uchar, OpMax<uchar> >
#define reduceMaxR16u reduceR_<ushort,ushort,OpMax<ushort> >
#define reduceMaxR16s reduceR_<short, short, OpMax<short> >
#define reduceMaxR32f reduceR_<float, float, OpMax<float> >
#define reduceMaxR64f reduceR_<double,double,OpMax<double> >

#define reduceMinR8u  reduceR_<uchar, uchar, OpMin<uchar> >
#define reduceMinR16u reduceR_<ushort,ushort,OpMin<ushort> >
#define reduceMinR16s reduceR_<short, short, OpMin<short> >
#define reduceMinR32f reduceR_<float, float, OpMin<float> >
#define reduceMinR64f reduceR_<double,double,OpMin<double> >

#define reduceSumC8u32s  reduceC_<uchar, int,   OpAdd<int> >
#define reduceSumC8u32f  reduceC_<uchar, float, OpAdd<int> >
#define reduceSumC8u64f  reduceC_<uchar, double,OpAdd<int> >
#define reduceSumC16u32f reduceC_<ushort,float, OpAdd<float> >
#define reduceSumC16u64f reduceC_<ushort,double,OpAdd<double> >
#define reduceSumC16s32f reduceC_<short, float, OpAdd<float> >
#define reduceSumC16s64f reduceC_<short, double,OpAdd<double> >
#define reduceSumC32f32f reduceC_<float, float, OpAdd<float> >
#define reduceSumC32f64f reduceC_<float, double,OpAdd<double> >
#define reduceSumC64f64f reduceC_<double,double,OpAdd<double> >

#define reduceMaxC8u  reduceC_<uchar, uchar, OpMax<uchar> >
#define reduceMaxC16u reduceC_<ushort,ushort,OpMax<ushort> >
#define reduceMaxC16s reduceC_<short, short, OpMax<short> >
#define reduceMaxC32f reduceC_<float, float, OpMax<float> >
#define reduceMaxC64f reduceC_<double,double,OpMax<double> >

#define reduceMinC8u  reduceC_<uchar, uchar, OpMin<uchar> >
#define reduceMinC16u reduceC_<ushort,ushort,OpMin<ushort> >
#define reduceMinC16s reduceC_<short, short, OpMin<short> >
#define reduceMinC32f reduceC_<float, float, OpMin<float> >
#define reduceMinC64f reduceC_<double,double,OpMin<double> >

void cv::reduce(InputArray _src, OutputArray _dst, int dim, int op, int dtype)
{
    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 );
    int op0 = op;
    int stype = src.type(), sdepth = src.depth(), cn = src.channels();
    if( dtype < 0 )
        dtype = _dst.fixedType() ? _dst.type() : stype;
    int ddepth = CV_MAT_DEPTH(dtype);

    _dst.create(dim == 0 ? 1 : src.rows, dim == 0 ? src.cols : 1,
                CV_MAKETYPE(dtype >= 0 ? dtype : stype, cn));
    Mat dst = _dst.getMat(), temp = dst;

    CV_Assert( op == CV_REDUCE_SUM || op == CV_REDUCE_MAX ||
               op == CV_REDUCE_MIN || op == CV_REDUCE_AVG );
    CV_Assert( src.channels() == dst.channels() );

    if( op == CV_REDUCE_AVG )
    {
        op = CV_REDUCE_SUM;
        if( sdepth < CV_32S && ddepth < CV_32S )
        {
            temp.create(dst.rows, dst.cols, CV_32SC(cn));
            ddepth = CV_32S;
        }
    }

    ReduceFunc func = 0;
    if( dim == 0 )
    {
        if( op == CV_REDUCE_SUM )
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumR8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumR8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumR8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumR16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumR16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumR16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumR16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumR32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumR32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumR64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxR64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinR8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinR16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinR16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinR32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinR64f;
        }
    }
    else
    {
        if(op == CV_REDUCE_SUM)
        {
            if(sdepth == CV_8U && ddepth == CV_32S)
                func = GET_OPTIMIZED(reduceSumC8u32s);
            else if(sdepth == CV_8U && ddepth == CV_32F)
                func = reduceSumC8u32f;
            else if(sdepth == CV_8U && ddepth == CV_64F)
                func = reduceSumC8u64f;
            else if(sdepth == CV_16U && ddepth == CV_32F)
                func = reduceSumC16u32f;
            else if(sdepth == CV_16U && ddepth == CV_64F)
                func = reduceSumC16u64f;
            else if(sdepth == CV_16S && ddepth == CV_32F)
                func = reduceSumC16s32f;
            else if(sdepth == CV_16S && ddepth == CV_64F)
                func = reduceSumC16s64f;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceSumC32f32f);
            else if(sdepth == CV_32F && ddepth == CV_64F)
                func = reduceSumC32f64f;
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceSumC64f64f;
        }
        else if(op == CV_REDUCE_MAX)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMaxC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMaxC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMaxC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMaxC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMaxC64f;
        }
        else if(op == CV_REDUCE_MIN)
        {
            if(sdepth == CV_8U && ddepth == CV_8U)
                func = GET_OPTIMIZED(reduceMinC8u);
            else if(sdepth == CV_16U && ddepth == CV_16U)
                func = reduceMinC16u;
            else if(sdepth == CV_16S && ddepth == CV_16S)
                func = reduceMinC16s;
            else if(sdepth == CV_32F && ddepth == CV_32F)
                func = GET_OPTIMIZED(reduceMinC32f);
            else if(sdepth == CV_64F && ddepth == CV_64F)
                func = reduceMinC64f;
        }
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat,
                  "Unsupported combination of input and output array formats" );

    func( src, temp );

    if( op0 == CV_REDUCE_AVG )
        temp.convertTo(dst, dst.type(), 1./(dim == 0 ? src.rows : src.cols));
}


//////////////////////////////////////// sort ///////////////////////////////////////////

namespace cv
{

template<typename T> static void sort_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    T* bptr;
    int i, j, n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool inplace = src.data == dst.data;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
    }
    bptr = (T*)buf;

    for( i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        if( sortRows )
        {
            T* dptr = (T*)(dst.data + dst.step*i);
            if( !inplace )
            {
                const T* sptr = (const T*)(src.data + src.step*i);
                for( j = 0; j < len; j++ )
                    dptr[j] = sptr[j];
            }
            ptr = dptr;
        }
        else
        {
            for( j = 0; j < len; j++ )
                ptr[j] = ((const T*)(src.data + src.step*j))[i];
        }
        std::sort( ptr, ptr + len, LessThan<T>() );
        if( sortDescending )
            for( j = 0; j < len/2; j++ )
                std::swap(ptr[j], ptr[len-1-j]);
        if( !sortRows )
            for( j = 0; j < len; j++ )
                ((T*)(dst.data + dst.step*j))[i] = ptr[j];
    }
}


template<typename T> static void sortIdx_( const Mat& src, Mat& dst, int flags )
{
    AutoBuffer<T> buf;
    AutoBuffer<int> ibuf;
    T* bptr;
    int* _iptr;
    int i, j, n, len;
    bool sortRows = (flags & 1) == CV_SORT_EVERY_ROW;
    bool sortDescending = (flags & CV_SORT_DESCENDING) != 0;

    CV_Assert( src.data != dst.data );

    if( sortRows )
        n = src.rows, len = src.cols;
    else
    {
        n = src.cols, len = src.rows;
        buf.allocate(len);
        ibuf.allocate(len);
    }
    bptr = (T*)buf;
    _iptr = (int*)ibuf;

    for( i = 0; i < n; i++ )
    {
        T* ptr = bptr;
        int* iptr = _iptr;

        if( sortRows )
        {
            ptr = (T*)(src.data + src.step*i);
            iptr = (int*)(dst.data + dst.step*i);
        }
        else
        {
            for( j = 0; j < len; j++ )
                ptr[j] = ((const T*)(src.data + src.step*j))[i];
        }
        for( j = 0; j < len; j++ )
            iptr[j] = j;
        std::sort( iptr, iptr + len, LessThanIdx<T>(ptr) );
        if( sortDescending )
            for( j = 0; j < len/2; j++ )
                std::swap(iptr[j], iptr[len-1-j]);
        if( !sortRows )
            for( j = 0; j < len; j++ )
                ((int*)(dst.data + dst.step*j))[i] = iptr[j];
    }
}

typedef void (*SortFunc)(const Mat& src, Mat& dst, int flags);

}

void cv::sort( InputArray _src, OutputArray _dst, int flags )
{
    static SortFunc tab[] =
    {
        sort_<uchar>, sort_<schar>, sort_<ushort>, sort_<short>,
        sort_<int>, sort_<float>, sort_<double>, 0
    };
    Mat src = _src.getMat();
    SortFunc func = tab[src.depth()];
    CV_Assert( src.dims <= 2 && src.channels() == 1 && func != 0 );
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();
    func( src, dst, flags );
}

void cv::sortIdx( InputArray _src, OutputArray _dst, int flags )
{
    static SortFunc tab[] =
    {
        sortIdx_<uchar>, sortIdx_<schar>, sortIdx_<ushort>, sortIdx_<short>,
        sortIdx_<int>, sortIdx_<float>, sortIdx_<double>, 0
    };
    Mat src = _src.getMat();
    SortFunc func = tab[src.depth()];
    CV_Assert( src.dims <= 2 && src.channels() == 1 && func != 0 );

    Mat dst = _dst.getMat();
    if( dst.data == src.data )
        _dst.release();
    _dst.create( src.size(), CV_32S );
    dst = _dst.getMat();
    func( src, dst, flags );
}


////////////////////////////////////////// kmeans ////////////////////////////////////////////

namespace cv
{

static void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}

class KMeansPPDistanceComputer : public ParallelLoopBody
{
public:
    KMeansPPDistanceComputer( float *_tdist2,
                              const float *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci )
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci) { }

    void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        for ( int i = begin; i<end; i++ )
        {
            tdist2[i] = std::min(normL2Sqr_(data + step*i, data + stepci, dims), dist[i]);
        }
    }

private:
    KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const float *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
};

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const float* data = _data.ptr<float>(0);
    size_t step = _data.step/sizeof(data[0]);
    vector<int> _centers(K);
    int* centers = &_centers[0];
    vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
        dist[i] = normL2Sqr_(data + step*i, data + step*centers[0], dims);
        sum0 += dist[i];
    }

    for( k = 1; k < K; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;

            parallel_for_(Range(0, N),
                         KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
            for( i = 0; i < N; i++ )
            {
                s += tdist2[i];
            }

            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
}

class KMeansDistanceComputer : public ParallelLoopBody
{
public:
    KMeansDistanceComputer( double *_distances,
                            int *_labels,
                            const Mat& _data,
                            const Mat& _centers )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers)
    {
    }

    void operator()( const Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        const float *sample;
        for( int i = begin; i<end; ++i)
        {
            sample = data.ptr<float>(i);
            int k_best = 0;
            double min_dist = DBL_MAX;

            for( int k = 0; k < K; k++ )
            {
                const float* center = centers.ptr<float>(k);
                const double dist = normL2Sqr_(sample, center, dims);

                if( min_dist > dist )
                {
                    min_dist = dist;
                    k_best = k;
                }
            }

            distances[i] = min_dist;
            labels[i] = k_best;
        }
    }

private:
    KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const Mat& data;
    const Mat& centers;
};

}

double cv::kmeans( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers )
{
    const int SPP_TRIALS = 3;
    Mat data0 = _data.getMat();
    bool isrow = data0.rows == 1 && data0.channels() > 1;
    int N = !isrow ? data0.rows : data0.cols;
    int dims = (!isrow ? data0.cols : 1)*data0.channels();
    int type = data0.depth();

    attempts = std::max(attempts, 1);
    CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    Mat data(N, dims, CV_32F, data0.data, isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

    _bestLabels.create(N, 1, CV_32S, -1, true);

    Mat _labels, best_labels = _bestLabels.getMat();
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels);
    }
    else
    {
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S);
        _labels.create(best_labels.size(), best_labels.type());
    }
    int* labels = _labels.ptr<int>();

    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
    vector<int> counters(K);
    vector<Vec2f> _box(dims);
    Vec2f* box = &_box[0];
    double best_compactness = DBL_MAX, compactness = 0;
    RNG& rng = theRNG();
    int a, iter, i, j, k;

    if( criteria.type & TermCriteria::EPS )
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    for( a = 0; a < attempts; a++ )
    {
        double max_center_shift = DBL_MAX;
        for( iter = 0;; )
        {
            swap(centers, old_centers);

            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
            {
                if( flags & KMEANS_PP_CENTERS )
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng);
                }
            }
            else
            {
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                {
                    for( i = 0; i < N; i++ )
                        CV_Assert( (unsigned)labels[i] < (unsigned)K );
                }

                // compute centers
                centers = Scalar(0);
                for( k = 0; k < K; k++ )
                    counters[k] = 0;

                for( i = 0; i < N; i++ )
                {
                    sample = data.ptr<float>(i);
                    k = labels[i];
                    float* center = centers.ptr<float>(k);
                    j=0;
                    #if CV_ENABLE_UNROLLED
                    for(; j <= dims - 4; j += 4 )
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    #endif
                    for( ; j < dims; j++ )
                        center[j] += sample[j];
                    counters[k]++;
                }

                if( iter > 0 )
                    max_center_shift = 0;

                for( k = 0; k < K; k++ )
                {
                    if( counters[k] != 0 )
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0;
                    for( int k1 = 1; k1 < K; k1++ )
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center = centers.ptr<float>(k);
                    float* old_center = centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];
                    for( j = 0; j < dims; j++ )
                        _old_center[j] = old_center[j]*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);
                        double dist = normL2Sqr_(sample, _old_center, dims);

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    for( j = 0; j < dims; j++ )
                    {
                        old_center[j] -= sample[j];
                        new_center[j] += sample[j];
                    }
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/counters[k];
                    for( j = 0; j < dims; j++ )
                        center[j] *= scale;

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            Mat dists(1, N, CV_64F);
            double* dist = dists.ptr<double>(0);
            parallel_for_(Range(0, N),
                         KMeansDistanceComputer(dist, labels, data, centers));
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                compactness += dist[i];
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
                centers.copyTo(_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}


CV_IMPL void cvSetIdentity( CvArr* arr, CvScalar value )
{
    cv::Mat m = cv::cvarrToMat(arr);
    cv::setIdentity(m, value);
}


CV_IMPL CvScalar cvTrace( const CvArr* arr )
{
    return cv::trace(cv::cvarrToMat(arr));
}


CV_IMPL void cvTranspose( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.cols && src.cols == dst.rows && src.type() == dst.type() );
    transpose( src, dst );
}


CV_IMPL void cvCompleteSymm( CvMat* matrix, int LtoR )
{
    cv::Mat m(matrix);
    cv::completeSymm( m, LtoR != 0 );
}


CV_IMPL void cvCrossProduct( const CvArr* srcAarr, const CvArr* srcBarr, CvArr* dstarr )
{
    cv::Mat srcA = cv::cvarrToMat(srcAarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( srcA.size() == dst.size() && srcA.type() == dst.type() );
    srcA.cross(cv::cvarrToMat(srcBarr)).copyTo(dst);
}


CV_IMPL void
cvReduce( const CvArr* srcarr, CvArr* dstarr, int dim, int op )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    if( dim < 0 )
        dim = src.rows > dst.rows ? 0 : src.cols > dst.cols ? 1 : dst.cols == 1;

    if( dim > 1 )
        CV_Error( CV_StsOutOfRange, "The reduced dimensionality index is out of range" );

    if( (dim == 0 && (dst.cols != src.cols || dst.rows != 1)) ||
        (dim == 1 && (dst.rows != src.rows || dst.cols != 1)) )
        CV_Error( CV_StsBadSize, "The output array size is incorrect" );

    if( src.channels() != dst.channels() )
        CV_Error( CV_StsUnmatchedFormats, "Input and output arrays must have the same number of channels" );

    cv::reduce(src, dst, dim, op, dst.type());
}


CV_IMPL CvArr*
cvRange( CvArr* arr, double start, double end )
{
    int ok = 0;

    CvMat stub, *mat = (CvMat*)arr;
    double delta;
    int type, step;
    double val = start;
    int i, j;
    int rows, cols;

    if( !CV_IS_MAT(mat) )
        mat = cvGetMat( mat, &stub);

    rows = mat->rows;
    cols = mat->cols;
    type = CV_MAT_TYPE(mat->type);
    delta = (end-start)/(rows*cols);

    if( CV_IS_MAT_CONT(mat->type) )
    {
        cols *= rows;
        rows = 1;
        step = 1;
    }
    else
        step = mat->step / CV_ELEM_SIZE(type);

    if( type == CV_32SC1 )
    {
        int* idata = mat->data.i;
        int ival = cvRound(val), idelta = cvRound(delta);

        if( fabs(val - ival) < DBL_EPSILON &&
            fabs(delta - idelta) < DBL_EPSILON )
        {
            for( i = 0; i < rows; i++, idata += step )
                for( j = 0; j < cols; j++, ival += idelta )
                    idata[j] = ival;
        }
        else
        {
            for( i = 0; i < rows; i++, idata += step )
                for( j = 0; j < cols; j++, val += delta )
                    idata[j] = cvRound(val);
        }
    }
    else if( type == CV_32FC1 )
    {
        float* fdata = mat->data.fl;
        for( i = 0; i < rows; i++, fdata += step )
            for( j = 0; j < cols; j++, val += delta )
                fdata[j] = (float)val;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "The function only supports 32sC1 and 32fC1 datatypes" );

    ok = 1;
    return ok ? arr : 0;
}


CV_IMPL void
cvSort( const CvArr* _src, CvArr* _dst, CvArr* _idx, int flags )
{
    cv::Mat src = cv::cvarrToMat(_src);

    if( _idx )
    {
        cv::Mat idx0 = cv::cvarrToMat(_idx), idx = idx0;
        CV_Assert( src.size() == idx.size() && idx.type() == CV_32S && src.data != idx.data );
        cv::sortIdx( src, idx, flags );
        CV_Assert( idx0.data == idx.data );
    }

    if( _dst )
    {
        cv::Mat dst0 = cv::cvarrToMat(_dst), dst = dst0;
        CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
        cv::sort( src, dst, flags );
        CV_Assert( dst0.data == dst.data );
    }
}


CV_IMPL int
cvKMeans2( const CvArr* _samples, int cluster_count, CvArr* _labels,
           CvTermCriteria termcrit, int attempts, CvRNG*,
           int flags, CvArr* _centers, double* _compactness )
{
    cv::Mat data = cv::cvarrToMat(_samples), labels = cv::cvarrToMat(_labels), centers;
    if( _centers )
    {
        centers = cv::cvarrToMat(_centers);

        centers = centers.reshape(1);
        data = data.reshape(1);

        CV_Assert( !centers.empty() );
        CV_Assert( centers.rows == cluster_count );
        CV_Assert( centers.cols == data.cols );
        CV_Assert( centers.depth() == data.depth() );
    }
    CV_Assert( labels.isContinuous() && labels.type() == CV_32S &&
        (labels.cols == 1 || labels.rows == 1) &&
        labels.cols + labels.rows - 1 == data.rows );

    double compactness = cv::kmeans(data, cluster_count, labels, termcrit, attempts,
                                    flags, _centers ? cv::_OutputArray(centers) : cv::_OutputArray() );
    if( _compactness )
        *_compactness = compactness;
    return 1;
}

///////////////////////////// n-dimensional matrices ////////////////////////////

namespace cv
{

Mat Mat::reshape(int _cn, int _newndims, const int* _newsz) const
{
    if(_newndims == dims)
    {
        if(_newsz == 0)
            return reshape(_cn);
        if(_newndims == 2)
            return reshape(_cn, _newsz[0]);
    }

    CV_Error(CV_StsNotImplemented, "");
    // TBD
    return Mat();
}

Mat::operator CvMatND() const
{
    CvMatND mat;
    cvInitMatNDHeader( &mat, dims, size, type(), data );
    int i, d = dims;
    for( i = 0; i < d; i++ )
        mat.dim[i].step = (int)step[i];
    mat.type |= flags & CONTINUOUS_FLAG;
    return mat;
}

NAryMatIterator::NAryMatIterator()
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, Mat* _planes, int _narrays)
: arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, _planes, 0, _narrays);
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, uchar** _ptrs, int _narrays)
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, 0, _ptrs, _narrays);
}

void NAryMatIterator::init(const Mat** _arrays, Mat* _planes, uchar** _ptrs, int _narrays)
{
    CV_Assert( _arrays && (_ptrs || _planes) );
    int i, j, d1=0, i0 = -1, d = -1;

    arrays = _arrays;
    ptrs = _ptrs;
    planes = _planes;
    narrays = _narrays;
    nplanes = 0;
    size = 0;

    if( narrays < 0 )
    {
        for( i = 0; _arrays[i] != 0; i++ )
            ;
        narrays = i;
        CV_Assert(narrays <= 1000);
    }

    iterdepth = 0;

    for( i = 0; i < narrays; i++ )
    {
        CV_Assert(arrays[i] != 0);
        const Mat& A = *arrays[i];
        if( ptrs )
            ptrs[i] = A.data;

        if( !A.data )
            continue;

        if( i0 < 0 )
        {
            i0 = i;
            d = A.dims;

            // find the first dimensionality which is different from 1;
            // in any of the arrays the first "d1" step do not affect the continuity
            for( d1 = 0; d1 < d; d1++ )
                if( A.size[d1] > 1 )
                    break;
        }
        else
            CV_Assert( A.size == arrays[i0]->size );

        if( !A.isContinuous() )
        {
            CV_Assert( A.step[d-1] == A.elemSize() );
            for( j = d-1; j > d1; j-- )
                if( A.step[j]*A.size[j] < A.step[j-1] )
                    break;
            iterdepth = std::max(iterdepth, j);
        }
    }

    if( i0 >= 0 )
    {
        size = arrays[i0]->size[d-1];
        for( j = d-1; j > iterdepth; j-- )
        {
            int64 total1 = (int64)size*arrays[i0]->size[j-1];
            if( total1 != (int)total1 )
                break;
            size = (int)total1;
        }

        iterdepth = j;
        if( iterdepth == d1 )
            iterdepth = 0;

        nplanes = 1;
        for( j = iterdepth-1; j >= 0; j-- )
            nplanes *= arrays[i0]->size[j];
    }
    else
        iterdepth = 0;

    idx = 0;

    if( !planes )
        return;

    for( i = 0; i < narrays; i++ )
    {
        CV_Assert(arrays[i] != 0);
        const Mat& A = *arrays[i];

        if( !A.data )
        {
            planes[i] = Mat();
            continue;
        }

        planes[i] = Mat(1, (int)size, A.type(), A.data);
    }
}


NAryMatIterator& NAryMatIterator::operator ++()
{
    if( idx >= nplanes-1 )
        return *this;
    ++idx;

    if( iterdepth == 1 )
    {
        if( ptrs )
        {
            for( int i = 0; i < narrays; i++ )
            {
                if( !ptrs[i] )
                    continue;
                ptrs[i] = arrays[i]->data + arrays[i]->step[0]*idx;
            }
        }
        if( planes )
        {
            for( int i = 0; i < narrays; i++ )
            {
                if( !planes[i].data )
                    continue;
                planes[i].data = arrays[i]->data + arrays[i]->step[0]*idx;
            }
        }
    }
    else
    {
        for( int i = 0; i < narrays; i++ )
        {
            const Mat& A = *arrays[i];
            if( !A.data )
                continue;
            int _idx = (int)idx;
            uchar* data = A.data;
            for( int j = iterdepth-1; j >= 0 && _idx > 0; j-- )
            {
                int szi = A.size[j], t = _idx/szi;
                data += (_idx - t * szi)*A.step[j];
                _idx = t;
            }
            if( ptrs )
                ptrs[i] = data;
            if( planes )
                planes[i].data = data;
        }
    }

    return *this;
}

NAryMatIterator NAryMatIterator::operator ++(int)
{
    NAryMatIterator it = *this;
    ++*this;
    return it;
}

///////////////////////////////////////////////////////////////////////////
//                              MatConstIterator                         //
///////////////////////////////////////////////////////////////////////////

Point MatConstIterator::pos() const
{
    if( !m )
        return Point();
    CV_DbgAssert(m->dims <= 2);

    ptrdiff_t ofs = ptr - m->data;
    int y = (int)(ofs/m->step[0]);
    return Point((int)((ofs - y*m->step[0])/elemSize), y);
}

void MatConstIterator::pos(int* _idx) const
{
    CV_Assert(m != 0 && _idx);
    ptrdiff_t ofs = ptr - m->data;
    for( int i = 0; i < m->dims; i++ )
    {
        size_t s = m->step[i], v = ofs/s;
        ofs -= v*s;
        _idx[i] = (int)v;
    }
}

ptrdiff_t MatConstIterator::lpos() const
{
    if(!m)
        return 0;
    if( m->isContinuous() )
        return (ptr - sliceStart)/elemSize;
    ptrdiff_t ofs = ptr - m->data;
    int i, d = m->dims;
    if( d == 2 )
    {
        ptrdiff_t y = ofs/m->step[0];
        return y*m->cols + (ofs - y*m->step[0])/elemSize;
    }
    ptrdiff_t result = 0;
    for( i = 0; i < d; i++ )
    {
        size_t s = m->step[i], v = ofs/s;
        ofs -= v*s;
        result = result*m->size[i] + v;
    }
    return result;
}

void MatConstIterator::seek(ptrdiff_t ofs, bool relative)
{
    if( m->isContinuous() )
    {
        ptr = (relative ? ptr : sliceStart) + ofs*elemSize;
        if( ptr < sliceStart )
            ptr = sliceStart;
        else if( ptr > sliceEnd )
            ptr = sliceEnd;
        return;
    }

    int d = m->dims;
    if( d == 2 )
    {
        ptrdiff_t ofs0, y;
        if( relative )
        {
            ofs0 = ptr - m->data;
            y = ofs0/m->step[0];
            ofs += y*m->cols + (ofs0 - y*m->step[0])/elemSize;
        }
        y = ofs/m->cols;
        int y1 = std::min(std::max((int)y, 0), m->rows-1);
        sliceStart = m->data + y1*m->step[0];
        sliceEnd = sliceStart + m->cols*elemSize;
        ptr = y < 0 ? sliceStart : y >= m->rows ? sliceEnd :
            sliceStart + (ofs - y*m->cols)*elemSize;
        return;
    }

    if( relative )
        ofs += lpos();

    if( ofs < 0 )
        ofs = 0;

    int szi = m->size[d-1];
    ptrdiff_t t = ofs/szi;
    int v = (int)(ofs - t*szi);
    ofs = t;
    ptr = m->data + v*elemSize;
    sliceStart = m->data;

    for( int i = d-2; i >= 0; i-- )
    {
        szi = m->size[i];
        t = ofs/szi;
        v = (int)(ofs - t*szi);
        ofs = t;
        sliceStart += v*m->step[i];
    }

    sliceEnd = sliceStart + m->size[d-1]*elemSize;
    if( ofs > 0 )
        ptr = sliceEnd;
    else
        ptr = sliceStart + (ptr - m->data);
}

void MatConstIterator::seek(const int* _idx, bool relative)
{
    int i, d = m->dims;
    ptrdiff_t ofs = 0;
    if( !_idx )
        ;
    else if( d == 2 )
        ofs = _idx[0]*m->size[1] + _idx[1];
    else
    {
        for( i = 0; i < d; i++ )
            ofs = ofs*m->size[i] + _idx[i];
    }
    seek(ofs, relative);
}

ptrdiff_t operator - (const MatConstIterator& b, const MatConstIterator& a)
{
    if( a.m != b.m )
        return INT_MAX;
    if( a.sliceEnd == b.sliceEnd )
        return (b.ptr - a.ptr)/static_cast<ptrdiff_t>(b.elemSize);

    return b.lpos() - a.lpos();
}

//////////////////////////////// SparseMat ////////////////////////////////

template<typename T1, typename T2> void
convertData_(const void* _from, void* _to, int cn)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]);
}

template<typename T1, typename T2> void
convertScaleData_(const void* _from, void* _to, int cn, double alpha, double beta)
{
    const T1* from = (const T1*)_from;
    T2* to = (T2*)_to;
    if( cn == 1 )
        *to = saturate_cast<T2>(*from*alpha + beta);
    else
        for( int i = 0; i < cn; i++ )
            to[i] = saturate_cast<T2>(from[i]*alpha + beta);
}

ConvertData getConvertElem(int fromType, int toType)
{
    static ConvertData tab[][8] =
    {{ convertData_<uchar, uchar>, convertData_<uchar, schar>,
      convertData_<uchar, ushort>, convertData_<uchar, short>,
      convertData_<uchar, int>, convertData_<uchar, float>,
      convertData_<uchar, double>, 0 },

    { convertData_<schar, uchar>, convertData_<schar, schar>,
      convertData_<schar, ushort>, convertData_<schar, short>,
      convertData_<schar, int>, convertData_<schar, float>,
      convertData_<schar, double>, 0 },

    { convertData_<ushort, uchar>, convertData_<ushort, schar>,
      convertData_<ushort, ushort>, convertData_<ushort, short>,
      convertData_<ushort, int>, convertData_<ushort, float>,
      convertData_<ushort, double>, 0 },

    { convertData_<short, uchar>, convertData_<short, schar>,
      convertData_<short, ushort>, convertData_<short, short>,
      convertData_<short, int>, convertData_<short, float>,
      convertData_<short, double>, 0 },

    { convertData_<int, uchar>, convertData_<int, schar>,
      convertData_<int, ushort>, convertData_<int, short>,
      convertData_<int, int>, convertData_<int, float>,
      convertData_<int, double>, 0 },

    { convertData_<float, uchar>, convertData_<float, schar>,
      convertData_<float, ushort>, convertData_<float, short>,
      convertData_<float, int>, convertData_<float, float>,
      convertData_<float, double>, 0 },

    { convertData_<double, uchar>, convertData_<double, schar>,
      convertData_<double, ushort>, convertData_<double, short>,
      convertData_<double, int>, convertData_<double, float>,
      convertData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

ConvertScaleData getConvertScaleElem(int fromType, int toType)
{
    static ConvertScaleData tab[][8] =
    {{ convertScaleData_<uchar, uchar>, convertScaleData_<uchar, schar>,
      convertScaleData_<uchar, ushort>, convertScaleData_<uchar, short>,
      convertScaleData_<uchar, int>, convertScaleData_<uchar, float>,
      convertScaleData_<uchar, double>, 0 },

    { convertScaleData_<schar, uchar>, convertScaleData_<schar, schar>,
      convertScaleData_<schar, ushort>, convertScaleData_<schar, short>,
      convertScaleData_<schar, int>, convertScaleData_<schar, float>,
      convertScaleData_<schar, double>, 0 },

    { convertScaleData_<ushort, uchar>, convertScaleData_<ushort, schar>,
      convertScaleData_<ushort, ushort>, convertScaleData_<ushort, short>,
      convertScaleData_<ushort, int>, convertScaleData_<ushort, float>,
      convertScaleData_<ushort, double>, 0 },

    { convertScaleData_<short, uchar>, convertScaleData_<short, schar>,
      convertScaleData_<short, ushort>, convertScaleData_<short, short>,
      convertScaleData_<short, int>, convertScaleData_<short, float>,
      convertScaleData_<short, double>, 0 },

    { convertScaleData_<int, uchar>, convertScaleData_<int, schar>,
      convertScaleData_<int, ushort>, convertScaleData_<int, short>,
      convertScaleData_<int, int>, convertScaleData_<int, float>,
      convertScaleData_<int, double>, 0 },

    { convertScaleData_<float, uchar>, convertScaleData_<float, schar>,
      convertScaleData_<float, ushort>, convertScaleData_<float, short>,
      convertScaleData_<float, int>, convertScaleData_<float, float>,
      convertScaleData_<float, double>, 0 },

    { convertScaleData_<double, uchar>, convertScaleData_<double, schar>,
      convertScaleData_<double, ushort>, convertScaleData_<double, short>,
      convertScaleData_<double, int>, convertScaleData_<double, float>,
      convertScaleData_<double, double>, 0 },

    { 0, 0, 0, 0, 0, 0, 0, 0 }};

    ConvertScaleData func = tab[CV_MAT_DEPTH(fromType)][CV_MAT_DEPTH(toType)];
    CV_Assert( func != 0 );
    return func;
}

enum { HASH_SIZE0 = 8 };

static inline void copyElem(const uchar* from, uchar* to, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        *(int*)(to + i) = *(const int*)(from + i);
    for( ; i < elemSize; i++ )
        to[i] = from[i];
}

static inline bool isZeroElem(const uchar* data, size_t elemSize)
{
    size_t i;
    for( i = 0; i + sizeof(int) <= elemSize; i += sizeof(int) )
        if( *(int*)(data + i) != 0 )
            return false;
    for( ; i < elemSize; i++ )
        if( data[i] != 0 )
            return false;
    return true;
}

SparseMat::Hdr::Hdr( int _dims, const int* _sizes, int _type )
{
    refcount = 1;

    dims = _dims;
    valueOffset = (int)alignSize(sizeof(SparseMat::Node) +
        sizeof(int)*std::max(dims - CV_MAX_DIM, 0), CV_ELEM_SIZE1(_type));
    nodeSize = alignSize(valueOffset +
        CV_ELEM_SIZE(_type), (int)sizeof(size_t));

    int i;
    for( i = 0; i < dims; i++ )
        size[i] = _sizes[i];
    for( ; i < CV_MAX_DIM; i++ )
        size[i] = 0;
    clear();
}

void SparseMat::Hdr::clear()
{
    hashtab.clear();
    hashtab.resize(HASH_SIZE0);
    pool.clear();
    pool.resize(nodeSize);
    nodeCount = freeList = 0;
}


SparseMat::SparseMat(const Mat& m)
: flags(MAGIC_VAL), hdr(0)
{
    create( m.dims, m.size, m.type() );

    int i, idx[CV_MAX_DIM] = {0}, d = m.dims, lastSize = m.size[d - 1];
    size_t esz = m.elemSize();
    uchar* dptr = m.data;

    for(;;)
    {
        for( i = 0; i < lastSize; i++, dptr += esz )
        {
            if( isZeroElem(dptr, esz) )
                continue;
            idx[d-1] = i;
            uchar* to = newNode(idx, hash(idx));
            copyElem( dptr, to, esz );
        }

        for( i = d - 2; i >= 0; i-- )
        {
            dptr += m.step[i] - m.size[i+1]*m.step[i+1];
            if( ++idx[i] < m.size[i] )
                break;
            idx[i] = 0;
        }
        if( i < 0 )
            break;
    }
}

SparseMat::SparseMat(const CvSparseMat* m)
: flags(MAGIC_VAL), hdr(0)
{
    CV_Assert(m);
    create( m->dims, &m->size[0], m->type );

    CvSparseMatIterator it;
    CvSparseNode* n = cvInitSparseMatIterator(m, &it);
    size_t esz = elemSize();

    for( ; n != 0; n = cvGetNextSparseNode(&it) )
    {
        const int* idx = CV_NODE_IDX(m, n);
        uchar* to = newNode(idx, hash(idx));
        copyElem((const uchar*)CV_NODE_VAL(m, n), to, esz);
    }
}

void SparseMat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert( _sizes && 0 < d && d <= CV_MAX_DIM );
    for( i = 0; i < d; i++ )
        CV_Assert( _sizes[i] > 0 );
    _type = CV_MAT_TYPE(_type);
    if( hdr && _type == type() && hdr->dims == d && hdr->refcount == 1 )
    {
        for( i = 0; i < d; i++ )
            if( _sizes[i] != hdr->size[i] )
                break;
        if( i == d )
        {
            clear();
            return;
        }
    }
    release();
    flags = MAGIC_VAL | _type;
    hdr = new Hdr(d, _sizes, _type);
}

void SparseMat::copyTo( SparseMat& m ) const
{
    if( hdr == m.hdr )
        return;
    if( !hdr )
    {
        m.release();
        return;
    }
    m.create( hdr->dims, hdr->size, type() );
    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = m.newNode(n->idx, n->hashval);
        copyElem( from.ptr, to, esz );
    }
}

void SparseMat::copyTo( Mat& m ) const
{
    CV_Assert( hdr );
    m.create( dims(), hdr->size, type() );
    m = Scalar(0);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        copyElem( from.ptr, m.ptr(n->idx), esz);
    }
}


void SparseMat::convertTo( SparseMat& m, int rtype, double alpha ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);
    if( hdr == m.hdr && rtype != type()  )
    {
        SparseMat temp;
        convertTo(temp, rtype, alpha);
        m = temp;
        return;
    }

    CV_Assert(hdr != 0);
    if( hdr != m.hdr )
        m.create( hdr->dims, hdr->size, rtype );

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount();

    if( alpha == 1 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = hdr == m.hdr ? from.ptr : m.newNode(n->idx, n->hashval);
            cvtfunc( from.ptr, to, cn, alpha, 0 );
        }
    }
}


void SparseMat::convertTo( Mat& m, int rtype, double alpha, double beta ) const
{
    int cn = channels();
    if( rtype < 0 )
        rtype = type();
    rtype = CV_MAKETYPE(rtype, cn);

    CV_Assert( hdr );
    m.create( dims(), hdr->size, rtype );
    m = Scalar(beta);

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount();

    if( alpha == 1 && beta == 0 )
    {
        ConvertData cvtfunc = getConvertElem(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn );
        }
    }
    else
    {
        ConvertScaleData cvtfunc = getConvertScaleElem(type(), rtype);
        for( i = 0; i < N; i++, ++from )
        {
            const Node* n = from.node();
            uchar* to = m.ptr(n->idx);
            cvtfunc( from.ptr, to, cn, alpha, beta );
        }
    }
}

void SparseMat::clear()
{
    if( hdr )
        hdr->clear();
}

SparseMat::operator CvSparseMat*() const
{
    if( !hdr )
        return 0;
    CvSparseMat* m = cvCreateSparseMat(hdr->dims, hdr->size, type());

    SparseMatConstIterator from = begin();
    size_t i, N = nzcount(), esz = elemSize();

    for( i = 0; i < N; i++, ++from )
    {
        const Node* n = from.node();
        uchar* to = cvPtrND(m, n->idx, 0, -2, 0);
        copyElem(from.ptr, to, esz);
    }
    return m;
}

uchar* SparseMat::ptr(int i0, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 1 );
    size_t h = hashval ? *hashval : hash(i0);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0 };
        return newNode( idx, h );
    }
    return 0;
}

uchar* SparseMat::ptr(int i0, int i1, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1 };
        return newNode( idx, h );
    }
    return 0;
}

uchar* SparseMat::ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            return &value<uchar>(elem);
        nidx = elem->next;
    }

    if( createMissing )
    {
        int idx[] = { i0, i1, i2 };
        return newNode( idx, h );
    }
    return 0;
}

uchar* SparseMat::ptr(const int* idx, bool createMissing, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx];
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                return &value<uchar>(elem);
        }
        nidx = elem->next;
    }

    return createMissing ? newNode(idx, h) : 0;
}

void SparseMat::erase(int i0, int i1, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 2 );
    size_t h = hashval ? *hashval : hash(i0, i1);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 && elem->idx[1] == i1 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(int i0, int i1, int i2, size_t* hashval)
{
    CV_Assert( hdr && hdr->dims == 3 );
    size_t h = hashval ? *hashval : hash(i0, i1, i2);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h && elem->idx[0] == i0 &&
            elem->idx[1] == i1 && elem->idx[2] == i2 )
            break;
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::erase(const int* idx, size_t* hashval)
{
    CV_Assert( hdr );
    int i, d = hdr->dims;
    size_t h = hashval ? *hashval : hash(idx);
    size_t hidx = h & (hdr->hashtab.size() - 1), nidx = hdr->hashtab[hidx], previdx=0;
    uchar* pool = &hdr->pool[0];
    while( nidx != 0 )
    {
        Node* elem = (Node*)(pool + nidx);
        if( elem->hashval == h )
        {
            for( i = 0; i < d; i++ )
                if( elem->idx[i] != idx[i] )
                    break;
            if( i == d )
                break;
        }
        previdx = nidx;
        nidx = elem->next;
    }

    if( nidx )
        removeNode(hidx, nidx, previdx);
}

void SparseMat::resizeHashTab(size_t newsize)
{
    newsize = std::max(newsize, (size_t)8);
    if((newsize & (newsize-1)) != 0)
        newsize = (size_t)1 << cvCeil(std::log((double)newsize)/CV_LOG2);

    size_t i, hsize = hdr->hashtab.size();
    vector<size_t> _newh(newsize);
    size_t* newh = &_newh[0];
    for( i = 0; i < newsize; i++ )
        newh[i] = 0;
    uchar* pool = &hdr->pool[0];
    for( i = 0; i < hsize; i++ )
    {
        size_t nidx = hdr->hashtab[i];
        while( nidx )
        {
            Node* elem = (Node*)(pool + nidx);
            size_t next = elem->next;
            size_t newhidx = elem->hashval & (newsize - 1);
            elem->next = newh[newhidx];
            newh[newhidx] = nidx;
            nidx = next;
        }
    }
    hdr->hashtab = _newh;
}

uchar* SparseMat::newNode(const int* idx, size_t hashval)
{
    const int HASH_MAX_FILL_FACTOR=3;
    assert(hdr);
    size_t hsize = hdr->hashtab.size();
    if( ++hdr->nodeCount > hsize*HASH_MAX_FILL_FACTOR )
    {
        resizeHashTab(std::max(hsize*2, (size_t)8));
        hsize = hdr->hashtab.size();
    }

    if( !hdr->freeList )
    {
        size_t i, nsz = hdr->nodeSize, psize = hdr->pool.size(),
            newpsize = std::max(psize*2, 8*nsz);
        hdr->pool.resize(newpsize);
        uchar* pool = &hdr->pool[0];
        hdr->freeList = std::max(psize, nsz);
        for( i = hdr->freeList; i < newpsize - nsz; i += nsz )
            ((Node*)(pool + i))->next = i + nsz;
        ((Node*)(pool + i))->next = 0;
    }
    size_t nidx = hdr->freeList;
    Node* elem = (Node*)&hdr->pool[nidx];
    hdr->freeList = elem->next;
    elem->hashval = hashval;
    size_t hidx = hashval & (hsize - 1);
    elem->next = hdr->hashtab[hidx];
    hdr->hashtab[hidx] = nidx;

    int i, d = hdr->dims;
    for( i = 0; i < d; i++ )
        elem->idx[i] = idx[i];
    size_t esz = elemSize();
    uchar* p = &value<uchar>(elem);
    if( esz == sizeof(float) )
        *((float*)p) = 0.f;
    else if( esz == sizeof(double) )
        *((double*)p) = 0.;
    else
        memset(p, 0, esz);

    return p;
}


void SparseMat::removeNode(size_t hidx, size_t nidx, size_t previdx)
{
    Node* n = node(nidx);
    if( previdx )
    {
        Node* prev = node(previdx);
        prev->next = n->next;
    }
    else
        hdr->hashtab[hidx] = n->next;
    n->next = hdr->freeList;
    hdr->freeList = nidx;
    --hdr->nodeCount;
}


SparseMatConstIterator::SparseMatConstIterator(const SparseMat* _m)
: m((SparseMat*)_m), hashidx(0), ptr(0)
{
    if(!_m || !_m->hdr)
        return;
    SparseMat::Hdr& hdr = *m->hdr;
    const vector<size_t>& htab = hdr.hashtab;
    size_t i, hsize = htab.size();
    for( i = 0; i < hsize; i++ )
    {
        size_t nidx = htab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return;
        }
    }
}

SparseMatConstIterator& SparseMatConstIterator::operator ++()
{
    if( !ptr || !m || !m->hdr )
        return *this;
    SparseMat::Hdr& hdr = *m->hdr;
    size_t next = ((const SparseMat::Node*)(ptr - hdr.valueOffset))->next;
    if( next )
    {
        ptr = &hdr.pool[next] + hdr.valueOffset;
        return *this;
    }
    size_t i = hashidx + 1, sz = hdr.hashtab.size();
    for( ; i < sz; i++ )
    {
        size_t nidx = hdr.hashtab[i];
        if( nidx )
        {
            hashidx = i;
            ptr = &hdr.pool[nidx] + hdr.valueOffset;
            return *this;
        }
    }
    hashidx = sz;
    ptr = 0;
    return *this;
}


double norm( const SparseMat& src, int normType )
{
    SparseMatConstIterator it = src.begin();

    size_t i, N = src.nzcount();
    normType &= NORM_TYPE_MASK;
    int type = src.type();
    double result = 0;

    CV_Assert( normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 );

    if( type == CV_32F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
                result = std::max(result, std::abs((double)*(const float*)it.ptr));
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
                result += std::abs(*(const float*)it.ptr);
        else
            for( i = 0; i < N; i++, ++it )
            {
                double v = *(const float*)it.ptr;
                result += v*v;
            }
    }
    else if( type == CV_64F )
    {
        if( normType == NORM_INF )
            for( i = 0; i < N; i++, ++it )
                result = std::max(result, std::abs(*(const double*)it.ptr));
        else if( normType == NORM_L1 )
            for( i = 0; i < N; i++, ++it )
                result += std::abs(*(const double*)it.ptr);
        else
            for( i = 0; i < N; i++, ++it )
            {
                double v = *(const double*)it.ptr;
                result += v*v;
            }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( normType == NORM_L2 )
        result = std::sqrt(result);
    return result;
}

void minMaxLoc( const SparseMat& src, double* _minval, double* _maxval, int* _minidx, int* _maxidx )
{
    SparseMatConstIterator it = src.begin();
    size_t i, N = src.nzcount(), d = src.hdr ? src.hdr->dims : 0;
    int type = src.type();
    const int *minidx = 0, *maxidx = 0;

    if( type == CV_32F )
    {
        float minval = FLT_MAX, maxval = -FLT_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            float v = *(const float*)it.ptr;
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else if( type == CV_64F )
    {
        double minval = DBL_MAX, maxval = -DBL_MAX;
        for( i = 0; i < N; i++, ++it )
        {
            double v = *(const double*)it.ptr;
            if( v < minval )
            {
                minval = v;
                minidx = it.node()->idx;
            }
            if( v > maxval )
            {
                maxval = v;
                maxidx = it.node()->idx;
            }
        }
        if( _minval )
            *_minval = minval;
        if( _maxval )
            *_maxval = maxval;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 32f and 64f are supported" );

    if( _minidx )
        for( i = 0; i < d; i++ )
            _minidx[i] = minidx[i];
    if( _maxidx )
        for( i = 0; i < d; i++ )
            _maxidx[i] = maxidx[i];
}


void normalize( const SparseMat& src, SparseMat& dst, double a, int norm_type )
{
    double scale = 1;
    if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );

    src.convertTo( dst, -1, scale );
}

////////////////////// RotatedRect //////////////////////

void RotatedRect::points(Point2f pt[]) const
{
    double _angle = angle*CV_PI/180.;
    float b = (float)cos(_angle)*0.5f;
    float a = (float)sin(_angle)*0.5f;

    pt[0].x = center.x - a*size.height - b*size.width;
    pt[0].y = center.y + b*size.height - a*size.width;
    pt[1].x = center.x + a*size.height - b*size.width;
    pt[1].y = center.y - b*size.height - a*size.width;
    pt[2].x = 2*center.x - pt[0].x;
    pt[2].y = 2*center.y - pt[0].y;
    pt[3].x = 2*center.x - pt[1].x;
    pt[3].y = 2*center.y - pt[1].y;
}

Rect RotatedRect::boundingRect() const
{
    Point2f pt[4];
    points(pt);
    Rect r(cvFloor(min(min(min(pt[0].x, pt[1].x), pt[2].x), pt[3].x)),
           cvFloor(min(min(min(pt[0].y, pt[1].y), pt[2].y), pt[3].y)),
           cvCeil(max(max(max(pt[0].x, pt[1].x), pt[2].x), pt[3].x)),
           cvCeil(max(max(max(pt[0].y, pt[1].y), pt[2].y), pt[3].y)));
    r.width -= r.x - 1;
    r.height -= r.y - 1;
    return r;
}

}

/* End of file. */
