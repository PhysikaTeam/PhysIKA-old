/*
 * @file image.cpp 
 * @brief Image class, support basic operations on image data.
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstring>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Range/range.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Image/image.h"

namespace Physika{

Image::Image()
    :width_(0),height_(0),data_format_(RGBA),raw_data_(NULL)
{
}

Image::Image(unsigned int width, unsigned int height, Image::DataFormat data_format, const unsigned char *raw_data)
    :width_(width),height_(height),data_format_(data_format),raw_data_(NULL)
{
    if(raw_data==NULL)
        throw PhysikaException("Pointer to image data is NULL.");
    allocMemory();
    unsigned int data_size = sizeof(unsigned char)*pixelSize()*width_*height_;
    memcpy(raw_data_,raw_data,data_size);
}

Image::Image(const Image &image)
{
    Image(image.width_,image.height_,image.data_format_,image.raw_data_);
}

Image& Image::operator=(const Image &image)
{
    setRawData(image.width_,image.height_,image.data_format_,image.raw_data_);
    return *this;
}

Image::~Image()
{
    if(raw_data_)
        delete[] raw_data_;
}

unsigned int Image::width() const
{
    return width_;
}

unsigned int Image::height() const
{
    return height_;
}

Image::DataFormat Image::dataFormat() const
{
    return data_format_;
}

const unsigned char* Image::rawData() const
{
    return raw_data_;
}

unsigned char* Image::rawData()
{
    return raw_data_;
}

void Image::setRawData(unsigned int width, unsigned int height, DataFormat data_format, const unsigned char *raw_data)
{
    if(raw_data==NULL)
        throw PhysikaException("Pointer to image data is NULL.");
    width_ = width;
    height_ = height;
    data_format_ = data_format;
    allocMemory();
    unsigned int data_size = sizeof(unsigned char)*pixelSize()*width_*height_;
    memcpy(raw_data_,raw_data,data_size);
}

void Image::flipHorizontally()
{
    unsigned int pixel_size = pixelSize();
    for(unsigned int i = 0; i < width_/2; ++i)
        for(unsigned int j = 0; j < height_; ++j)
            for(unsigned int k = 0; k < pixel_size; ++k)
            {
                unsigned char temp = raw_data_[(i+j*width_)*pixel_size+k];
                raw_data_[(i+j*width_)*pixel_size+k] = raw_data_[((width_-1-i)+j*width_)*pixel_size+k];
                raw_data_[((width_-1-i)+j*width_)*pixel_size+k] = temp;
            }
}

void Image::flipVertically()
{
    unsigned int pixel_size = pixelSize();
    for(unsigned int j = 0; j < height_/2; ++j)
        for(unsigned int i = 0; i < pixel_size*width_; ++i)
        {
            unsigned char temp = raw_data_[i+j*width_*pixel_size];
            raw_data_[i+j*width_*pixel_size] = raw_data_[i+ (height_-1-j)*width_*pixel_size];
            raw_data_[i+ (height_-1-j)*width_*pixel_size] = temp;
        } 
}

Image Image::mirrorImage() const
{
    unsigned int pixel_size = pixelSize();
    unsigned char *data = new unsigned char[pixel_size*width_*height_];
    PHYSIKA_ASSERT(data);
    memcpy(data,raw_data_,sizeof(unsigned char)*pixel_size*width_*height_);
    Image image(width_,height_,data_format_,data);
    image.flipHorizontally();
    delete[] data;
    return image;
}

Image Image::upsideDownImage() const
{
    unsigned int pixel_size = pixelSize();
    unsigned char *data = new unsigned char[pixel_size*width_*height_];
    PHYSIKA_ASSERT(data);
    memcpy(data,raw_data_,sizeof(unsigned char)*pixel_size*width_*height_);
    Image image(width_,height_,data_format_,data);
    image.flipVertically();
    delete[] data;
    return image;
}

//return sub image within given range (left up corner is the origin)
Image Image::subImage(const Range<unsigned int,2> &range) const
{
    Vector<unsigned int,2> size = range.edgeLengths();
    unsigned int pixel_size = pixelSize();
    unsigned char *data = new unsigned char[pixel_size*size[0]*size[1]];
    PHYSIKA_ASSERT(data);
    Vector<unsigned int,2> start = range.minCorner();
    for(unsigned int i = 0; i < size[0]*pixel_size; ++i)
        for(unsigned int j = 0; j < size[1]; ++j)
        {
            unsigned int target_index = i+j*size[0]*pixel_size;
            unsigned int source_index = start[0]+i+(j+start[1])*width_*pixel_size;
            data[target_index] = raw_data_[source_index];
        }
    Image image(size[0],size[1],data_format_,data);
    delete[] data;
    return image;
}

void Image::allocMemory()
{
    if(raw_data_)
        delete[] raw_data_;
    unsigned int data_size = pixelSize()*width_*height_;
    raw_data_ = new unsigned char[data_size];
    PHYSIKA_ASSERT(raw_data_);
}

unsigned int Image::pixelSize() const
{
    unsigned int pixel_size = 0;
    if(data_format_ == RGBA)
        pixel_size = 4;
    else if(data_format_ == RGB)
        pixel_size = 3;
    else
        PHYSIKA_ERROR("Invalid pixel data format.");
    return pixel_size;
}

} //end of namespace Physika
