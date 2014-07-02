/*
 * @file image.cpp 
 * @brief Image class, support basic operations on image data.
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstring>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Range/range.h"
#include "Physika_Core/Image/image.h"

namespace Physika{

Image::Image()
    :width_(0),height_(0),raw_data_(NULL),data_format_(RGBA)
{
}

Image::Image(unsigned int width, unsigned int height, Image::DataFormat data_format, const unsigned char *raw_data)
    :width_(width),height_(height),data_format_(data_format)
{
    if(raw_data==NULL)
    {
        std::cerr<<"Pointer to image data is NULL.\n";
        std::exit(EXIT_FAILURE);
    }
    allocMemory();
    unsigned int data_size = 0;
    if(data_format_== RGBA)
        data_size = 4*width_*height_;
    else if(data_format_ == RGB)
        data_size = 3*width_*height_;
    else
        PHYSIKA_ERROR("Invalid pixel data format.");
    memcpy(raw_data_,raw_data,sizeof(unsigned char)*data_size);
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
    {
        std::cerr<<"Pointer to image data is NULL.\n";
        std::exit(EXIT_FAILURE);
    }
    width_ = width;
    height_ = height;
    data_format_ = data_format;
    allocMemory();
    unsigned int data_size = 0;
    if(data_format_== RGBA)
        data_size = 4*width_*height_;
    else if(data_format_ == RGB)
        data_size = 3*width_*height_;
    else
        PHYSIKA_ERROR("Invalid pixel data format.");
    memcpy(raw_data_,raw_data,sizeof(unsigned char)*data_size);
}

void Image::flipHorizontally()
{
//TO DO
}

void Image::flipVertically()
{
//TO DO
}

Image Image::mirrorImage() const
{
//TO DO
    return Image();
}

Image Image::upsideDownImage() const
{
//TO DO
    return Image();
}

Image Image::subImage(const Range<unsigned int,2> &range) const
{
//TO DO
    return Image();
}

void Image::allocMemory()
{
    if(raw_data_)
        delete[] raw_data_;
    unsigned int data_size = 0;
    if(data_format_== RGBA)
        data_size = 4*width_*height_;
    else if(data_format_ == RGB)
        data_size = 3*width_*height_;
    else
        PHYSIKA_ERROR("Invalid pixel data format.");
    raw_data_ = new unsigned char[data_size];
    PHYSIKA_ASSERT(raw_data_);
}

} //end of namespace Physika










