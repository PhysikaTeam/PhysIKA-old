/*
 * @file image_io.cpp 
 * @Brief image_io class, it is used to import/save image files such as bmp etc.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_IO/Image_IO/png_io.h"
#include "ppm_io.h"
using std::string;

namespace Physika{

bool ImageIO::load(const string &filename, Image *image)
{
    return load(filename,image,Image::RGBA);
}

bool ImageIO::load(const string &filename, Image * image, Image::DataFormat data_format)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension found for the image file!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
        return PngIO::load(filename, image, data_format);
    else  if(suffix==string(".ppm"))
        return PPMIO::load(filename, image, data_format);
    else
    {
        std::cerr<<"Unknown image file format!\n";
        return false;
    }
}

bool ImageIO::save(const string &filename, const Image * image)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx >= filename.size())
    {
        std::cerr<<"No file extension specified!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
    {
            return PngIO::save(filename, image);
    }
    else if(suffix==string(".ppm"))
    {
            return PPMIO::save(filename, image);
    }
    else
    {
        std::cerr<<"Unknown image file format specified!\n";
        return false;
    }
}

} //end of namespace Physika
