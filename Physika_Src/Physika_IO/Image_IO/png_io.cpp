/*
 * @file png_io.cpp 
 * @Brief load/save png file
 * @author Wei Chen, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <vector>
#include <iostream>
#include "Physika_IO/Image_IO/png_io.h"
#include "Physika_Dependency/LodePNG/lodepng.h"
#include "Physika_Core/Utilities/physika_assert.h"

using std::string;

namespace Physika{


bool PngIO::load(const string &filename,Image *image )
{
    return PngIO::load(filename, image, Image::RGBA);
}

bool PngIO::load(const std::string &filename, Image * image, Image::DataFormat data_format)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension found for the image file!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix!=string(".png"))                                     //if the filename is not ended with ".png"
    {
        std::cerr<<"Unknown image file format!\n";
        return false;
    }

    unsigned int width, height;
    unsigned int error;
    std::vector<unsigned char> image_vec;

    if(data_format == Image::RGBA) //RGBA format
    {
        error = lodepng::decode(image_vec, width, height, filename);   //decode png file to image
    }
    else //RGB format
    {
        error = lodepng::decode(image_vec, width, height, filename, LCT_RGB);
    }
    string error_message = "decoder error "+error+string(": ")+lodepng_error_text(error);
    if(error!=0)
    {
        std::cerr<<error_message<<std::endl;
        return false;
    }
    
    unsigned char * image_data;
    if(data_format == Image::RGBA) //RGBA format
    {
        image_data= new unsigned char[width*height*4];  //allocate memory
    }
    else
    {
        image_data= new unsigned char[width*height*3];
    }
    if(image_data == NULL)
    {
        std::cerr<<"error: can't allocate memory!"<<std::endl;
        return false;
    }
    if(data_format == Image::RGBA)   //RGBA format
    {
        for(unsigned int i=0; i<image_vec.size(); i=i+4) //loop for perPixel
        {
            image_data[i] = image_vec[i];        // red   color
            image_data[i+1] = image_vec[i+1];    // green color
            image_data[i+2] = image_vec[i+2];    // blue  color
            image_data[i+3] = image_vec[i+3];    // alpha value
        }
    }
    else
    {
        for(unsigned int i=0; i<image_vec.size(); i=i+3) //loop for perPixel
        {
            image_data[i] = image_vec[i];        // red   color
            image_data[i+1] = image_vec[i+1];    // green color
            image_data[i+2] = image_vec[i+2];    // blue  color
        }
    }
    image->setRawData(width, height,data_format, image_data);
    delete [] image_data;
    return true;
}

bool PngIO::save(const string &filename, const Image *image)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension specified!\n";
        return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix!=string(".png"))                                     //if the filename is not ended with ".png"
    {
        std::cerr<<"Wrong file extension specified for PNG file!\n";
        return false;
    }
    unsigned int error;
    if(image->dataFormat() == Image::RGBA)
    {
        error = lodepng::encode(filename, image->rawData(), image->width(), image->height());   //encode the image_data to file
    }
    else
    {
        error = lodepng::encode(filename, image->rawData(), image->width(), image->height(), LCT_RGB);
    }
    string error_message = "decoder error "+error+string(": ")+lodepng_error_text(error);   //define the error message 
    if(error!=0)
    {
        std::cerr<<error_message<<std::endl;
        return false;
    }                                     
    return true;
}

} //end of namespace Physika


















