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

/// warning: this function only read color data(IDAT chunk) from png file
///	(i.e. it ignores all other data chunks ,such as ancillary chunks)
/// since only IDAT chunk makes sense for our texture.Thus if you load from a png file and
///  resave image data to another png file,the file size will be smaller than the origin one.
unsigned char* PngIO::load(const string &filename, int &width,int &height)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
        std::cerr<<"No file extension found for the image file!\n";
        return NULL;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix!=string(".png"))                                     //if the filename is not ended with ".png"
    {
        std::cerr<<"Unknown image file format!\n";
        return NULL;
    }

    std::vector<unsigned char> image;
    unsigned int error = lodepng::decode(image, (unsigned int &)width,(unsigned int &) height, filename);   //decode png file to image
    string error_message = "decoder error "+error+string(": ")+lodepng_error_text(error);
    PHYSIKA_MESSAGE_ASSERT(error==0, error_message);
    
    unsigned char * image_data= new unsigned char[width*height*4];  //allocate memory
    PHYSIKA_ASSERT(image_data);
    for(long i=0; i<image.size(); i=i+4) //loop for perPixel
    {
        image_data[i] = image[i];        // red   color
        image_data[i+1] = image[i+1];    // green color
        image_data[i+2] = image[i+2];    // blue  color
        image_data[i+3] = image[i+3];    // alpha value
    }
    return image_data;
}

bool PngIO::save(const string &filename, int width, int height, const unsigned char *image_data)
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
    unsigned error = lodepng::encode(filename, image_data, width, height);   //encode the image_data to file
    string error_message = "decoder error "+error+string(": ")+lodepng_error_text(error);   //difine the error message 
    PHYSIKA_MESSAGE_ASSERT(error==0, error_message);                                       // if an error happends, output the error message
    return true;
}

} //end of namespace Physika


















