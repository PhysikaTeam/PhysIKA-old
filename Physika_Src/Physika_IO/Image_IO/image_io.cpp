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

namespace Physika{

unsigned char* ImageIO::load(const string &filename, int &width, int &height)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
	std::cerr<<"No file extension found for the image file!\n";
	return NULL;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
	return PngIO::load(filename,width,height);
    else
    {
	std::cerr<<"Unknown image file format!\n";
	return NULL;
    }
}

bool ImageIO::save(const string &filename, int width, int height, const unsigned char *image_data)
{
    string::size_type suffix_idx = filename.rfind('.');
    if(suffix_idx>=filename.size())
    {
	std::cerr<<"No file extension specified!\n";
	return false;
    }
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
	return PngIO::save(filename,width,height,image_data);
    else
    {
	std::cerr<<"Unknown image file format specified!\n";
	return false;
    }
}

} //end of namespace Physika


















