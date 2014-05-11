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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_IO/Image_IO/png_io.h"

namespace Physika{

unsigned char* ImageIO::load(const string &filename, int &width, int &height)
{
    string::size_type suffix_idx = filename.rfind('.');
    PHYSIKA_ASSERT(suffix_idx<filename.size());
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
	return PngIO::load(filename,width,height);
    else
	PHYSIKA_ERROR("Unknown image file format!");
}

void ImageIO::save(const string &filename, int width, int height, const unsigned char *image_data)
{
    string::size_type suffix_idx = filename.rfind('.');
    PHYSIKA_ASSERT(suffix_idx<filename.size());
    string suffix = filename.substr(suffix_idx);
    if(suffix==string(".png"))
	PngIO::save(filename,width,height,image_data);
    else
	PHYSIKA_ERROR("Unknown image file format specified!");
}

} //end of namespace Physika


















