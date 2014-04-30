/*
 * @file png_io.h 
 * @Brief load/save png file
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_IO_IMAGE_IO_PNG_IO_H_
#define PHYSIKA_IO_IMAGE_IO_PNG_IO_H_

#include <string>
using std::string;

namespace Physika{

class PngIO
{
public:
    PngIO(){}
    ~PngIO(){}
    static unsigned char* load(const string &filename, int &width, int &height);
    static void save(const string &filename, int width, int height, const unsigned char *image_data);
protected:

};

} //end of namespace Physika

#endif //PHYSIKA_IO_IMAGE_IO_PNG_IO_H_










