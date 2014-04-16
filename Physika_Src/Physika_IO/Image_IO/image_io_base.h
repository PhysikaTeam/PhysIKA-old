/*
 * @file image_io_base.h 
 * @brief, base class to import image files such as bmp etc.
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

#ifndef PHYSIKA_IO_IMAGE_IO_IMAGE_IO_BASE_H_
#define PHYSIKA_IO_IMAGE_IO_IMAGE_IO_BASE_H_

namespace Physika{

class ImageIOBase
{
public:
    ImageIOBase();
    ~ImageIOBase();
protected:
};

} //end of namespace Physika

#endif //PHYSIKA_IO_IMAGE_IO_IMAGE_IO_BASE_H_
