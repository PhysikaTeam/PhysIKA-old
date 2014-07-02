/*
 * @file glut_window_test.cpp 
 * @brief Test Image class of Physika.
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

#include <string>
#include <iostream>
#include "Physika_Core/Image/image.h"
#include "Physika_Core/Range/range.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_IO/Image_IO/image_io.h"
using namespace std;
using Physika::Image;
using Physika::ImageIO;
using Physika::Range;
using Physika::Vector;

int main()
{
    string file_name("image_test_input.png");
    //string file_name("screen_capture_0.png");
    Image image;
    bool status = ImageIO::load(file_name,&image);
    if(status)
    {
        cout<<"Width: "<<image.width()<<", Height: "<<image.height()<<"\n";
        cout<<"Save horizontally flipped image to flip_horizon.png:\n";
        Image flip_horizon = image.mirrorImage();
        ImageIO::save(string("flip_horizon.png"),&flip_horizon);
        cout<<"Done.\n";
        // cout<<"Save vertically flipped image to flip_vertical.png:\n";
        // Image flip_vertical = image.upsideDownImage();
        // ImageIO::save(string("flip_vertical.png"),&flip_vertical);
        // cout<<"Done.\n";
        // cout<<"Save left lower quarter to left_lower_quarter.png:\n";
        // Range<unsigned int,2> range(Vector<unsigned int,2>(0,image.height()/2),Vector<unsigned int,2>(image.width()/2,image.height()));
        // Image sub_image = image.subImage(range);
        // ImageIO::save(string("left_lower_quarter.png"),&sub_image);
        // cout<<"Done.\n";
    }
    return 0;
}







