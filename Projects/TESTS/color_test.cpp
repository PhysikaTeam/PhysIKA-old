/*
 * @file color_test.cpp 
 * @brief Test the Color class of Physika.
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

#include <iostream>
#include "Physika_Core/Utilities/opengl_headers.h"
#include "Physika_Render/Color/color.h"
using namespace std;
using Physika::Color;

int main()
{
    Color<GLfloat> color_float;
    cout<<"Test color float:\n";
    cout<<color_float.redChannel()<<" "<<color_float.greenChannel()<<" "<<color_float.blueChannel()<<" "<<color_float.alphaChannel()<<"\n";
    typedef unsigned char byte;
    Color<byte> color_byte(255,255,255);
    cout<<"Test color byte:\n";
    cout<<static_cast<unsigned int>(color_byte.redChannel())<<" "<<static_cast<unsigned int>(color_byte.greenChannel())<<" ";
    cout<<static_cast<unsigned int>(color_byte.blueChannel())<<" "<<static_cast<unsigned int>(color_byte.alphaChannel())<<"\n";
    cout<<"Test << operator:\n";
    cout<<color_byte<<"\n";
    return 0;
}
