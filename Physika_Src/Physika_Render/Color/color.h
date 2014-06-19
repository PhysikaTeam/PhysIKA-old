/*
 * @file color.h 
 * @Brief the color for rendering.
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

#ifndef PHYSIKA_RENDER_COLOR_COLOR_H_
#define PHYSIKA_RENDER_COLOR_COLOR_H_

#include <iostream>
#include "Physika_Core/Utilities/type_utilities.h"

namespace Physika{

/*
 * Scalar can be types accepted by glColor, namely: byte, short, int, float, double,
 * unsigned byte, unsigned short, and unsigned int
 * 
 * the value range of each channel is that of the type
 */

template <typename Scalar>
class Color
{
public:
    Color(); //black
    Color(Scalar red, Scalar green, Scalar blue);
    Color(Scalar red, Scalar green, Scalar blue, Scalar aplha);
    Color(const Color &color);
    Color& operator= (const Color &color);
    bool operator== (const Color &color);
    bool operator!= (const Color &color);
    Scalar redChannel() const;
    Scalar greenChannel() const;
    Scalar blueChannel() const;
    Scalar alphaChannel() const;
    void setRedChannel(Scalar);
    void setGreenChannel(Scalar);
    void setBlueChannel(Scalar);
    void setAlphaChannel(Scalar);
protected:
    Scalar rgba_[4]; 
};

//override << for Color
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Color<Scalar> &color)
{
    Scalar r = color.redChannel(), g = color.greenChannel(), b = color.blueChannel(), alpha = color.alphaChannel();
    //char types are casted to integers before output
    if(is_same<Scalar,char>::value) //byte
        s<<"("<<static_cast<int>(r)<<","<<static_cast<int>(g)<<","<<static_cast<int>(b)<<","<<static_cast<int>(alpha)<<")";
    else if(is_same<Scalar,unsigned char>::value) //unsigned byte
        s<<"("<<static_cast<unsigned int>(r)<<","<<static_cast<unsigned int>(g)<<","<<static_cast<unsigned int>(b)<<","<<static_cast<unsigned int>(alpha)<<")";
    else
        s<<"("<<r<<","<<g<<","<<b<<","<<alpha<<")";
    return s;
}

}  //endof namespace Physika

#endif //PHYSIKA_RENDER_COLOR_COLOR_H_
