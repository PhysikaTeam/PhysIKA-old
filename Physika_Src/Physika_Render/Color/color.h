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

#include <limits>
#include <iostream>
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Range/interval.h"

namespace Physika{

/*
 * Scalar are the types accepted by glColor, namely: byte, short, int, float, double,
 * unsigned byte, unsigned short, and unsigned int
 * 
 * Note on the range of color channels (from opengl specification):
 * The value range of each channel is that of the type. When color is passed to opengl,
 * the float-point format color will be clamped to [0,1], unsigned integers will be linearly
 * mapped to [0,1], and signed integers will be linearly mapped to [-1,1].
 */

template <typename Scalar>
class Color
{
public:
    Color(); //black
    Color(Scalar red, Scalar green, Scalar blue);
    Color(Scalar red, Scalar green, Scalar blue, Scalar aplha);
    Color(const Color<Scalar> &color);
    Color& operator= (const Color<Scalar> &color);
    bool operator== (const Color<Scalar> &color);
    bool operator!= (const Color<Scalar> &color);
    Scalar redChannel() const;
    Scalar greenChannel() const;
    Scalar blueChannel() const;
    Scalar alphaChannel() const;
    void setRedChannel(Scalar);
    void setGreenChannel(Scalar);
    void setBlueChannel(Scalar);
    void setAlphaChannel(Scalar);

    //predefined colors
    //naming of these methods break the coding style in order to highlight their specialness
    static Color<Scalar> Red();
    static Color<Scalar> Green();
    static Color<Scalar> Blue();
    static Color<Scalar> White();
    static Color<Scalar> Black();
    static Color<Scalar> Gray();
    static Color<Scalar> Yellow();
    static Color<Scalar> Purple();
    static Color<Scalar> Cyan();

    //method to convert between different color types
    template <typename TargetType>
    static Color<TargetType> convertColor(const Color<Scalar> &color);

protected:
    Scalar rgba_[4]; 
};

//override << for Color
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Color<Scalar> &color)
{
    Scalar r = color.redChannel(), g = color.greenChannel(), b = color.blueChannel(), alpha = color.alphaChannel();
    //char types are casted to integers before output
    if(is_same<Scalar,signed char>::value) //byte
        s<<"("<<static_cast<int>(r)<<","<<static_cast<int>(g)<<","<<static_cast<int>(b)<<","<<static_cast<int>(alpha)<<")";
    else if(is_same<Scalar,unsigned char>::value) //unsigned byte
        s<<"("<<static_cast<unsigned int>(r)<<","<<static_cast<unsigned int>(g)<<","<<static_cast<unsigned int>(b)<<","<<static_cast<unsigned int>(alpha)<<")";
    else
        s<<"("<<r<<","<<g<<","<<b<<","<<alpha<<")";
    return s;
}

//implementation of color convertion
template <typename Scalar>
template <typename TargetType>
Color<TargetType> Color<Scalar>::convertColor(const Color<Scalar> &color)
{
    Interval<TargetType> target_type_range(std::numeric_limits<TargetType>::min(),std::numeric_limits<TargetType>::max());
    Interval<Scalar> src_type_range(std::numeric_limits<Scalar>::min(),std::numeric_limits<Scalar>::max());
    //TO DO
}

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_COLOR_COLOR_H_
