/*
 * @file color.h 
 * @Brief the color for rendering.
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
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
 * The value range of each channel is the range of the type. When color is passed to opengl,
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
    //usage:
    //1. If the template parameter of src_color is unknown, must add "template" 
    //   keyword before the method name:
    //   Color<TargetType> color = src_color.template convertColor<TargetType>();
    //2. If the template parameter of src_color is known, "template" keyword is optional:
    //   Color<TargetType> color = src_color.convertColor<TargetType>();
    //3. Conclusion: add "template" keyword is always right
    template <typename TargetType>
    Color<TargetType> convertColor() const;

protected:
    Scalar rgba_[4]; 
};

//override << for Color
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Color<Scalar> &color)
{
    Scalar r = color.redChannel(), g = color.greenChannel(), b = color.blueChannel(), alpha = color.alphaChannel();
    //char types are casted to integers before output
    if((is_same<Scalar,signed char>::value)||(is_same<Scalar,unsigned char>::value)) //byte or unsigned byte
        s<<"("<<static_cast<int>(r)<<","<<static_cast<int>(g)<<","<<static_cast<int>(b)<<","<<static_cast<int>(alpha)<<")";
    else
        s<<"("<<r<<","<<g<<","<<b<<","<<alpha<<")";
    return s;
}

//implementation of color conversion
template <typename Scalar>
template <typename TargetType>
Color<TargetType> Color<Scalar>::convertColor() const
{
    Interval<TargetType> target_type_range((std::numeric_limits<TargetType>::min)(),(std::numeric_limits<TargetType>::max)());
    Interval<Scalar> src_type_range((std::numeric_limits<Scalar>::min)(),(std::numeric_limits<Scalar>::max)());

    Scalar src_red = this->redChannel();
    Scalar src_green = this->greenChannel();
    Scalar src_blue = this->blueChannel();
    Scalar src_alpha = this->alphaChannel();

    if (is_same<TargetType, Scalar>::value || (is_floating_point<Scalar>::value && is_floating_point<TargetType>::value))  //target type and source type are the same, or both are float type
    {
        return Color<TargetType>(static_cast<TargetType>(src_red),
                                 static_cast<TargetType>(src_green),
                                 static_cast<TargetType>(src_blue),
                                 static_cast<TargetType>(src_alpha));
    }
    else if(is_floating_point<Scalar>::value)  //from float type to integer type, first clamp it into [0,1] before conversion
    {
        src_red = src_red > 1 ? 1 : src_red;
        src_red = src_red < 0 ? 0 : src_red;
        src_green = src_green > 1 ? 1 : src_green;
        src_green = src_green < 0 ? 0 : src_green;
        src_blue = src_blue > 1 ? 1 : src_blue;
        src_blue = src_blue < 0 ? 0 : src_blue;
        src_alpha = src_alpha > 1 ? 1 : src_alpha;
        src_alpha = src_alpha < 0 ? 0 : src_alpha;

        TargetType target_red = static_cast<TargetType>(src_red * std::numeric_limits<Scalar>::max());
        TargetType target_green = static_cast<TargetType>(src_green * std::numeric_limits<Scalar>::max());
        TargetType target_blue = static_cast<TargetType>(src_blue * std::numeric_limits<Scalar>::max());
        TargetType target_alpha = static_cast<TargetType>(src_alpha * std::numeric_limits<Scalar>::max());

        return Color<TargetType>(target_red, target_green, target_blue, target_alpha);
    }
    else if(is_integer<Scalar>::value && is_floating_point<TargetType>::value) //from integer type to floating type
    {
        TargetType target_red = static_cast<TargetType>(src_red) / std::numeric_limits<Scalar>::max();
        TargetType target_green = static_cast<TargetType>(src_green) / std::numeric_limits<Scalar>::max();
        TargetType target_blue = static_cast<TargetType>(src_blue) / std::numeric_limits<Scalar>::max();
        TargetType target_alpha = static_cast<TargetType>(src_alpha) / std::numeric_limits<Scalar>::max();

        return Color<TargetType>(target_red, target_green, target_blue, target_alpha);
    }
    else //from integer type to integer type
    {
        TargetType target_red = static_cast<float>(src_red) / std::numeric_limits<Scalar>::max() * std::numeric_limits<TargetType>::max();
        TargetType target_green = static_cast<float>(src_green) / std::numeric_limits<Scalar>::max() * std::numeric_limits<TargetType>::max();
        TargetType target_blue = static_cast<float>(src_blue) / std::numeric_limits<Scalar>::max() * std::numeric_limits<TargetType>::max();
        TargetType target_alpha = static_cast<float>(src_alpha) / std::numeric_limits<Scalar>::max() * std::numeric_limits<TargetType>::max();

        return Color<TargetType>(target_red, target_green, target_blue, target_alpha);
    }
}

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_COLOR_COLOR_H_
