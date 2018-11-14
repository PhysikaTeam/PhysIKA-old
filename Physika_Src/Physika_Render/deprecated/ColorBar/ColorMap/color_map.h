/*
 * @file color_map.h 
 * @Brief a color map class for OpenGL.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_COLORBAR_COLORMAP_COLOR_MAP_H_
#define PHYSIKA_RENDER_COLORBAR_COLORMAP_COLOR_MAP_H_

#include <vector>
#include <iostream>
#include "Physika_Render/Color/color.h"

namespace Physika{

// Type of ColorMap
enum ColorMapType
{
    Gray,
    Red,
    Green,
    Blue,
    Jet,
    Spring,
    Summer,
    Autumn,
    Winter,
    Hot,
    Cool
};

template <typename Scalar>
class ColorMap
{
public:
    ColorMap();  //default size: 64, default Type: Jet
    ColorMap(ColorMapType color_map_type, unsigned int color_size);
    ~ColorMap();

    //getter
    ColorMapType colorMapType() const;
    unsigned int colorMapSize() const;
    const std::vector<Color<Scalar> > & colorMapVec() const;

    // get color via ratio ranging from 0.0 to 1.0, ratio of small than 0.0 or greater than 1.0 is clamped.  
    const Color<Scalar> & colorViaRatio(Scalar ratio) const;
    // get color via index, index greater than size of colormap is clamped.
    const Color<Scalar> & colorViaIndex(unsigned int index) const;

    //setter
    void setColorMapType(ColorMapType color_map_type);
    void setColorMapSize(unsigned int color_size);
    void setColorMapTypeAndSize(ColorMapType color_map_type, unsigned int color_size);

protected:
    void grayColorMap();
    void redColorMap();
    void greenColorMap();
    void blueColorMap();
    void jetColorMap();
    void springColorMap();
    void summerColorMap();
    void autumnColorMap();
    void winterColorMap();
    void hotColorMap();
    void coolColorMap();

protected:
    std::vector<Color<Scalar> > color_map_vec_;
    ColorMapType color_map_type_;
};

template <typename Scalar>
std::ostream & operator << (std::ostream & out, const ColorMap<Scalar> & color_map)
{
    out<<"color map: \n";
    out<<"color map type: ";
    switch(color_map.colorMapType())
    {
    case ColorMapType::Gray:   {out<<"Gray \n"; break;}
    case ColorMapType::Red:    {out<<"Red \n"; break;}
    case ColorMapType::Green:  {out<<"Green \n"; break;}
    case ColorMapType::Blue:   {out<<"Blue \n"; break;}
    case ColorMapType::Jet:    {out<<"Jet \n"; break;}
    case ColorMapType::Spring: {out<<"Spring \n"; break;}
    case ColorMapType::Summer: {out<<"Summer \n"; break;}
    case ColorMapType::Autumn: {out<<"Autumn \n"; break;}
    case ColorMapType::Winter: {out<<"Winter \n"; break;}
    case ColorMapType::Hot:    {out<<"Hot \n"; break;}
    case ColorMapType::Cool:   {out<<"Cool \n"; break;}
    }
    out<<"color map size: "<<color_map.colorMapSize()<<std::endl;
    for (unsigned int i=0; i<color_map.colorMapSize(); i++)
    {
        out<<color_map.colorViaIndex(i)<<std::endl;
    }
    return out;
}

}  //end of namespace Physika

#endif  //PHYSIKA_RENDER_COLORBAR_COLORMAP_COLOR_MAP_H