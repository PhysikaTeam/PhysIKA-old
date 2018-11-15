/*
 * @file color_bar.h 
 * @Brief a color bar class for OpenGL.
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

#ifndef PHYSIKA_RENDER_COLORBAR_COLOR_BAR_H_
#define PHYSIKA_RENDER_COLORBAR_COLOR_BAR_H_

#include "Physika_Render/ColorBar/ColorMap/color_map.h"
#include "Physika_Core/Vectors/vector_2d.h"

namespace Physika{

template <typename Scalar>
class ColorBar
{
public:
    ColorBar();
    ~ColorBar();

    //getter
    const Vector<Scalar, 2> & startPoint() const;
    unsigned int width() const;
    unsigned int height() const;
    const ColorMap<Scalar> & colorMap() const;

    //setter
    void setColorMap(const ColorMap<Scalar> & color_map);
    void setColorMapSize(unsigned int color_size);
    void setColorMapType(ColorMapType color_map_type);
    void setColorMapTypeAndSize(ColorMapType color_map_type, unsigned int color_size);
    void setStartPoint(const Vector<Scalar, 2> & start_point);
    void setWidth(unsigned int width); // in the unit of pixels
    void setHeight(unsigned int height); // in the unit of pixels
    void setWidthAndHeight(unsigned int width, unsigned int height);

    void enableHorizon();  //put the colorbar horizontally
    void disableHorizon();
    bool isHorizon() const;


protected:
    ColorMap<Scalar> color_map_;
    Vector<Scalar,2> start_point_; // left bottom corner
    unsigned int width_;  // in the unit of pixels
    unsigned int height_; // in the unit of pixels
    bool enable_horizon_;  //whether to put the colorbar horizontally, default: false
};


} // end of namespace Physika

#endif //PHYSIKA_RENDER_COLORBAR_COLOR_BAR_H