/*
 * @file point_render.h 
 * @Brief render of point, it is used to draw the simulation result.
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

#ifndef PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
#define PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename ElementType> class Array;

/*
 * PointRender is intended for both 2D&&3D points
 * Thus template parameter PointType can be either 2d vector or 3d vector
 */
template <typename PointType>
class PointRender: public RenderBase
{
public:
    /* Constructions */
    PointRender();
    PointRender(const Array<PointType> *points);
    ~PointRender();

    /* Get and Set */
    inline const PointType* pointsData() const { return points_->data(); }
    inline const Array<PointType>* points() const { return points_;}
    inline int numPoints() const { return points_->elementCount(); }
    inline void setPoints(const Array<PointType> *points) { points_ = points; }
    inline unsigned int pointSize() const { return point_size_;}
    inline void setPointSize(unsigned int point_size) { point_size_ = point_size;}
    inline void useDefaultPointSize() const { point_size_ = default_point_size_;}
    inline unsigned int defaultPointSize() const { return default_point_size_;}

    /* Render */
    virtual void render();


protected:
    const Array<PointType> *points_;
    unsigned int point_size_; //point size on screen
    static const unsigned int default_point_size_; //default point size

};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_














