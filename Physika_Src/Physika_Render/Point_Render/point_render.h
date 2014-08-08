/*
 * @file point_render.h 
 * @Brief render of points.
 * @author Sheng Yang.
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

#include <utility>
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"


namespace Physika{

template <typename Scalar> class SurfaceMesh;

template <typename Scalar, int Dim>
class PointRender: public RenderBase
{
public:
    //Constructions
    PointRender(); //default mode
    PointRender(const Vector<Scalar, Dim>* points, unsigned int points_num, float point_size); //provide points
    PointRender(const Vector<Scalar, Dim>* points, unsigned int points_num, const Color<Scalar>* colors, float point_size); //provide color.
    ~PointRender();

    //Get and Set
    const Vector<Scalar, Dim>* points() const { return points_; }
    const unsigned int pointsNum() const { return points_num_; }
    const Color<Scalar>* colors() const {return colors_; }
    float pointSize() { return point_size_; }
    
    void setPoints(const Vector<Scalar, Dim>* points) { points_ = points; }
    void setPointSize(float point_size) { point_size_ = point_size; }
    void setColors(const Color<Scalar>* colors) { colors_ = colors; }
    //Render
    virtual void render();

protected:
    //render mode
    unsigned int render_mode_;
    //data
    const Vector<Scalar, Dim>* points_;
    unsigned int points_num_;

    const Color<Scalar>* colors_;
    
    //point size;
    float point_size_; 
 
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
