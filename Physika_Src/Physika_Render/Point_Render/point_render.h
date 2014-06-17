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
#include "Physika_Core/Arrays/array.h"
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"


namespace Physika{

template <typename Scalar> class SurfaceMesh;

template <typename Scalar, int Dim>
class PointRender: public RenderBase
{
public:
    //Constructions
    PointRender(); //default mode
    PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num); //provide points
    PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num, const Scalar* map, const Scalar &ref_value = 0); //provide points maps and its ref value; and render provide color calculating
    PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num, const Vector<Scalar, 3>* color); //provide color.
    ~PointRender();

    //Get and Set
    const Vector<Scalar, Dim>* points() const { return points_; }
    const unsigned int pointsNum() const { return points_num_; }
    const Scalar* map() const { return map_; }
    const Scalar ref_value() const { return ref_value_; }
    const Vector<Scalar, 3>* color() const { return color_; }
    float pointSize() { return point_size_; }


    void setPoints(const Vector<Scalar, Dim>* points) { points_ = points; }
    void setMap(const Scalar* map) { map_ = map; /* ref_value_ = ref_value;*/ }
    void setRefValue(const Scalar& ref_value) { ref_value_ = ref_value; }
    void setColor(const Vector<Scalar, 3>* color) { color_ = color; }
    void setPointSize(float point_size) { point_size_ = point_size; }


    //Calculate Colors;
    void calculateColor();

    //whenever the mesh is modified, synchronize() must be called to update the render
    void synchronize();   

    //Render
    virtual void render();

protected:
    //render mode
    unsigned int render_mode_;
    //data
    const Vector<Scalar, Dim> *points_;
    unsigned int points_num_;
    //color map
    const Scalar *map_;
    //ref value;
    Scalar ref_value_;
    //colors
    const Vector<Scalar, 3> *color_;

    //point size;
    float point_size_; 
 
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
