/*
 * @file point_render.h 
 * @Brief render of points.
 * @author Sheng Yang, Fei Zhu.
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
#define PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_

#include <vector>
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/Color/color.h"


namespace Physika{

/*
 * PointRender: render point data, optionally with custom size and color
 * point size is number of pixels if rendered as points, or the
 * radius of sphere if rendered as sphere
 */

template <typename Scalar, int Dim>
class PointRender: public RenderBase
{
public:
    //Constructions
    PointRender(); 
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num);  //points with default size and color
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num, Scalar point_size); //points with same size and default color
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num, const std::vector<Scalar> &point_size);  //points with different size and default color
    //points with same size and same color
    template <typename ColorScalar>
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num,
                Scalar point_size, const Color<ColorScalar> &point_color);
    //points with different size and same color
    template <typename ColorScalar>
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num,
                const std::vector<Scalar> &point_size, const Color<ColorScalar> &point_color);
    //points with same size and different color
    template <typename ColorScalar>
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num,
                Scalar point_size, const std::vector<Color<ColorScalar> > &point_color);
    //points with different size and different color
    template <typename ColorScalar>
    PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num,
                const std::vector<Scalar> &point_size, const std::vector<Color<ColorScalar> > &point_color);

    ~PointRender();

    //Get and Set
    const Vector<Scalar, Dim>* points() const;
    unsigned int pointNum() const;
    void setPoints(const Vector<Scalar, Dim>* points, unsigned int point_num);
    Scalar pointSize(unsigned int point_idx) const;
    void setPointSize(unsigned int point_idx, Scalar point_size);
    void setPointSize(Scalar point_size);  //all points with same size
    void setPointSize(const std::vector<Scalar> &point_sizes);  //points  with different size
    template <typename ColorScalar>
    Color<ColorScalar> pointColor(unsigned int point_idx) const;
    template <typename ColorScalar>
    void setPointColor(unsigned int point_idx, const Color<ColorScalar> &point_color);
    template <typename ColorScalar>
    void setPointColor(const Color<ColorScalar> &point_color);  //all points with same color
    template <typename ColorScalar>
    void setPointColor(const std::vector<Color<ColorScalar> > &point_colors);  //points with different color

    //render mode
    void setRenderAsPoint();
    void setRenderAsSphere();  //sphere radius must be properly set


    //Render
    virtual void render();

protected:
    //data
    const Vector<Scalar, Dim>* points_; 
    unsigned int point_num_;
    //the size of following two vectors may be smaller than point number
    //if vec.size() == 1, all points use same value
    //if vec.size() < point number, the last points use default value
    //if vec.size() == point number, each point uses one value
    std::vector<Scalar> point_sizes_;  
    std::vector<Color<Scalar> > colors_;

    //render mode
    unsigned int render_mode_;
    //predefined render modes
    const static unsigned int point_mode_;   //render points as opengl points
    const static unsigned int sphere_mode_;  //render points as sphere
    //default point size
    const static Scalar default_point_size_;
};

} //end of namespace Physika

//implementation
#include "Physika_Render/Point_Render/point_render-inl.h"

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_H_
