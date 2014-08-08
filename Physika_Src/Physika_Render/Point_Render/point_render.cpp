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

#include <cstddef>
#include <iostream>
#include "Physika_Render/Point_Render/point_render.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika{

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender()
    :points_(NULL), points_num_(0),
    colors_(NULL), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, unsigned int points_num, float point_size_)
    :points_(points), points_num_(points_num),
    colors_(NULL), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, unsigned int points_num, const Color<Scalar>* colors, float point_size_)
    :points_(points), points_num_(points_num),
    colors_(colors), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::~PointRender()
{

}

template<typename Scalar, int Dim>
void PointRender<Scalar, Dim>::render()
{
    if(this->points_==NULL)
    {
        std::cerr<<"No Point is binded to the PointRender!\n";
        return;

    }

    //Draw points;
    glDisable(GL_LIGHTING);
    glPointSize(point_size_);
    glBegin(GL_POINTS);
    
    for (unsigned int i=0; i<this->points_num_; i++) 
    {
        if(colors_ != NULL)
            openGLColor3(colors_[i]);
        else
        {
            //TO DO get color from color_map and ref_values;
            openGLColor3(Color<Scalar>::Blue());
        }
        openGLVertex(points_[i]);
    }
    glEnd();
}

template class PointRender<float ,3>;
template class PointRender<float ,2>;
template class PointRender<double ,3>;
template class PointRender<double, 2>;

} //end of namespace Physika
