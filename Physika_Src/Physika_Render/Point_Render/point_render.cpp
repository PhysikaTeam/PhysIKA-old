/*
 * @file surface_mesh_render.cpp 
 * @Basic render of surface mesh.
 * @author Fei Zhu ,Wei Chen
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
#include "Physika_IO/Image_IO/image_io.h"
#include "GL/gl.h"


namespace Physika{

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender()
    :points_(NULL), points_num_(0),
    map_(NULL), ref_value_(0),
    color_(NULL), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num)
    :points_(points), points_num_(points_num),
    map_(NULL), ref_value_(0),
    color_(NULL), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num, const Scalar* map, const Scalar &ref_value /* = 0 */)
    :points_(points), points_num_(points_num),
    map_(map), ref_value_(ref_value),
    color_(NULL), point_size_(3)
{

}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, const unsigned int& points_num, const Vector<Scalar, 3>* color)
    :points_(points), points_num_(points_num),
    map_(NULL), ref_value_(0),
    color_(color), point_size_(3)
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

    //Draw axis;
    glDisable(GL_LIGHTING);
    glLineWidth(4);
    glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0,0,0);
    glVertex3f(2,0,0);
    glColor3f(0,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,2,0);
    glColor3f(0,0,1);
    glVertex3f(0,0,0);
    glVertex3f(0,0,2);
    glEnd();

    //Draw points;
    glDisable(GL_LIGHTING);
    glPointSize(point_size_);
    glBegin(GL_POINTS);


    for (int i=0; i<this->points_num_; i++) {
        
        if(color_ != NULL)
           glColor3f(color_[i][0], color_[i][1], color_[i][2]);
        else
        {
            //TO DO get color from map and ref_values;
            glColor3f(1.f, 0.f, 0.f);
        }
        if(Dim == 2)
            glVertex3f(points_[i][0], points_[i][1], 0.f);
        else
            glVertex3f(points_[i][0], points_[i][1], points_[i][2]);
    }

    glEnd();
}


template class PointRender<float ,3>;
template class PointRender<float ,2>;
template class PointRender<double ,3>;
template class PointRender<double, 2>;

} //end of namespace Physika
