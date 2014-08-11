/*
 * @file point_render.h 
 * @Brief render of points.
 * @author Sheng Yang, Fei Zhu.
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
#include <GL/glut.h>
#include "Physika_Render/Point_Render/point_render.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

namespace Physika{

template<typename Scalar, int Dim>
const unsigned int PointRender<Scalar,Dim>::point_mode_ = 1<<0;
template<typename Scalar, int Dim>
const unsigned int PointRender<Scalar,Dim>::sphere_mode_ = 1<<1;
template<typename Scalar, int Dim>
const Scalar PointRender<Scalar,Dim>::default_point_size_ = 5.0;

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender()
    :points_(NULL), point_num_(0)
{
    setPointSize(default_point_size_);
    setPointColor(Color<Scalar>::Green());
    render_mode_ = point_mode_;
}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim> *points, unsigned int point_num)
    :points_(points), point_num_(point_num)
{
    setPointSize(default_point_size_);
    setPointColor(Color<Scalar>::Green());
    render_mode_ = point_mode_;
}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, unsigned int point_num, Scalar point_size)
    :points_(points), point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(Color<Scalar>::Green());
    render_mode_ = point_mode_;
}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::PointRender(const Vector<Scalar, Dim>* points, unsigned int point_num, const std::vector<Scalar> &point_size)
    :points_(points), point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(Color<Scalar>::Green());
    render_mode_ = point_mode_;
}

template<typename Scalar, int Dim>
PointRender<Scalar, Dim>::~PointRender()
{

}

template<typename Scalar, int Dim>
const Vector<Scalar,Dim>* PointRender<Scalar,Dim>::points() const
{
    return points_;
}

template<typename Scalar, int Dim>
unsigned int PointRender<Scalar,Dim>::pointNum() const
{
    return point_num_;
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setPoints(const Vector<Scalar,Dim> *points, unsigned int point_num)
{
    if(points==NULL && point_num>0)
    {
        std::cerr<<"Warning: NULL pointer passed to PointRender, ignore the input point number!\n";
        points_ = points;
        point_num_ = 0;
    }
    else
    {
        points_ = points;
        point_num_ = point_num;
    }
}

template<typename Scalar, int Dim>
Scalar PointRender<Scalar,Dim>::pointSize(unsigned int point_idx) const
{
    if(point_idx>=point_num_)
    {
        std::cerr<<"Error: Point index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    if(point_sizes_.size()==1)
        return point_sizes_[0];
    else if(point_idx>=point_sizes_.size())
        return default_point_size_;
    else
        return point_sizes_[point_idx];
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setPointSize(unsigned int point_idx, Scalar point_size)
{
    if(point_idx>=point_num_)
    {
        std::cerr<<"Error: Point index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    if(point_size<=0)
    {
        std::cerr<<"Warning: invalid point size is provided, "<<default_point_size_<<" is used instead!\n";
        point_size = default_point_size_;
    }
    if(point_idx >= point_sizes_.size())
    {
        for(unsigned int i = 0; i < point_idx - point_sizes_.size(); ++i)
            point_sizes_.push_back(default_point_size_);
        point_sizes_.push_back(point_size);
    }
    else
        point_sizes_.push_back(point_size);
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setPointSize(Scalar point_size)
{
    point_sizes_.clear();
    if(point_size <= 0)
    {
        std::cerr<<"Warning: invalid point size is provided,  "<<default_point_size_<<" is used instead!\n";
        point_sizes_.push_back(default_point_size_);
    }
    else
        point_sizes_.push_back(point_size);
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setPointSize(const std::vector<Scalar> &point_sizes)
{
    bool invalid_point_size = false;
    unsigned int min_point_num = point_sizes.size()<point_num_ ? point_sizes.size() : point_num_;
    point_sizes_.resize(min_point_num);
    for(unsigned int i = 0; i < min_point_num; ++i)
    {
        if(point_sizes[i]<=0)
        {
            invalid_point_size = true;
            point_sizes_[i] = default_point_size_;
        }
        else
            point_sizes_[i] = point_sizes[i];
    }
    if(invalid_point_size)
        std::cerr<<"Warning: there exists invalid point size, "<<default_point_size_<<" is used instead!\n";
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setRenderAsPoint()
{
    render_mode_ |= point_mode_;
    render_mode_ &= ~sphere_mode_;
}

template<typename Scalar, int Dim>
void PointRender<Scalar,Dim>::setRenderAsSphere()
{
    render_mode_ |= sphere_mode_;
    render_mode_ &= ~point_mode_;
}

template<typename Scalar, int Dim>
void PointRender<Scalar, Dim>::render()
{
    if(this->points_==NULL)
    {
        std::cerr<<"Warning: No Point is binded to the PointRender!\n";
        return;
    }

    //Draw points;
    glDisable(GL_LIGHTING);
    
    for (unsigned int i=0; i<this->point_num_; i++) 
    {
        if(render_mode_ & point_mode_)
        {
            glPointSize(pointSize(i));
            glBegin(GL_POINTS);
            openGLColor3(pointColor<Scalar>(i));
            openGLVertex(points_[i]);
            glEnd();
        }
        else if(render_mode_ & sphere_mode_)
        {
            openGLColor3(pointColor<Scalar>(i));
            openGLTranslate(points_[i]);
            glutSolidSphere(pointSize(i),20,20);
            openGLTranslate(-points_[i]);
        }
    }
}

template class PointRender<float ,3>;
template class PointRender<float ,2>;
template class PointRender<double ,3>;
template class PointRender<double, 2>;

} //end of namespace Physika
