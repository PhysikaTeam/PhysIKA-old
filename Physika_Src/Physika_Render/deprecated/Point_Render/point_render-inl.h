/*
 * @file point_render-inl.h 
 * @Brief render of points.
 * @author Fei Zhu.
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_INL_H_
#define PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_INL_H_

#include "Physika_Core/Utilities/physika_exception.h"

namespace Physika{

template <typename Scalar, int Dim>
template <typename ColorScalar>
PointRender<Scalar,Dim>::PointRender(const Vector<Scalar,Dim> *points, unsigned int point_num, Scalar point_size, const Color<ColorScalar> &point_color)
    :points_(points),point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(point_color);
    render_mode_ = point_mode_;
}

template <typename Scalar, int Dim>
template <typename ColorScalar>
    PointRender<Scalar,Dim>::PointRender(const Vector<Scalar,Dim> *points, unsigned int point_num, const std::vector<Scalar> &point_size, const Color<ColorScalar> &point_color)
    :points_(points),point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(point_color);
    render_mode_ = point_mode_;
}
   
template <typename Scalar, int Dim>
template <typename ColorScalar>
PointRender<Scalar,Dim>::PointRender(const Vector<Scalar,Dim> *points, unsigned int point_num, Scalar point_size, const std::vector<Color<ColorScalar> > &point_color)
    :points_(points),point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(point_color);
    render_mode_ = point_mode_;
}
   
template <typename Scalar, int Dim>
template <typename ColorScalar>
PointRender<Scalar,Dim>::PointRender(const Vector<Scalar,Dim> *points, unsigned int point_num, const std::vector<Scalar> &point_size, const std::vector<Color<ColorScalar> > &point_color)
    :points_(points),point_num_(point_num)
{
    setPointSize(point_size);
    setPointColor(point_color);
    render_mode_ = point_mode_;
}
  
template <typename Scalar, int Dim>
template <typename ColorScalar>
Color<ColorScalar> PointRender<Scalar,Dim>::pointColor(unsigned int point_idx) const
{
    if(point_idx>=point_num_)
        throw PhysikaException("Point index out of range!");
    if(colors_.size()==1)
        return colors_[0].template convertColor<ColorScalar>();
    else if(point_idx>=colors_.size())
        return Color<ColorScalar>::Green();
    else
        return colors_[point_idx].template convertColor<ColorScalar>();
}
  
template <typename Scalar, int Dim>
template <typename ColorScalar>
void PointRender<Scalar,Dim>::setPointColor(unsigned int point_idx, const Color<ColorScalar> &point_color)
{
    if(point_idx>=point_num_)
        throw PhysikaException("Point index out of range!");
    if(colors_.size() == 1) //either only one point or all points use one color
    {
        Color<Scalar> color_before = colors_[0];
        colors_.resize(point_num_);
        for(unsigned int i = 0; i < point_idx; ++i)
            colors_[i] = color_before;
        colors_[point_idx] = point_color.template convertColor<Scalar>();
        for(unsigned int i = point_idx + 1; i < point_num_; ++i)
            colors_[i] = color_before;
    }
    else if(point_idx>=colors_.size())  //previously color unset
    {
        for(unsigned int i = 0; i < point_idx - colors_.size(); ++i)
            colors_.push_back(Color<Scalar>::Green());  //set Green for previous unset points
        colors_.push_back(point_color.template convertColor<Scalar>());
    }
    else
        colors_[point_idx] = point_color.template convertColor<Scalar>();
}
  
template <typename Scalar, int Dim>
template <typename ColorScalar>
void PointRender<Scalar,Dim>::setPointColor(const Color<ColorScalar> &point_color)
{
    colors_.clear();
    colors_.push_back(point_color.template convertColor<Scalar>());
}
  
template <typename Scalar, int Dim>
template <typename ColorScalar>
void PointRender<Scalar,Dim>::setPointColor(const std::vector<Color<ColorScalar> > &point_colors)
{
    unsigned int min_point_num = point_colors.size()<point_num_ ? point_colors.size() : point_num_;
    colors_.resize(min_point_num);
    for(unsigned int i = 0; i < min_point_num; ++i)
        colors_[i] = point_colors[i].template convertColor<Scalar>();
}

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_POINT_RENDER_POINT_RENDER_INL_H_
