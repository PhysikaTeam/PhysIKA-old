/*
 * @file color_map.cpp
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

#include <cmath>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Render/ColorBar/ColorMap/color_map.h"

namespace Physika{

template <typename Scalar>
ColorMap<Scalar>::ColorMap()
{
    this->color_map_vec_.resize(64);
    this->color_map_type_ = ColorMapType::Jet;
    this->jetColorMap();
}

template <typename Scalar>
ColorMap<Scalar>::ColorMap(ColorMapType color_map_type, unsigned int color_size)
{
    this->color_map_vec_.resize(color_size);
    this->setColorMapType(color_map_type);
}

template <typename Scalar>
ColorMap<Scalar>::~ColorMap()
{

}

template <typename Scalar>
ColorMapType ColorMap<Scalar>::colorMapType() const
{
    return this->color_map_type_;
}

template <typename Scalar>
unsigned int ColorMap<Scalar>::colorMapSize() const
{
    return this->color_map_vec_.size();
}

template <typename Scalar>
const std::vector<Color<Scalar> > & ColorMap<Scalar>::colorMapVec() const
{
    return this->color_map_vec_;
}

template <typename Scalar>
const Color<Scalar> & ColorMap<Scalar>::colorViaRatio(Scalar ratio) const
{
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    unsigned int index = ratio*this->color_map_vec_.size();
    return this->colorViaIndex(index);
}

template <typename Scalar>
const Color<Scalar> & ColorMap<Scalar>::colorViaIndex(unsigned int index)const
{
    if (index >= this->color_map_vec_.size()) index = this->color_map_vec_.size()-1;
    return this->color_map_vec_[index];
}

template <typename Scalar>
void ColorMap<Scalar>::setColorMapSize(unsigned int color_size)
{
    if (color_size < 1)
    {
        std::cerr<<"color map size should be greater than 1.\n";
        std::exit(EXIT_FAILURE);
    }
    this->color_map_vec_.resize(color_size);
    this->setColorMapType(this->color_map_type_);
}

template <typename Scalar>
void ColorMap<Scalar>::setColorMapType(ColorMapType color_map_type)
{
    
    for (unsigned int i=0; i< color_map_vec_.size(); i++)
    {
        color_map_vec_[i] = Color<Scalar>();
    }
    
    this->color_map_type_ = color_map_type;
    switch(color_map_type)
    {
        case ColorMapType::Gray:   {grayColorMap(); break;}
        case ColorMapType::Red:    {redColorMap(); break;}
        case ColorMapType::Green:  {greenColorMap(); break;}
        case ColorMapType::Blue:   {blueColorMap(); break;}
        case ColorMapType::Jet:    {jetColorMap(); break;}
        case ColorMapType::Spring: {springColorMap(); break;}
        case ColorMapType::Summer: {summerColorMap(); break;}
        case ColorMapType::Autumn: {autumnColorMap(); break;}
        case ColorMapType::Winter: {winterColorMap(); break;}
        case ColorMapType::Hot:    {hotColorMap(); break;}
        case ColorMapType::Cool:   {coolColorMap(); break;}
    }
}

template <typename Scalar>
void ColorMap<Scalar>::setColorMapTypeAndSize(ColorMapType color_map_type, unsigned int color_size)
{
    if (color_size < 1)
    {
        std::cerr<<"color map size should be greater than 1.\n";
        std::exit(EXIT_FAILURE);
    }
    this->color_map_vec_.resize(color_size);
    this->setColorMapType(color_map_type);
}

template <typename Scalar>
void ColorMap<Scalar>::grayColorMap()
{
    unsigned int m = max<unsigned int>(color_map_vec_.size()-1,1);
    for (unsigned int i=0; i<color_map_vec_.size(); i++)
    {
        Scalar g = static_cast<Scalar>(i)/m;
        color_map_vec_[i].setRedChannel(g);
        color_map_vec_[i].setGreenChannel(g);
        color_map_vec_[i].setBlueChannel(g);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::redColorMap()
{
    unsigned int m = max<unsigned int>(color_map_vec_.size()-1,1);
    for (unsigned int i=0; i<color_map_vec_.size(); i++)
    {
        Scalar r = static_cast<Scalar>(i)/m;
        color_map_vec_[i].setRedChannel(r);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::greenColorMap()
{
    unsigned int m = max<unsigned int>(color_map_vec_.size()-1,1);
    for (unsigned int i=0; i<color_map_vec_.size(); i++)
    {
        Scalar g = static_cast<Scalar>(i)/m;
        color_map_vec_[i].setGreenChannel(g);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::blueColorMap()
{
    unsigned int m = max<unsigned int>(color_map_vec_.size()-1,1);
    for (unsigned int i=0; i<color_map_vec_.size(); i++)
    {
        Scalar b = static_cast<Scalar>(i)/m;
        color_map_vec_[i].setBlueChannel(b);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::jetColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = std::ceil((float)m/4);
    std::vector<Scalar> u;
    for (unsigned int i=1; i<=n; i++)
    {
        u.push_back(static_cast<Scalar>(i)/n);
    }
    for (unsigned int i=1; i<=n-1; i++)
    {
        u.push_back(1.0);
    }
    for (unsigned int i=n; i>=1; i--)
    {
        u.push_back(static_cast<Scalar>(i)/n);
    }
    std::vector<unsigned int> g;
    int temp = std::ceil((float)n/2) - (m%4==1);
    for (unsigned int i=1; i<=u.size(); i++)
    {
        int temp_g = temp+i;
        if(temp_g<=m) g.push_back(temp_g);
    }
    std::vector<unsigned int> r;
    for (unsigned int i=0; i< g.size(); i++)
    {
        int temp_r = g[i]+n;
        if(temp_r<=m) r.push_back(temp_r);
    }
    std::vector<unsigned int> b;
    for (unsigned int i=0; i< g.size(); i++)
    {
        int temp_b = g[i]-n;
        if(temp_b>=1) b.push_back(temp_b);
    }

    for (unsigned int i=0; i<r.size(); i++)
    {
        color_map_vec_[r[i]-1].setRedChannel(u[i]);
    }
    for (unsigned int i=0; i<g.size(); i++)
    {
        color_map_vec_[g[i]-1].setGreenChannel(u[i]);
    }
    temp = u.size() - b.size();
    for (unsigned int i=0; i<b.size(); i++)
    {
        color_map_vec_[b[i]-1].setBlueChannel(u[temp+i]);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::springColorMap()
{
    unsigned int m = max<unsigned int>(color_map_vec_.size()-1,1);
    for (unsigned int i=0; i<color_map_vec_.size(); i++)
    {
        Scalar r = static_cast<Scalar>(i)/m;
        color_map_vec_[i].setRedChannel(1.0);
        color_map_vec_[i].setGreenChannel(r);
        color_map_vec_[i].setBlueChannel(1.0-r);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::summerColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = max<unsigned int>(m-1,1);
    std::vector<Scalar> r;
    for (unsigned int i=0; i<= m-1; i++)
    {
        r.push_back(static_cast<Scalar>(i)/n);
    }
    for (unsigned int i=0; i<m; i++)
    {
        color_map_vec_[i].setRedChannel(r[i]);
        color_map_vec_[i].setGreenChannel(0.5+r[i]/2);
        color_map_vec_[i].setBlueChannel(0.4);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::autumnColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = max<unsigned int>(m-1,1);
    std::vector<Scalar> r;
    for (unsigned int i=0; i<= m-1; i++)
    {
        r.push_back(static_cast<Scalar>(i)/n);
    }
    for (unsigned int i=0; i<m; i++)
    {
        color_map_vec_[i].setRedChannel(1.0);
        color_map_vec_[i].setGreenChannel(r[i]);
        color_map_vec_[i].setBlueChannel(0.0);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::winterColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = max<unsigned int>(m-1,1);
    std::vector<Scalar> r;
    for (unsigned int i=0; i<= m-1; i++)
    {
        r.push_back(static_cast<Scalar>(i)/n);
    }
    for (unsigned int i=0; i<m; i++)
    {
        color_map_vec_[i].setRedChannel(0.0);
        color_map_vec_[i].setGreenChannel(r[i]);
        color_map_vec_[i].setBlueChannel(0.5+(1-r[i])/2);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::hotColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = floor(3.0/8.0*m);
    std::vector<Scalar> r;
    for (unsigned int i=1; i<=n ;i++)
    {
        r.push_back(static_cast<Scalar>(i)/n);
    }
    r.insert(r.end(), m-n, 1);

    std::vector<Scalar> g;
    g.insert(g.end(),n,0);
    for (unsigned int i=1; i<=n; i++)
    {
        g.push_back(static_cast<Scalar>(i)/n);
    }
    g.insert(g.end(), m-2*n, 1);

    std::vector<Scalar> b;
    b.insert(b.end(),2*n,0);
    for (unsigned int i=1; i<= m-2*n; i++)
    {
        b.push_back(static_cast<Scalar>(i)/(m-2*n));
    }
    for (unsigned int i=0; i<m; i++)
    {
        color_map_vec_[i].setRedChannel(r[i]);
        color_map_vec_[i].setGreenChannel(g[i]);
        color_map_vec_[i].setBlueChannel(b[i]);
    }
}

template <typename Scalar>
void ColorMap<Scalar>::coolColorMap()
{
    unsigned int m = color_map_vec_.size();
    unsigned int n = max<unsigned int>(m-1,1);
    std::vector<Scalar> r;
    for (unsigned int i=0; i<= m-1; i++)
    {
        r.push_back(static_cast<Scalar>(i)/n);
    }
    for (unsigned int i=0; i<m; i++)
    {
        color_map_vec_[i].setRedChannel(r[i]);
        color_map_vec_[i].setGreenChannel(1-r[i]);
        color_map_vec_[i].setBlueChannel(1.0);
    }
}

//explicit instantiations
template class ColorMap<float>;
template class ColorMap<double>;

} // end of namespace Physika