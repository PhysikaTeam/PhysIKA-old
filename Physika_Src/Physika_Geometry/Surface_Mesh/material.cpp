/*
 * @file material.cpp 
 * @brief material of 3d surface mesh
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Surface_Mesh/material.h"
using std::string;

namespace Physika{

namespace SurfaceMeshInternal{

template <typename Scalar>
Material<Scalar>::Material()
{
}

template <typename Scalar>
Material<Scalar>::~Material()
{
}

template <typename Scalar>
Material<Scalar>::Material(const string &name, const Vector<Scalar,3> &Ka, const Vector<Scalar,3> &Kd, const Vector<Scalar,3> &Ks, Scalar shininess, const string &texture_file_name)
    :name_(name),Ka_(Ka),Kd_(Kd),Ks_(Ks),shininess_(shininess),texture_file_name_(texture_file_name)
{
}

template <typename Scalar>
const string& Material<Scalar>::name() const
{
    return name_;
}

template <typename Scalar>
void Material<Scalar>::setName(const string &name)
{
    name_ = name;
}

template <typename Scalar>
const Vector<Scalar,3>& Material<Scalar>::Ka() const
{
    return Ka_;
}

template <typename Scalar>
void Material<Scalar>::setKa(const Vector<Scalar,3> &Ka)
{
    Ka_ = Ka;
}

template <typename Scalar>
const Vector<Scalar,3>& Material<Scalar>::Kd() const
{
    return Kd_;
}

template <typename Scalar>
void Material<Scalar>::setKd(const Vector<Scalar,3> &Kd)
{
    Kd_ = Kd;
}

template <typename Scalar>
const Vector<Scalar,3>& Material<Scalar>::Ks() const
{
    return Ks_;
}

template <typename Scalar>
void Material<Scalar>::setKs(const Vector<Scalar,3> &Ks)
{
    Ks_ =Ks;
}

template <typename Scalar>
Scalar Material<Scalar>::shininess() const
{
    return shininess_;
}

template <typename Scalar>
void Material<Scalar>::setShininess(Scalar shininess)
{
    shininess_ = shininess;
}

template <typename Scalar>
Scalar Material<Scalar>::alpha() const
{
    return alpha_;
}

template <typename Scalar>
void Material<Scalar>::setAlpha(Scalar alpha)
{
    alpha_ = alpha;
}

template <typename Scalar>
bool Material<Scalar>::hasTexture() const
{
    return texture_file_name_.size()>0;
}

template <typename Scalar>
const string& Material<Scalar>::textureFileName() const
{
    return texture_file_name_;
}

template <typename Scalar>
void Material<Scalar>::setTextureFileName(const string &texture_file_name)
{
    texture_file_name_ = texture_file_name;
}

//explicit instantitation
template class Material<float>;
template class Material<double>;

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika
