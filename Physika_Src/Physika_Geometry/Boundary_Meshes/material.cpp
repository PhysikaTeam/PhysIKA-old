/*
 * @file material.cpp 
 * @brief material of 3d surface mesh and 2d polygon
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

#include "Physika_Geometry/Boundary_Meshes/material.h"
using std::string;

namespace Physika{

namespace BoundaryMeshInternal{

template <typename Scalar>
Material<Scalar>::Material()
{
}

template <typename Scalar>
Material<Scalar>::~Material()
{
}

template <typename Scalar>
Material<Scalar>::Material(const string &name, const Vector<Scalar,3> &Ka, const Vector<Scalar,3> &Kd,
                           const Vector<Scalar,3> &Ks, Scalar shininess, Scalar alpha, const string &texture_file_name)
    :name_(name),Ka_(Ka),Kd_(Kd),Ks_(Ks),shininess_(shininess),alpha_(alpha),full_path_texture_file_name_(texture_file_name)
{
}

template <typename Scalar>
Material<Scalar>::Material(const Material<Scalar> &material)
    :name_(material.name_),Ka_(material.Ka_),Kd_(material.Kd_),Ks_(material.Ks_),
     shininess_(material.shininess_),alpha_(material.alpha_),full_path_texture_file_name_(material.full_path_texture_file_name_)
{
}

template <typename Scalar>
Material<Scalar>& Material<Scalar>::operator= (const Material<Scalar> &material)
{
    name_ = material.name_;
    Ka_ = material.Ka_;
    Kd_ = material.Kd_;
    Ks_ = material.Ks_;
    shininess_ = material.shininess_;
    alpha_ = material.alpha_;
    full_path_texture_file_name_ = material.full_path_texture_file_name_;
    return *this;
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
Vector<Scalar,3> Material<Scalar>::Ka() const
{
    return Ka_;
}

template <typename Scalar>
void Material<Scalar>::setKa(const Vector<Scalar,3> &Ka)
{
    Ka_ = Ka;
}

template <typename Scalar>
Vector<Scalar,3> Material<Scalar>::Kd() const
{
    return Kd_;
}

template <typename Scalar>
void Material<Scalar>::setKd(const Vector<Scalar,3> &Kd)
{
    Kd_ = Kd;
}

template <typename Scalar>
Vector<Scalar,3> Material<Scalar>::Ks() const
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
    return full_path_texture_file_name_.size()>0;
}

template <typename Scalar>
const string& Material<Scalar>::textureFileName() const
{
    return full_path_texture_file_name_;
}

template <typename Scalar>
void Material<Scalar>::setTextureFileName(const string &texture_file_name)
{
    full_path_texture_file_name_ = texture_file_name;
}

template <typename Scalar>
Material<Scalar> Material<Scalar>::Rubber()
{
	// materail name,ka,kd,ks,shiness,texture file name
	return Material<Scalar>(string("Rubber"),Vector<Scalar,3>(1.0, 1.0, 1.0),Vector<Scalar,3>(1.0, 1.0, 1.0),
                            Vector<Scalar,3>(1.0, 1.0, 1.0),1.0,1.0,string(""));
}

template <typename Scalar>
Material<Scalar> Material<Scalar>::Iron()
{
	// materail name,ka,kd,ks,shiness,texture file name
	return Material<Scalar>(string("Iron"),Vector<Scalar,3>(0.1),Vector<Scalar,3>(0.5),
                            Vector<Scalar,3>(0),65,1.0,string(""));
}

template <typename Scalar>
Material<Scalar> Material<Scalar>::Wood()
{
	// materail name,ka,kd,ks,shiness,texture file name
	return Material<Scalar>(string("Wood"),Vector<Scalar,3>(1.0, 1.0, 1.0),Vector<Scalar,3>(1.0, 1.0, 1.0),
                            Vector<Scalar,3>(1.0, 1.0, 1.0),1.0,1.0,string(""));
}

//explicit instantitation
template class Material<float>;
template class Material<double>;

} //end of namespace BoundaryMeshInternal

} //end of namespace Physika
