/*
 * @file material.h 
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_MATERIAL_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_MATERIAL_H_

#include <string>
#include "Physika_Core/Vectors/vector_3d.h"
using std::string;

namespace Physika{

namespace SurfaceMeshInternal{

template <typename Scalar>
class Material
{
public:
    Material();
    ~Material();
    Material(const string &name, const Vector<Scalar,3> &Ka, const Vector<Scalar,3> &Kd, const Vector<Scalar,3> &Ks, Scalar shininess, const string &texture_file_name=string());
    const string& name() const;
    void setName(const string &name);
    const Vector<Scalar,3>& Ka() const;
    void setKa(const Vector<Scalar,3> &Ka);
    const Vector<Scalar,3>& Kd() const;
    void setKd(const Vector<Scalar,3> &Kd);
    const Vector<Scalar,3>& Ks() const;
    void setKs(const Vector<Scalar,3> &Ks);
    Scalar shininess() const;
    void setShininess(Scalar);
    Scalar alpha() const;
    void setAlpha(Scalar);
    bool hasTexture() const;
    const string& textureFileName() const;
    void setTextureFileName(const string &texture_file_name);

    //TO DO: Add some static member methods here, predefined materials: rubber, iron, etc.

protected:
    Vector<Scalar,3> Ka_, Kd_, Ks_;
    Scalar shininess_;
    Scalar alpha_;
    string name_;
    string texture_file_name_;
};

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_MATERIAL_H_







