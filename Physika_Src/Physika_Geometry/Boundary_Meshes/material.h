/*
 * @file material.h 
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

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_MATERIAL_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_MATERIAL_H_

#include <string>
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

namespace BoundaryMeshInternal{

/*
 * 'illum','map_Ka','map_Ks' in material file are ignored
 * we assume materials can be identified by their names
 */
template <typename Scalar>
class Material
{
public:
    Material();
    ~Material();
    Material(const std::string &name, const Vector<Scalar,3> &Ka, const Vector<Scalar,3> &Kd, const Vector<Scalar,3> &Ks,
             Scalar shininess, Scalar alpha, const std::string &texture_file_name=std::string());
    Material(const Material<Scalar> &material);
    Material<Scalar>& operator= (const Material<Scalar> &material);
    const std::string& name() const;
    void setName(const std::string &name);
    Vector<Scalar,3> Ka() const;
    void setKa(const Vector<Scalar,3> &Ka);
    Vector<Scalar,3> Kd() const;
    void setKd(const Vector<Scalar,3> &Kd);
    Vector<Scalar,3> Ks() const;
    void setKs(const Vector<Scalar,3> &Ks);
    Scalar shininess() const;
    void setShininess(Scalar);
    Scalar alpha() const;
    void setAlpha(Scalar);
    bool hasTexture() const;
    const std::string& textureFileName() const;
    void setTextureFileName(const std::string &texture_file_name);

    // predefined materials: rubber, iron, etc.
	// names of the functions break the rule of coding style in order to emphasize the specialness of the static functions
	static Material<Scalar> Rubber();
	static Material<Scalar> Iron();
	static Material<Scalar> Wood();

protected:
    std::string name_;
    Vector<Scalar,3> Ka_, Kd_, Ks_;
    Scalar shininess_;
    Scalar alpha_;
    std::string full_path_texture_file_name_;  //store the full path of texture file so that texture could be correctly loaded while rendering
};

} //end of namespace BoundaryMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_MATERIAL_H_
