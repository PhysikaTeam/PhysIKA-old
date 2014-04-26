/*
 * @file vertex.h 
 * @brief vertex of 3d surface mesh
 *        position does not uniquely determine vertex, 2 vertices could have identical positions
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_

namespace Physika{

namespace SurfaceMeshInternal{

template <typename Scalar>
class Vertex
{
public:
    Vertex();
    ~Vertex();
    explicit Vertex(unsigned int position_index);
    Vertex(unsigned int position_index, unsigned int normal_index);
    Vertex(unsigned int position_index, unsigned int normal_index, unsigned int texture_index);

    unsigned int positionIndex() const;
    void setPositionIndex(unsigned int);
    unsigned int normalIndex()const;
    void setNormalIndex(unsigned int);    
    unsigned int textureCoordinateIndex() const;
    void setTextureCoordinateIndex(unsigned int);
    bool hasNormal() const;
    bool hasTexture() const;

protected:
    //indices in the mesh
    unsigned int position_index_;
    unsigned int normal_index_;
    unsigned int texture_index_;
    bool has_normal_;
    bool has_texture_;
};

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_VERTEX_H_


















