/*
 * @file vertex.cpp 
 * @brief vertex of 3d surface mesh and 2d polygon
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Boundary_Meshes/vertex.h"

namespace Physika{

namespace BoundaryMeshInternal{

template <typename Scalar>
Vertex<Scalar>::Vertex()
    :has_normal_(false),has_texture_(false)
{
}

template <typename Scalar>
Vertex<Scalar>::~Vertex()
{
}

template <typename Scalar>
Vertex<Scalar>::Vertex(unsigned int position_index)
    :position_index_(position_index),has_normal_(false),has_texture_(false)
{
}

template <typename Scalar>
Vertex<Scalar>::Vertex(unsigned int position_index, unsigned int normal_index)
    :position_index_(position_index),normal_index_(normal_index),has_normal_(true),has_texture_(false)
{
}

template <typename Scalar>
Vertex<Scalar>::Vertex(unsigned int position_index, unsigned int normal_index, unsigned int texture_index)
    :position_index_(position_index),normal_index_(normal_index),texture_index_(texture_index),has_normal_(true),has_texture_(true)
{
}

template <typename Scalar>
unsigned int Vertex<Scalar>::positionIndex() const
{
    return position_index_;
}

template <typename Scalar>
void Vertex<Scalar>::setPositionIndex(unsigned int position_index)
{
    position_index_ = position_index;
}

template <typename Scalar>
unsigned int Vertex<Scalar>::normalIndex() const
{
    PHYSIKA_ASSERT(has_normal_);
    return normal_index_;
}

template <typename Scalar>
void Vertex<Scalar>::setNormalIndex(unsigned int normal_index)
{
    normal_index_ = normal_index;
    has_normal_ = true;
}

template <typename Scalar>
unsigned int Vertex<Scalar>::textureCoordinateIndex() const
{
    PHYSIKA_ASSERT(has_texture_);
    return texture_index_;
}

template <typename Scalar>
void Vertex<Scalar>::setTextureCoordinateIndex(unsigned int texture_index)
{
    texture_index_ = texture_index;
    has_texture_ = true;
}

template <typename Scalar>
bool Vertex<Scalar>::hasNormal() const
{
    return has_normal_;
}

template <typename Scalar>
bool Vertex<Scalar>::hasTexture() const
{
    return has_texture_;
}

//explicit instantitation
template class Vertex<float>;
template class Vertex<double>;

} //end of namespace BoundaryMeshInternal

} //end of namespace Physika
