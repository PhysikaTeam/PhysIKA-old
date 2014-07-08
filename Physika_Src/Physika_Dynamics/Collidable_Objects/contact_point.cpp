/*
 * @file  contact_point.cpp
 * @contact point of rigid body simulaion
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <limits>
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"

namespace Physika{

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::ContactPoint():
    contact_index_(0),
    object_lhs_index_(0),
    object_rhs_index_(0),
    global_contact_position_(0),
    global_contact_normal_lhs_(0)
{

}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::ContactPoint(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
    const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs):
    contact_index_(contact_index),
    object_lhs_index_(object_lhs_index),
    object_rhs_index_(object_rhs_index),
    global_contact_position_(global_contact_position),
    global_contact_normal_lhs_(global_contact_normal_lhs)
{

}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>::~ContactPoint()
{

}

template <typename Scalar,int Dim>
void ContactPoint<Scalar, Dim>::setProperty(unsigned int contact_index, unsigned int object_lhs_index, unsigned int object_rhs_index,
    const Vector<Scalar, Dim>& global_contact_position, const Vector<Scalar, Dim>& global_contact_normal_lhs)
{
    contact_index_ = contact_index;
    object_lhs_index_ = object_lhs_index;
    object_rhs_index_ = object_rhs_index;
    global_contact_position_ = global_contact_position;
    global_contact_normal_lhs_ = global_contact_normal_lhs;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::contactIndex() const
{
    return contact_index_;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::objectLhsIndex() const
{
    return object_lhs_index_;
}

template <typename Scalar,int Dim>
unsigned int ContactPoint<Scalar, Dim>::objectRhsIndex() const
{
    return object_rhs_index_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactPosition() const
{
    return global_contact_position_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactNormalLhs() const
{
    return global_contact_normal_lhs_;
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> ContactPoint<Scalar, Dim>::globalContactNormalRhs() const
{
    return -global_contact_normal_lhs_;
}

template <typename Scalar,int Dim>
ContactPointManager<Scalar, Dim>::ContactPointManager()
{

}

template <typename Scalar,int Dim>
ContactPointManager<Scalar, Dim>::~ContactPointManager()
{
    cleanContactPoints();
}

template <typename Scalar,int Dim>
void ContactPointManager<Scalar, Dim>::setCollisionResult(CollisionDetectionResult<Scalar, Dim>& collision_result)
{
    unsigned int num_collision = collision_result.numberCollision();
    CollisionPairBase<Scalar, Dim>* collision_pair;
    for(unsigned int i = 0; i < num_collision; ++i)
    {
        collision_pair = collision_result.collisionPairs()[i];
        if(collision_pair->objectTypeLhs() == CollidableObject<Scalar, Dim>::MESH_BASED && collision_pair->objectTypeRhs() == CollidableObject<Scalar, Dim>::MESH_BASED)
        {
            getMeshContactPoint(dynamic_cast<CollisionPairMeshToMesh<Scalar, Dim>*>(collision_pair));
        }
        else
        {
            std::cerr<<"Wrong object type of collision pair: should be MESH_BASED!"<<std::endl;
            return;
        }
    }
}

template <typename Scalar,int Dim>
unsigned int ContactPointManager<Scalar, Dim>::numContactPoint() const
{
    return static_cast<unsigned int>(contact_points_.size());
}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>* ContactPointManager<Scalar, Dim>::contactPoint(unsigned int contact_index)
{
    if(contact_index >= numContactPoint())
    {
        std::cerr<<"Contact index our of range!"<<std::endl;
        return NULL;
    }
    return contact_points_[contact_index];
}

template <typename Scalar,int Dim>
const std::vector<ContactPoint<Scalar, Dim>* >& ContactPointManager<Scalar, Dim>::contactPoints() const
{
    return contact_points_;
}

template <typename Scalar,int Dim>
std::vector<ContactPoint<Scalar, Dim>* >& ContactPointManager<Scalar, Dim>::contactPoints()
{
    return contact_points_;
}

template <typename Scalar,int Dim>
ContactPoint<Scalar, Dim>* ContactPointManager<Scalar, Dim>::operator[] (unsigned int contact_index)
{
    if(contact_index >= numContactPoint())
    {
        std::cerr<<"Contact index our of range!"<<std::endl;
        return NULL;
    }
    return contact_points_[contact_index];
}

template <typename Scalar,int Dim>
void ContactPointManager<Scalar, Dim>::cleanContactPoints()
{
    unsigned int num_contact = numContactPoint();
    for(unsigned int i = 0; i < num_contact; ++i)
    {
        delete contact_points_[i];
    }
    contact_points_.clear();
}

template <typename Scalar,int Dim>
void ContactPointManager<Scalar, Dim>::getMeshContactPoint(CollisionPairMeshToMesh<Scalar, Dim>* collision_pair)
{
    if(collision_pair == NULL)
    {
        std::cerr<<"Null collision pair!"<<std::endl;
        return;
    }
    
    Face<Scalar>* face_lhs = collision_pair->faceLhsPtr();
    Face<Scalar>* face_rhs = collision_pair->faceRhsPtr();
    unsigned int num_vertex_lhs = face_lhs->numVertices();
    unsigned int num_vertex_rhs = face_rhs->numVertices();
    Vector<Scalar, 3>* vertex_lhs = new Vector<Scalar, 3>[num_vertex_lhs];
    Vector<Scalar, 3>* vertex_rhs = new Vector<Scalar, 3>[num_vertex_rhs];

    for(unsigned int i = 0; i < num_vertex_lhs; i++)
    {
        vertex_lhs[i] = collision_pair->meshObjectLhs()->vertexPosition(face_lhs->vertex(i).positionIndex());
    }
    for(unsigned int i = 0; i < num_vertex_rhs; i++)
    {
        vertex_rhs[i] = collision_pair->meshObjectRhs()->vertexPosition(face_rhs->vertex(i).positionIndex());
    }

    unsigned int num_overlap = 0;
    bool is_lhs_tri = (num_vertex_lhs == 3);
    bool is_rhs_tri = (num_vertex_rhs == 3);
    bool is_lhs_quad = (num_vertex_lhs == 4);
    bool is_rhs_quad = (num_vertex_rhs == 4);
    Vector<Scalar, 3> overlap_point(0);
    Vector<Scalar, 3> temp_overlap_point(0);
    Vector<Scalar, 3> contact_normal_lhs(0);

    //test each edge of lhs with the face of rhs
    for(unsigned int i = 0; i < num_vertex_lhs; i++)
    {
        if(is_rhs_tri)//triangle
        {
            if(MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeTriangle(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2], temp_overlap_point))
            {
                num_overlap++;
                overlap_point += temp_overlap_point;
            }
        }
        if(is_rhs_quad)//quad
        {
            if(MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeQuad(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2], vertex_rhs[3], temp_overlap_point))
            {
                num_overlap++;
                overlap_point += temp_overlap_point;
            }
        }
    }

    //test each edge of rhs with the face of lhs
    for(unsigned int i = 0; i < num_vertex_rhs; i++)
    {
        if(is_lhs_tri)//triangle
        {
            if(MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeTriangle(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2], temp_overlap_point))
            {
                num_overlap++;
                overlap_point += temp_overlap_point;
            }
        }
        if(is_lhs_quad)//quad
        {
            if(MeshBasedCollidableObject<Scalar, Dim>::overlapEdgeQuad(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2], vertex_lhs[3], temp_overlap_point))
            {
                num_overlap++;
                overlap_point += temp_overlap_point;
            }
        }
    }

    if(num_overlap > 0)
    {
        overlap_point /= static_cast<Scalar>(num_overlap);
        contact_normal_lhs = static_cast<Scalar>(0.5) * (collision_pair->meshObjectLhs()->faceNormal(collision_pair->faceLhsIdx()) - 
                                                        collision_pair->meshObjectRhs()->faceNormal(collision_pair->faceRhsIdx()));
        if(contact_normal_lhs.norm() > std::numeric_limits<Scalar>::epsilon())
        {
            contact_normal_lhs.normalize();
            ContactPoint<Scalar, Dim>* contact_point = new ContactPoint<Scalar, Dim>(numContactPoint(), collision_pair->objectLhsIdx(), collision_pair->objectRhsIdx(),
                                                                                    overlap_point, contact_normal_lhs);
            contact_points_.push_back(contact_point);
        }
    }

    delete[] vertex_lhs;
    delete[] vertex_rhs;
}

template class ContactPoint<float, 3>;
template class ContactPoint<double, 3>;
template class ContactPointManager<float, 3>;
template class ContactPointManager<double, 3>;

}
