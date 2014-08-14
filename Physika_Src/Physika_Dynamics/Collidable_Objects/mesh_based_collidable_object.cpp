/*
 * @file  mesh_based_collidable_object.cpp
 * @collidable object based on the mesh of object
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

#include "Physika_Core/Transform/transform_3d.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair_manager.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;

template <typename Scalar>
MeshBasedCollidableObject<Scalar>::MeshBasedCollidableObject():
	mesh_(NULL),
	transform_(NULL)
{
}

template <typename Scalar>
MeshBasedCollidableObject<Scalar>::~MeshBasedCollidableObject()
{
}

template <typename Scalar>
typename CollidableObjectInternal::ObjectType MeshBasedCollidableObject<Scalar>::objectType() const
{
	return CollidableObjectInternal::MESH_BASED;
}

template <typename Scalar>
const SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar>::mesh() const
{
	return mesh_;
}

template <typename Scalar>
SurfaceMesh<Scalar>* MeshBasedCollidableObject<Scalar>::mesh()
{
	return mesh_;
}


template <typename Scalar>
void MeshBasedCollidableObject<Scalar>::setMesh(SurfaceMesh<Scalar>* mesh)
{
	mesh_ = mesh;
}

template <typename Scalar>
Vector<Scalar, 3> MeshBasedCollidableObject<Scalar>::vertexPosition(unsigned int vertex_index) const
{
	if(transform_ != NULL)
		return transform_->transform(mesh_->vertexPosition(vertex_index));
	else
		return mesh_->vertexPosition(vertex_index);
}

template <typename Scalar>
Vector<Scalar, 3> MeshBasedCollidableObject<Scalar>::faceNormal(unsigned int face_index) const
{
    Face<Scalar>& face = mesh_->face(face_index);
    PHYSIKA_ASSERT(face.numVertices()>=3);
    unsigned int vert_idx1 = face.vertex(0).positionIndex();
    unsigned int vert_idx2 = face.vertex(1).positionIndex();
    unsigned int vert_idx3 =  face.vertex(2).positionIndex();
    const Vector<Scalar,3> &vert_pos1 = vertexPosition(vert_idx1);
    const Vector<Scalar,3> &vert_pos2 = vertexPosition(vert_idx2);
    const Vector<Scalar,3> &vert_pos3 = vertexPosition(vert_idx3);
    Vector<Scalar,3> vec1 = vert_pos2 - vert_pos1;
    Vector<Scalar,3> vec2 = vert_pos3 - vert_pos1;
    Vector<Scalar,3> normal = (vec1.cross(vec2)).normalize();
    return normal;
}

template <typename Scalar>
const Transform<Scalar, 3>* MeshBasedCollidableObject<Scalar>::transform() const
{
	return transform_;
}

template <typename Scalar>
Transform<Scalar, 3>* MeshBasedCollidableObject<Scalar>::transform()
{
	return transform_;
}

template <typename Scalar>
void MeshBasedCollidableObject<Scalar>::setTransform(Transform<Scalar, 3>* transform)
{
	transform_ = transform;
}

template <typename Scalar>
bool MeshBasedCollidableObject<Scalar>::collideWithPoint(Vector<Scalar, 3> *point, Vector<Scalar, 3> &contact_normal)
{
    //to do
    return false;
}

template <typename Scalar>
bool MeshBasedCollidableObject<Scalar>::collideWithMesh(MeshBasedCollidableObject<Scalar>* object, unsigned int face_index_lhs, unsigned int face_index_rhs)
{
	if(object == NULL || object->mesh() == NULL)
		return false;

	Face<Scalar>& face_lhs = mesh_->face(face_index_lhs);
	Face<Scalar>& face_rhs = object->mesh()->face(face_index_rhs);
	unsigned int num_vertex_lhs = face_lhs.numVertices();
	unsigned int num_vertex_rhs = face_rhs.numVertices();
	Vector<Scalar, 3>* vertex_lhs = new Vector<Scalar, 3>[num_vertex_lhs];
	Vector<Scalar, 3>* vertex_rhs = new Vector<Scalar, 3>[num_vertex_rhs];

	for(unsigned int i = 0; i < num_vertex_lhs; i++)
	{
		vertex_lhs[i] = vertexPosition(face_lhs.vertex(i).positionIndex());
	}
	for(unsigned int i = 0; i < num_vertex_rhs; i++)
	{
		vertex_rhs[i] = object->vertexPosition(face_rhs.vertex(i).positionIndex());
	}

	bool is_overlap = false;
	bool is_lhs_tri = (num_vertex_lhs == 3);
	bool is_rhs_tri = (num_vertex_rhs == 3);
	bool is_lhs_quad = (num_vertex_lhs == 4);
	bool is_rhs_quad = (num_vertex_rhs == 4);
    Vector<Scalar, 3> overlap_point;

	//test each edge of lhs with the face of rhs
	for(unsigned int i = 0; i < num_vertex_lhs; i++)
	{
		if(is_rhs_tri)//triangle
		{
			if(overlapEdgeTriangle(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2], overlap_point))
            {
				is_overlap = true;
                break;
            }
		}
		if(is_rhs_quad)//quad
		{
			if(overlapEdgeQuad(vertex_lhs[i], vertex_lhs[(i + 1)%num_vertex_lhs], vertex_rhs[0], vertex_rhs[1], vertex_rhs[2], vertex_rhs[3], overlap_point))
            {
				is_overlap = true;
                break;
            }
		}
	}

    if(is_overlap)
    {
        delete[] vertex_lhs;
        delete[] vertex_rhs;
        return is_overlap;
    }

	//test each edge of rhs with the face of lhs
	for(unsigned int i = 0; i < num_vertex_rhs; i++)
	{
		if(is_lhs_tri)//triangle
		{
			if(overlapEdgeTriangle(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2], overlap_point))
            {
				is_overlap = true;
                break;
            }
		}
		if(is_lhs_quad)//quad
		{
			if(overlapEdgeQuad(vertex_rhs[i], vertex_rhs[(i + 1)%num_vertex_rhs], vertex_lhs[0], vertex_lhs[1], vertex_lhs[2], vertex_lhs[3], overlap_point))
            {
				is_overlap = true;
                break;
            }
		}
	}

	delete[] vertex_lhs;
	delete[] vertex_rhs;
	return is_overlap;
}

template <typename Scalar>
bool MeshBasedCollidableObject<Scalar>::overlapEdgeTriangle(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c, Vector<Scalar, 3>& overlap_point)
{
	Vector<Scalar, 3> face_normal = (vertex_face_b - vertex_face_a).cross(vertex_face_c - vertex_face_a);
	face_normal.normalize();
	Scalar length = face_normal.dot(vertex_edge_b - vertex_edge_a);
	if(abs(length) < FLOAT_EPSILON)
		return false;
	Scalar ratio = face_normal.dot(vertex_face_a - vertex_edge_a)/length;
	if(ratio < 0 || ratio > 1)
		return false;
	Vector<Scalar, 3> projection = vertex_edge_a + (vertex_edge_b - vertex_edge_a) * ratio;
	bool inTriTestBA = (((vertex_face_b - vertex_face_a).cross(projection - vertex_face_a)).dot(face_normal) >= 0);
	bool inTriTestCB = (((vertex_face_c - vertex_face_b).cross(projection - vertex_face_b)).dot(face_normal) >= 0);
	bool inTriTestAC = (((vertex_face_a - vertex_face_c).cross(projection - vertex_face_c)).dot(face_normal) >= 0);
	if(inTriTestBA && inTriTestCB && inTriTestAC)
    {
        overlap_point = projection;
		return true;
    }
	return false;
}

template <typename Scalar>
bool MeshBasedCollidableObject<Scalar>::overlapEdgeQuad(const Vector<Scalar, 3>& vertex_edge_a, const Vector<Scalar, 3>& vertex_edge_b, const Vector<Scalar, 3>& vertex_face_a, const Vector<Scalar, 3>& vertex_face_b, const Vector<Scalar, 3>& vertex_face_c, const Vector<Scalar, 3>& vertex_face_d, Vector<Scalar, 3>& overlap_point)
{
    //to do
	return false;
}

//explicit instantitation
template class MeshBasedCollidableObject<float>;
template class MeshBasedCollidableObject<double>;

}
