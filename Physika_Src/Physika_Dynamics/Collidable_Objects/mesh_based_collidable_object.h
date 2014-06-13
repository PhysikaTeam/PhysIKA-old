/*
 * @file  mesh_based_collidable_object.h
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

#ifndef PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_
#define PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_

#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar> class SurfaceMesh;

template <typename Scalar,int Dim>
class MeshBasedCollidableObject: public CollidableObject<Scalar,Dim>
{
public:
	//constructors && deconstructors
	MeshBasedCollidableObject();
	~MeshBasedCollidableObject();

	//get & set
	typename CollidableObject<Scalar,Dim>::ObjectType getObjectType() const;
	const SurfaceMesh<Scalar>* getMesh() const;
	SurfaceMesh<Scalar>* getMesh();
	void setMesh(SurfaceMesh<Scalar>* mesh);
	Vector<Scalar, 3> getPointPosition(unsigned int point_index);
	
protected:
	SurfaceMesh<Scalar>* mesh_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_COLLIDABLE_OBJECTS_MESH_BASED_COLLIDABLE_OBJECT_H_
















