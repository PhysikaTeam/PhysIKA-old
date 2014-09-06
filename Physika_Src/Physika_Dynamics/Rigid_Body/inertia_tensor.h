/*
 * @file inertia_tensor.h 
 * @Compute the inertia_tensor of a mesh
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_INERTIA_TENSOR_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_INERTIA_TENSOR_H_

#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Quaternion/quaternion.h"

namespace Physika{

template <typename Scalar,int Dim> class Vector;
template <typename Scalar> class SurfaceMesh;

//InertiaTensor is only defined for 3-dimension.
//It can only compute inertia tensor for triangle meshes.
//Support for quad-mesh remains to be implemented.
template <typename Scalar>
class InertiaTensor
{
public:
    InertiaTensor();
    ~InertiaTensor();

    const SquareMatrix<Scalar, 3> bodyInertiaTensor() const;
    SquareMatrix<Scalar, 3> bodyInertiaTensor();
    const SquareMatrix<Scalar, 3> spatialInertiaTensor() const;
    SquareMatrix<Scalar, 3> spatialInertiaTensor();
    const SquareMatrix<Scalar, 3> bodyInertiaTensorInverse() const;
    SquareMatrix<Scalar, 3> bodyInertiaTensorInverse();
    const SquareMatrix<Scalar, 3> spatialInertiaTensorInverse() const;
    SquareMatrix<Scalar, 3> spatialInertiaTensorInverse();

    //set a body to this inertia tensor. Mesh, scale and density should be provided.
    //mass_center and mass will be modified after calling this function in order to get the center of mass and the value of mass.
    void setBody(SurfaceMesh<Scalar>* mesh, Vector<Scalar, 3> scale, Scalar density, Vector<Scalar, 3>& mass_center, Scalar& mass);

    //give the rotation of this body and get the inertia tensor after rotation.
    //spatial_inertia_tensor_ is modified in this function while body_inertia_tensor_ remains unchanged.
    SquareMatrix<Scalar, 3> rotate(Quaternion<Scalar>& quad);

    InertiaTensor<Scalar>& operator = (const InertiaTensor<Scalar>& inertia_tensor);

protected:

    //the inertia tensor of a body referring to its mass center
    //it remains unchanged after setBody()
    SquareMatrix<Scalar, 3> body_inertia_tensor_;
    SquareMatrix<Scalar, 3> body_inertia_tensor_inverse_;

    //the inertia tensor of a body in spatial frame. This is a common used inertia tensor in rigid body simulation
    //it will be modified after calling rotate(Quaternion<Scalar>& quad)
    SquareMatrix<Scalar, 3> spatial_inertia_tensor_;
    SquareMatrix<Scalar, 3> spatial_inertia_tensor_inverse_;

private:

    /* maximum number of verts per polygonal face */
    const static int MAX_POLYGON_SZ = 3;
    const static int X = 0;
    const static int Y = 1;
    const static int Z = 2;

    inline Scalar SQR(Scalar x){return x * x;};
    inline Scalar CUBE(Scalar x){return x * x * x;};

    struct InertiaTensorPolyhedron; //forward declaration
    struct InertiaTensorFace
    {
        unsigned int numVerts_;
        Scalar norm_[3];
        Scalar w_;
        unsigned int verts_[MAX_POLYGON_SZ];
        typename InertiaTensor<Scalar>::InertiaTensorPolyhedron *poly_;
    };

    struct InertiaTensorPolyhedron
    {
        unsigned int numVerts_;
        unsigned int numFaces_;
        Scalar** verts_;//numVerts x 3 dimension
        typename InertiaTensor<Scalar>::InertiaTensorFace* faces_;//numFaces dimension
    };

    int A;   /* alpha */
    int B;   /* beta */
    int C;   /* gamma */

    /* projection integrals */
    Scalar P1, Pa, Pb, Paa, Pab, Pbb, Paaa, Paab, Pabb, Pbbb;

    /* face integrals */
    Scalar Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca;

    /* volume integrals */
    Scalar T0, T1[3], T2[3], TP[3];

    /* compute various integrations over projection of face */
    void compProjectionIntegrals(InertiaTensorFace *f);
    void compFaceIntegrals(InertiaTensorFace *f);
    void compVolumeIntegrals(InertiaTensorPolyhedron *p);

};

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_INERTIA_TENSOR_H_
