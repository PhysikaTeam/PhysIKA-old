/*
 * @file inertia_tensor.cpp 
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

#include "Physika_Dynamics/Rigid_Body/inertia_tensor.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
InertiaTensor<Scalar>::InertiaTensor()
{

}

template <typename Scalar>
InertiaTensor<Scalar>::~InertiaTensor()
{

}

template <typename Scalar>
const SquareMatrix<Scalar, 3> InertiaTensor<Scalar>::bodyInertiaTensor() const
{
    return body_inertia_tensor_;
}

template <typename Scalar>
SquareMatrix<Scalar, 3> InertiaTensor<Scalar>::bodyInertiaTensor()
{
    return body_inertia_tensor_;
}

template <typename Scalar>
const SquareMatrix<Scalar, 3> InertiaTensor<Scalar>::spatialInertiaTensor() const
{
    return spatial_inertia_tensor_;
}

template <typename Scalar>
SquareMatrix<Scalar, 3> InertiaTensor<Scalar>::spatialInertiaTensor()
{
    return spatial_inertia_tensor_;
}

template <typename Scalar>
SquareMatrix<Scalar, 3> InertiaTensor<Scalar>::rotate(Quaternion<Scalar>& quad)
{
    return spatial_inertia_tensor_;
}

template <typename Scalar>
void InertiaTensor<Scalar>::setBody(SurfaceMesh<Scalar>* mesh, Vector<Scalar, 3> scale, Scalar density, Vector<Scalar, 3>& mass_center, Scalar& mass)
{
    if(!mesh->isTriangularMesh())
    {
        std::cerr<<"Inertia tensor doesn't support quad mesh!"<<std::endl;
        return;
    }

    Scalar dx1, dy1, dz1, dx2, dy2, dz2, nx, ny, nz, len;
    InertiaTensor<Scalar>::InertiaTensorFace *f;
    InertiaTensor<Scalar>::InertiaTensorPolyhedron* p = new InertiaTensor<Scalar>::InertiaTensorPolyhedron;

    unsigned int _vtxNum = mesh->numVertices();
    unsigned int _triNum = mesh->numFaces();

    p->numVerts_ = _vtxNum;
    p->numFaces_ = _triNum;
    p->verts_ = new Scalar*[_vtxNum];
    p->faces_ = new InertiaTensorFace[_triNum];
    for(unsigned int i = 0; i < _vtxNum; i++)
        p->verts_[i] = new Scalar[3];

    for(unsigned int i = 0; i < _vtxNum; ++i)
    {
        p->verts_[i][0] = mesh->vertexPosition(i)[0] * scale[0];
        p->verts_[i][1] = mesh->vertexPosition(i)[1] * scale[1];
        p->verts_[i][2] = mesh->vertexPosition(i)[2] * scale[2];
    }

    for (unsigned int i = 0; i < p->numFaces_; ++i)
    {
        f = &((p->faces_)[i]);
        f->poly_ = p;
        f->numVerts_ = 3;
        for (int j = 0; j < MAX_POLYGON_SZ; j++)
            f->verts_[j] = mesh->face(i).vertex(j).positionIndex();

        /* compute face normal and offset w from first 3 vertices */
        dx1 = p->verts_[f->verts_[1]][X] - p->verts_[f->verts_[0]][X];
        dy1 = p->verts_[f->verts_[1]][Y] - p->verts_[f->verts_[0]][Y];
        dz1 = p->verts_[f->verts_[1]][Z] - p->verts_[f->verts_[0]][Z];
        dx2 = p->verts_[f->verts_[2]][X] - p->verts_[f->verts_[1]][X];
        dy2 = p->verts_[f->verts_[2]][Y] - p->verts_[f->verts_[1]][Y];
        dz2 = p->verts_[f->verts_[2]][Z] - p->verts_[f->verts_[1]][Z];
        nx = dy1 * dz2 - dy2 * dz1;
        ny = dz1 * dx2 - dz2 * dx1;
        nz = dx1 * dy2 - dx2 * dy1;
        len = sqrt(nx * nx + ny * ny + nz * nz);
        f->norm_[X] = nx / len;
        f->norm_[Y] = ny / len;
        f->norm_[Z] = nz / len;
        f->w_ = - f->norm_[X] * p->verts_[f->verts_[0]][X]
        - f->norm_[Y] * p->verts_[f->verts_[0]][Y]
        - f->norm_[Z] * p->verts_[f->verts_[0]][Z];

    }

    compVolumeIntegrals(p);

    mass = density * T0;

    /* compute center of mass */
    mass_center[X] = T1[X] / T0;
    mass_center[Y] = T1[Y] / T0;
    mass_center[Z] = T1[Z] / T0;

    /* compute inertia tensor */
    body_inertia_tensor_(X, X) = density * (T2[Y] + T2[Z]);
    body_inertia_tensor_(Y, Y) = density * (T2[Z] + T2[X]);
    body_inertia_tensor_(Z, Z) = density * (T2[X] + T2[Y]);
    body_inertia_tensor_(X, Y) = body_inertia_tensor_(Y, X) = - density * TP[X];
    body_inertia_tensor_(Y, Z) = body_inertia_tensor_(Z, Y) = - density * TP[Y];
    body_inertia_tensor_(Z, X) = body_inertia_tensor_(X, Z) = - density * TP[Z];

    /* translate inertia tensor to center of mass */
    body_inertia_tensor_(X, X) -= mass * (mass_center[Y] * mass_center[Y] + mass_center[Z] * mass_center[Z]);
    body_inertia_tensor_(Y, Y) -= mass * (mass_center[Z] * mass_center[Z] + mass_center[X] * mass_center[X]);
    body_inertia_tensor_(Z, Z) -= mass * (mass_center[X] * mass_center[X] + mass_center[Y] * mass_center[Y]);
    body_inertia_tensor_(X, Y) = body_inertia_tensor_(Y, X) += mass * mass_center[X] * mass_center[Y]; 
    body_inertia_tensor_(Y, Z) = body_inertia_tensor_(Z, Y) += mass * mass_center[Y] * mass_center[Z]; 
    body_inertia_tensor_(Z, X) = body_inertia_tensor_(X, Z) += mass * mass_center[Z] * mass_center[X]; 

    for(unsigned int i = 0; i < _vtxNum; ++i)
        delete [] p->verts_[i];
    delete [] p->verts_;
    delete [] p->faces_;

}

template <typename Scalar>
void InertiaTensor<Scalar>::compProjectionIntegrals(InertiaTensorFace *f)
{
    Scalar a0, a1, da;
    Scalar b0, b1, db;
    Scalar a0_2, a0_3, a0_4, b0_2, b0_3, b0_4;
    Scalar a1_2, a1_3, b1_2, b1_3;
    Scalar C1, Ca, Caa, Caaa, Cb, Cbb, Cbbb;
    Scalar Cab, Kab, Caab, Kaab, Cabb, Kabb;

    P1 = Pa = Pb = Paa = Pab = Pbb = Paaa = Paab = Pabb = Pbbb = 0.0;

    for (unsigned int i = 0; i < f->numVerts_; ++i)
    {
        a0 = f->poly_->verts_[f->verts_[i]][A];
        b0 = f->poly_->verts_[f->verts_[i]][B];
        a1 = f->poly_->verts_[f->verts_[(i+1) % f->numVerts_]][A];
        b1 = f->poly_->verts_[f->verts_[(i+1) % f->numVerts_]][B];
        da = a1 - a0;
        db = b1 - b0;
        a0_2 = a0 * a0; a0_3 = a0_2 * a0; a0_4 = a0_3 * a0;
        b0_2 = b0 * b0; b0_3 = b0_2 * b0; b0_4 = b0_3 * b0;
        a1_2 = a1 * a1; a1_3 = a1_2 * a1; 
        b1_2 = b1 * b1; b1_3 = b1_2 * b1;

        C1 = a1 + a0;
        Ca = a1*C1 + a0_2; Caa = a1*Ca + a0_3; Caaa = a1*Caa + a0_4;
        Cb = b1*(b1 + b0) + b0_2; Cbb = b1*Cb + b0_3; Cbbb = b1*Cbb + b0_4;
        Cab = 3*a1_2 + 2*a1*a0 + a0_2; Kab = a1_2 + 2*a1*a0 + 3*a0_2;
        Caab = a0*Cab + 4*a1_3; Kaab = a1*Kab + 4*a0_3;
        Cabb = 4*b1_3 + 3*b1_2*b0 + 2*b1*b0_2 + b0_3;
        Kabb = b1_3 + 2*b1_2*b0 + 3*b1*b0_2 + 4*b0_3;

        P1 += db*C1;
        Pa += db*Ca;
        Paa += db*Caa;
        Paaa += db*Caaa;
        Pb += da*Cb;
        Pbb += da*Cbb;
        Pbbb += da*Cbbb;
        Pab += db*(b1*Cab + b0*Kab);
        Paab += db*(b1*Caab + b0*Kaab);
        Pabb += da*(a1*Cabb + a0*Kabb);
    }

    P1 /= 2.0;
    Pa /= 6.0;
    Paa /= 12.0;
    Paaa /= 20.0;
    Pb /= -6.0;
    Pbb /= -12.0;
    Pbbb /= -20.0;
    Pab /= 24.0;
    Paab /= 60.0;
    Pabb /= -60.0;
}

template <typename Scalar>
void InertiaTensor<Scalar>::compFaceIntegrals(InertiaTensorFace *f)
{
    Scalar *n, w;
    Scalar k1, k2, k3, k4;

    compProjectionIntegrals(f);

    w = f->w_;
    n = f->norm_;
    k1 = 1 / n[C]; k2 = k1 * k1; k3 = k2 * k1; k4 = k3 * k1;

    Fa = k1 * Pa;
    Fb = k1 * Pb;
    Fc = -k2 * (n[A]*Pa + n[B]*Pb + w*P1);

    Faa = k1 * Paa;
    Fbb = k1 * Pbb;
    Fcc = k3 * (SQR(n[A])*Paa + 2*n[A]*n[B]*Pab + SQR(n[B])*Pbb
        + w*(2*(n[A]*Pa + n[B]*Pb) + w*P1));

    Faaa = k1 * Paaa;
    Fbbb = k1 * Pbbb;
    Fccc = -k4 * (CUBE(n[A])*Paaa + 3*SQR(n[A])*n[B]*Paab 
        + 3*n[A]*SQR(n[B])*Pabb + CUBE(n[B])*Pbbb
        + 3*w*(SQR(n[A])*Paa + 2*n[A]*n[B]*Pab + SQR(n[B])*Pbb)
        + w*w*(3*(n[A]*Pa + n[B]*Pb) + w*P1));

    Faab = k1 * Paab;
    Fbbc = -k2 * (n[A]*Pabb + n[B]*Pbbb + w*Pbb);
    Fcca = k3 * (SQR(n[A])*Paaa + 2*n[A]*n[B]*Paab + SQR(n[B])*Pabb
        + w*(2*(n[A]*Paa + n[B]*Pab) + w*Pa));
}

template <typename Scalar>
void InertiaTensor<Scalar>::compVolumeIntegrals(InertiaTensorPolyhedron *p)
{
    InertiaTensorFace *f;
    Scalar nx, ny, nz;

    T0 = T1[X] = T1[Y] = T1[Z] 
    = T2[X] = T2[Y] = T2[Z] 
    = TP[X] = TP[Y] = TP[Z] = 0;

    for (unsigned int i = 0; i < p->numFaces_; ++i)
    {

        f = &p->faces_[i];

        nx = fabs(f->norm_[X]);
        ny = fabs(f->norm_[Y]);
        nz = fabs(f->norm_[Z]);
        if (nx > ny && nx > nz) C = X;
        else C = (ny > nz) ? Y : Z;
        A = (C + 1) % 3;
        B = (A + 1) % 3;

        compFaceIntegrals(f);

        T0 += f->norm_[X] * ((A == X) ? Fa : ((B == X) ? Fb : Fc));

        T1[A] += f->norm_[A] * Faa;
        T1[B] += f->norm_[B] * Fbb;
        T1[C] += f->norm_[C] * Fcc;
        T2[A] += f->norm_[A] * Faaa;
        T2[B] += f->norm_[B] * Fbbb;
        T2[C] += f->norm_[C] * Fccc;
        TP[A] += f->norm_[A] * Faab;
        TP[B] += f->norm_[B] * Fbbc;
        TP[C] += f->norm_[C] * Fcca;
    }

    T1[X] /= 2; T1[Y] /= 2; T1[Z] /= 2;
    T2[X] /= 3; T2[Y] /= 3; T2[Z] /= 3;
    TP[X] /= 2; TP[Y] /= 2; TP[Z] /= 2;
}

template class InertiaTensor<float>;
template class InertiaTensor<double>;

}