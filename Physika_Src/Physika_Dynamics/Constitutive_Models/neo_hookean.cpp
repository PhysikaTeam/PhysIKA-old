/*
 * @file  neo_hookean.cpp
 * @brief Neo-Hookean hyperelastic material model
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

#include <cmath>
#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/neo_hookean.h"

namespace Physika{

template <typename Scalar, int Dim>
NeoHookean<Scalar,Dim>::NeoHookean()
{
}

template <typename Scalar, int Dim>
NeoHookean<Scalar,Dim>::NeoHookean(Scalar par1, Scalar par2, ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type)
{
}

template <typename Scalar, int Dim>
NeoHookean<Scalar,Dim>::~NeoHookean()
{
}

template <typename Scalar, int Dim>
void NeoHookean<Scalar,Dim>::info() const
{
    std::cout<<"Compressible Neo-Hookean material:"<<std::endl;
    std::cout<<"Energy density: Psi = mu/2*(trace(C)-Dim)-mu*lnJ+lambda/2*(lnJ)^2"<<std::endl;
    std::cout<<"Where: C = transpose(F)*F, J = det(F)."<<std::endl;
}

template <typename Scalar, int Dim>
Scalar NeoHookean<Scalar,Dim>::energy(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar trace_c = (F.transpose()*F).trace();
    Scalar J = F.determinant();
    Scalar lnJ = log(J);
    Scalar mu = this->mu_;
    Scalar lambda = this->lambda_;
    Scalar energy = mu/2*(trace_c-Dim)-mu*lnJ+lambda/2*lnJ*lnJ;
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> NeoHookean<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> P = F*secondPiolaKirchhoffStress(F);
    return P;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> NeoHookean<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix();
    SquareMatrix<Scalar,Dim> inverse_c = (F.transpose()*F).inverse();
    Scalar lnJ = log(F.determinant());
    Scalar mu = this->mu_;
    Scalar lambda = this->lambda_;
    SquareMatrix<Scalar,Dim> S = mu*(identity-inverse_c)+lambda*lnJ*inverse_c;
    return S;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> NeoHookean<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar J = F.determinant();
    SquareMatrix<Scalar,Dim> stress = 1/J*firstPiolaKirchhoffStress(F)*F.transpose();
    return stress;
}

//explicit instantiation of template so that it could be compiled into a lib
template class NeoHookean<float,2>;
template class NeoHookean<double,2>;
template class NeoHookean<float,3>;
template class NeoHookean<double,3>;

} //end of namespace Physika
