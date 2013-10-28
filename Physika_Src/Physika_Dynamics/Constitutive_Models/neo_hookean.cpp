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
NeoHookean<Scalar,Dim>::NeoHookean(Scalar lambda, Scalar mu)
    :lambda_(lambda),mu_(mu)
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
    std::cout<<"Energy density: Psi = mu/2*(trace(C)-3)-mu*lnJ+lambda/2*(lnJ)^2"<<std::endl;
    std::cout<<"Where: C = transpose(F)*F, J = det(F)."<<std::endl;
}

template <typename Scalar, int Dim>
Scalar NeoHookean<Scalar,Dim>::energy(const SquareMatrix<Scalar,Dim> &F) const
{
    // Scalar trace_c = ((F.derived()).transpose()*F.derived()).trace();
    // Scalar J = (F.derived()).determinant();
    // Scalar lnJ = log(J);
    // Scalar energy = mu_/2*(trace_c-3)-mu_*lnJ+lambda/2*lnJ*lnJ;
    // return energy;
    return 0;
}

template <typename Scalar, int Dim>
void NeoHookean<Scalar,Dim>::energyGradient(const SquareMatrix<Scalar,Dim> &F, SquareMatrix<Scalar,Dim> &energy_gradient) const
{
}

template <typename Scalar, int Dim>
void NeoHookean<Scalar,Dim>::energyHessian(const SquareMatrix<Scalar,Dim> &F, SquareMatrix<Scalar,Dim> &energy_hessian) const
{
}

//explicit instantiation of template so that it could be compiled into a lib
template class NeoHookean<float,2>;
template class NeoHookean<double,2>;
template class NeoHookean<float,3>;
template class NeoHookean<double,3>;

} //end of namespace Physika
