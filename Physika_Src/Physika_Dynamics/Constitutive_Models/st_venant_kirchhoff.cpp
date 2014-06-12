/*
 * @file  st_venant_kirchhoff.cpp
 * @brief St.Venant-Kirchhoff hyperelastic material model
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

#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/st_venant_kirchhoff.h"

namespace Physika{

template <typename Scalar, int Dim>
StVK<Scalar,Dim>::StVK()
{
}

template <typename Scalar, int Dim>
StVK<Scalar,Dim>::StVK(Scalar par1, Scalar par2, typename IsotropicHyperelasticMaterial<Scalar,Dim>::ModulusType par_type)
    :IsotropicHyperelasticMaterial<Scalar,Dim>(par1,par2,par_type)
{
}

template <typename Scalar, int Dim>
StVK<Scalar,Dim>::~StVK()
{
}

template <typename Scalar, int Dim>
void StVK<Scalar,Dim>::printInfo() const
{
    std::cout<<"St.Venant-Kirchhoff material:"<<std::endl;
    std::cout<<"Energy density: Psi = 1/2*lambda*(trace(E)^2)+mu*E:E"<<std::endl;
    std::cout<<"Where: Green Strain E = (transpose(F)*F-I)/2, operator ':' is double contraction."<<std::endl;
}

template <typename Scalar, int Dim>
Scalar StVK<Scalar,Dim>::energy(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix();
    SquareMatrix<Scalar,Dim> E = (F.transpose()*F-identity)/2;
    Scalar trace_E = E.trace();
    Scalar mu = this->mu_;
    Scalar lambda = this->lambda_;
    Scalar energy = lambda/2*trace_E*trace_E+mu*E.doubleContraction(E);
    return energy;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StVK<Scalar,Dim>::firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> P = F*secondPiolaKirchhoffStress(F);
    return P;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StVK<Scalar,Dim>::secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const
{
    SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix();
    SquareMatrix<Scalar,Dim> E = (F.transpose()*F-identity)/2;
    Scalar trace_E = E.trace();
    Scalar mu = this->mu_;
    Scalar lambda = this->lambda_;
    SquareMatrix<Scalar,Dim> S = lambda*trace_E*identity+2*mu*E;
    return S;
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> StVK<Scalar,Dim>::cauchyStress(const SquareMatrix<Scalar,Dim> &F) const
{
    Scalar J = F.determinant();
    SquareMatrix<Scalar,Dim> stress = 1/J*firstPiolaKirchhoffStress(F)*F.transpose();
    return stress;
}

//explicit instantiation of template so that it could be compiled into a lib
template class StVK<float,2>;
template class StVK<double,2>;
template class StVK<float,3>;
template class StVK<double,3>;

}  //end of namespace Physika
