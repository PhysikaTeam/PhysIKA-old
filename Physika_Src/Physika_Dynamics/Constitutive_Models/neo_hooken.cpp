/*
 * @file  neo_hooken.cpp
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

#include "Physika_Dynamics/Constitutive_Models/neo_hooken.h"

namespace Physika{

template <typename Scalar, int Dim>
NeoHooken<Scalar,Dim>::NeoHooken()
{
}

template <typename Scalar, int Dim>
NeoHooken<Scalar,Dim>::NeoHooken(Scalar lambda, Scalar mu)
    :lambda_(lambda),mu_(mu)
{
}

template <typename Scalar, int Dim>
NeoHooken<Scalar,Dim>::~NeoHooken()
{
}

template <typename Scalar, int Dim>
void NeoHooken<Scalar,Dim>::info() const
{
}

template <typename Scalar, int Dim>
Scalar NeoHooken<Scalar,Dim>::energyDensity(const MatrixBase &F) const
{
    return F.rows();
}

//explicit instantiation of template so that it could be compiled into a lib
template class NeoHooken<float,2>;
template class NeoHooken<double,2>;
template class NeoHooken<float,3>;
template class NeoHooken<double,3>;

} //end of namespace Physika
