/*
 * @file sph_solid.cpp 
 * @Basic SPH_solid class, basic deformation simulation uses sph.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/SPH/sph_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
SPHSolid<Scalar, Dim>::SPHSolid()
{

}

template <typename Scalar, int Dim>
SPHSolid<Scalar, Dim>::~SPHSolid()
{

}

template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::initialize()
{

}

template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::initSceneBoundary()
{

}

template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::advance(Scalar dt)
{



}
template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::stepEuler(Scalar dt)
{

}

template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::computeNeighbors()
{

}


template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::computeDensity()
{

}

template <typename Scalar, int Dim>
void SPHSolid<Scalar, Dim>::computeVolume()
{

}


//template class SPHSolid<float, 3>;
//template class SPHSolid<double, 3>;
} //end of namespace Physika
