/*
 * @file basic_geometry.cpp
 * @brief base class of all basic geometry like plane, sphere, etc. 
 * @author FeiZhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Basic_Geometry/basic_geometry.h"

namespace Physika{

template <typename Scalar, int Dim>
BasicGeometry<Scalar,Dim>::BasicGeometry()
{
}

template <typename Scalar, int Dim>
BasicGeometry<Scalar,Dim>::BasicGeometry(const BasicGeometry<Scalar,Dim> &geometry)
{
}

template <typename Scalar, int Dim>
BasicGeometry<Scalar,Dim>::~BasicGeometry()
{
}

template <typename Scalar, int Dim>
BasicGeometry<Scalar,Dim>& BasicGeometry<Scalar,Dim>::operator= (const BasicGeometry<Scalar,Dim> &geometry)
{
    return *this;
}

//explicit instantiations
template class BasicGeometry<float,2>;
template class BasicGeometry<float,3>;
template class BasicGeometry<double,2>;
template class BasicGeometry<double,3>;

}  //end of namespace Physika
