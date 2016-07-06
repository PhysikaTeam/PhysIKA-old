/*
 * @file  bounding_volume.cpp
 * @bounding volume of a collidable object
 * @author Tianxiang Zhang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Geometry/Bounding_Volumes/bounding_volume.h"
#include "Physika_Geometry/Bounding_Volumes/bounding_volume_kdop18.h"
#include "Physika_Geometry/Bounding_Volumes/bounding_volume_octagon.h"
#include "Physika_Geometry/Bounding_Volumes/bounding_volume_axis_aligned_box.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

namespace BoundingVolumeInternal{

template<typename Scalar, int Dim>
BoundingVolume<Scalar, Dim>* createBoundingVolume(BVType bv_type)
{
    BoundingVolume<Scalar, Dim>* bounding_volume = NULL;
    switch(bv_type)
    {
    case KDOP18:
        bounding_volume = dynamic_cast<BoundingVolume<Scalar, Dim>* >(new BoundingVolumeKDOP18<Scalar>());
        break;
    case OCTAGON: 
        bounding_volume = dynamic_cast<BoundingVolume<Scalar, Dim>* >(new BoundingVolumeOctagon<Scalar>());
        break;
	case AXIS_ALIGNED_BOX:
        bounding_volume = dynamic_cast<BoundingVolume<Scalar, Dim>* >(new BoundingVolumeAxisAlignedBox<Scalar>());
        break;
    default: 
        std::cerr<<"Wrong bounding volume type!"<<std::endl;
        bounding_volume = NULL;
        break;
    }
    return bounding_volume;
}

template BoundingVolume<float, 2>* createBoundingVolume(BVType bv_type);
template BoundingVolume<float, 3>* createBoundingVolume(BVType bv_type);
template BoundingVolume<double, 2>* createBoundingVolume(BVType bv_type);
template BoundingVolume<double, 3>* createBoundingVolume(BVType bv_type);

}

template <typename Scalar, int Dim>
BoundingVolume<Scalar, Dim>::BoundingVolume()
{
}

template <typename Scalar, int Dim>
BoundingVolume<Scalar, Dim>::~BoundingVolume()
{
}

//explicit instantitation
template class BoundingVolume<float, 2>;
template class BoundingVolume<double, 2>;
template class BoundingVolume<float, 3>;
template class BoundingVolume<double, 3>;

}