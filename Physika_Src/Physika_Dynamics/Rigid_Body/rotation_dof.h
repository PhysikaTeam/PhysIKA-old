/*
 * @file compressed_jacobian_matrix.h 
 * @The compressed Jacobian matrix used in rigid body simulation
 * refer to "Iterative Dynamics with Temporal Coherence", Catto et al. 2005
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_ROTATION_DOF_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_ROTATION_DOF_H_

namespace Physika{

template<int Dim>
class RotationDof
{
public:
    enum {degree = 2 * Dim - 3};
};

}
#endif//PHYSIKA_DYNAMICS_RIGID_BODY_ROTATION_DOF_H_