/*
 * @file rigid_response_method_BLCP.h 
 * @Rigid-body collision response using BLCP in
 * "Mass Splitting for Jitter-Free Parallel Rigid Body Simulation"
 * Tonge et al. 2012
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_BLCP_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_BLCP_H_

#include "Physika_Dynamics/Rigid_Body/rigid_response_method.h"

namespace Physika{

template <typename Scalar,int Dim>
class RigidResponseMethodBLCP : public RigidResponseMethod<Scalar, Dim>
{
public:
    //constructor
    RigidResponseMethodBLCP();
    virtual ~RigidResponseMethodBLCP();

    //dynamic function used in a driver
    void collisionResponse();

protected:
};

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_RESPONSE_METHOD_BLCP_H_