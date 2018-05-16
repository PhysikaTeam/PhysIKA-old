/*
 * @file PDM_collision_method_base.h 
 * @brief base class of collision method for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_BASE_H

namespace Physika{

template <typename Scalar, int Dim> class PDMBase;

template <typename Scalar, int Dim>
class PDMCollisionMethodBase
{
public:
    PDMCollisionMethodBase();
    virtual ~PDMCollisionMethodBase();

    virtual void setDriver(PDMBase<Scalar, Dim> * driver);

    Scalar Kc() const;
    void setKc(Scalar Kc);

    virtual void collisionMethod() = 0;

protected:
    PDMBase<Scalar, Dim> * driver_;
    Scalar Kc_; //default:0.0, model parameter

};

} // end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_COLLISION_METHODS_PDM_COLLISION_METHOD_BASE_H