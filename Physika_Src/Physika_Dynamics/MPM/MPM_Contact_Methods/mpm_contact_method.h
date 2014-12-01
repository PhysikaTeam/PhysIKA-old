/*
 * @file mpm_contact_method.h 
 * @Brief base class of all mpm contact methods. The contact methods are alternatives
 *        to the default action of MPM which resolves contact automatically via the backgroud
 *        grid
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_CONTACT_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_CONTACT_METHOD_H_

namespace Physika{

template <typename Scalar, int Dim> class MPMBase;
    
template <typename Scalar, int Dim>
class MPMContactMethod
{
public:
    MPMContactMethod();
    virtual ~MPMContactMethod() = 0;
    void setMPMDriver(MPMBase<Scalar,Dim> *mpm_driver);
protected:
    MPMBase<Scalar,Dim> *mpm_driver_; 
};
    
}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_CONTACT_METHODS_MPM_CONTACT_METHOD_H_
