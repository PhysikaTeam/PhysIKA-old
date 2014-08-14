/*
 * @file CPDI_update_method.h 
 * @Brief the particle domain update procedure introduced in paper:
 *        "A convected particle domain interpolation technique to extend applicability of
 *         the material point method for problems involving massive deformations"
 *        It's the base class of all update methods derived from CPDI
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_

namespace Physika{

template <typename Scalar, int Dim> class CPDIMPMSolid;

template <typename Scalar, int Dim>
class CPDIUpdateMethod
{
public:
    CPDIUpdateMethod();
    virtual ~CPDIUpdateMethod();
    virtual void updateParticleDomain();
    void setCPDIDriver(CPDIMPMSolid<Scalar,Dim> *cpdi_driver);
protected:
    CPDIMPMSolid<Scalar,Dim> *cpdi_driver_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_
