/*
 * @file fem_solid_force_model.h 
 * @Brief the "engine" for fem solid drivers.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_

namespace Physika{

/*
 * the "engine" for fem solid drivers, do the actual force computation 
 * the drivers pass data of simulation to the force model, and the force
 * model return force, force differential, etc. to the drivers
 *
 * Note: with this design we're able to integrate fem computation for different element types 
 *         into one driver class without messing up the code
 */

template <typename Scalar, int Dim> class FEMSolid;

template <typename Scalar, int Dim>
class FEMSolidForceModel
{
public:
    FEMSolidForceModel();
    explicit FEMSolidForceModel(const FEMSolid<Scalar,Dim> *fem_solid_driver);
    virtual ~FEMSolidForceModel() = 0;
    virtual const FEMSolid<Scalar,Dim>* driver() const;
    virtual void setDriver(const FEMSolid<Scalar,Dim> *fem_solid_driver);
protected:
    const FEMSolid<Scalar,Dim> *fem_solid_driver_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_FORCE_MODEL_FEM_SOLID_FORCE_MODEL_H_
