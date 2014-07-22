/*
 * @file fem_solid.h 
 * @Brief FEM driver for solids, not necessarily homogeneous.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_

#include <vector>
#include <string>
#include "Physika_Dynamics/FEM/fem_base.h"

namespace Physika{

template <typename Scalar, int Dim> class ConstitutiveModel;

/*
 * FEM driver for solids, not necessarily homogeneous:
 * 1. All elements share one constitutive model if only one is provided. The solid
 *    is homogeneous in this case.
 * 2. Elements in the same region of simulation mesh share one constitutive model. In this
 *    case the number of constitutive models equals the number of regions of the simulation mesh.
 * 3. Element-wise consitutive model is used.
 *
 */

template <typename Scalar, int Dim>
class FEMSolid: public FEMBase<Scalar,Dim>
{
public:
    FEMSolid();
    FEMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    ~FEMSolid();

    //virtual methods
    void initConfiguration(const std::string &file_name);
    void advanceStep(Scalar dt);
    bool withRestartSupport() const;
    void write(const std::string &file_name);
    void read(const std::string &file_name);
    void addPlugin(DriverPluginBase<Scalar> *plugin);

    //set&&get constitutive model (data are copied)
    unsigned int materialNum() const;
    void setHomogeneousMaterial(const ConstitutiveModel<Scalar,Dim> &material);  
    void setRegionWiseMaterial(const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials);  //the number of materials must be no less than the number of regions on simulation mesh
    void setElementWiseMaterial(const std::vector<ConstitutiveModel<Scalar,Dim>*> &materials);  //the number of materials must be no less than the number of simulation elements
    const ConstitutiveModel<Scalar,Dim>* elementMaterial(unsigned int ele_idx) const;  //return the material of specific simulation element, return NULL if not set
    ConstitutiveModel<Scalar,Dim>* elementMaterial(unsigned int ele_idx);
protected:
    void initialize();
    void addMaterial(const ConstitutiveModel<Scalar,Dim> &material);
protected:
    std::vector<ConstitutiveModel<Scalar,Dim> *> constitutive_model_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_SOLID_H_
