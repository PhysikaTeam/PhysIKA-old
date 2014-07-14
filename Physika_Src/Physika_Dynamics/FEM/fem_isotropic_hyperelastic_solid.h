/*
 * @file fem_isotropic_hyperelastic_solid.h 
 * @Brief FEM driver for isotropic hyperelastic solids, not necessarilly homogeneous.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_ISOTROPIC_HYPERELASTIC_SOLID_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_ISOTROPIC_HYPERELASTIC_SOLID_H_

#include <vector>
#include "Physika_Dynamics/FEM/fem_base.h"

namespace Physika{

template <typename Scalar, int Dim> class IsotropicHyperelasticMaterial;

/*
 * FEM driver for isotropic hyperelastic solids, not necessarilly homogeneous:
 * 1. All elements share one constitutive model if only one is provided. The solid
 *    is homogeneous in this case.
 * 2. Elements in the same region of simulation mesh share one constitutive model. In this
 *    case the number of constitutive models equals the number of regions of the simulation mesh.
 * 3. Element-wise consitutive model is used.
 *
 */

template <typename Scalar, int Dim>
class FEMIsotropicHyperelasticSolid: public FEMBase<Scalar,Dim>
{
public:
    FEMIsotropicHyperelasticSolid();
    FEMIsotropicHyperelasticSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    ~FEMIsotropicHyperelasticSolid();

    //virtual methods
    void initialize();
    void advanceStep(Scalar dt);
    void write(const std::string &file_name);
    void read(const std::string &file_name);
    void addPlugin(DriverPluginBase<Scalar> *plugin);
    void initConfiguration(const std::string &file_name);

protected:
    std::vector<IsotropicHyperelasticMaterial<Scalar,Dim> *> constitutive_model_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_FEM_FEM_ISOTROPIC_HYPERELASTIC_SOLID_H_
