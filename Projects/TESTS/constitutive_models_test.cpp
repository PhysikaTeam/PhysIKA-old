/*
 * @file constitutive_models_test.cpp 
 * @brief Test the constitutive models of solids.
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

#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/neo_hookean.h"
#include "Physika_Dynamics/Constitutive_Models/st_venant_kirchhoff.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_linear_elasticity.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"
using namespace std;
using Physika::NeoHookean;
using Physika::StVK;
using Physika::IsotropicLinearElasticity;
using Physika::IsotropicHyperelasticMaterial;
using Physika::IsotropicHyperelasticMaterial::ModulusType;

int main()
{
    float lambda = 1.0, mu =1.0;
    ModulusType par_type = Physika::LAME_COEFFICIENTS;
    NeoHookean<float,2> neo_hookean_material(lambda,mu,par_type);
    StVK<float,2> stvk_material(lambda,mu,par_type);
    IsotropicLinearElasticity<float,2> linear_material(lambda,mu,par_type);
    IsotropicHyperelasticMaterial<float,2> *isotropic_hyperelastic_material;
    Physika::Matrix2x2(float) F(2.0f,1.0f,0.0f,1.0f);
    cout<<"Deformation gradient: "<<endl;
    cout<<F<<endl;
    isotropic_hyperelastic_material = &neo_hookean_material;
    isotropic_hyperelastic_material->info();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energy(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    cout<<"E: "<<isotropic_hyperelastic_material->youngsModulus()<<", Nu: "<<isotropic_hyperelastic_material->poissonRatio()<<endl;
    cout<<"Setting Poisson Ratio to 0.4: "<<endl;
    isotropic_hyperelastic_material->setPoissonRatio(0.4);
    cout<<"E: "<<isotropic_hyperelastic_material->youngsModulus()<<", Nu: "<<isotropic_hyperelastic_material->poissonRatio()<<endl;
    cout<<"Lambda: "<<isotropic_hyperelastic_material->lambda()<<", Mu: "<<isotropic_hyperelastic_material->mu()<<endl;
    isotropic_hyperelastic_material = &stvk_material;
    isotropic_hyperelastic_material->info();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energy(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    isotropic_hyperelastic_material = &linear_material;
    isotropic_hyperelastic_material->info();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energy(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    return 0;
}
