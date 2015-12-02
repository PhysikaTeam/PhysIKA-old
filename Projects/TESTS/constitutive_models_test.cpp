/*
* @file constitutive_models_test.cpp
* @brief Test the constitutive models of solids.
* @author Fei Zhu
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include <ctime>
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Constitutive_Models/neo_hookean.h"
#include "Physika_Dynamics/Constitutive_Models/st_venant_kirchhoff.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_linear_elasticity.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_fixed_corotated_material.h"
#include "Physika_Dynamics/Constitutive_Models/isotropic_hyperelastic_material.h"
#include "Physika_Dynamics/Constitutive_Models/stomakhin_snow_model.h"
using namespace std;
using Physika::SquareMatrix;
using Physika::StomakhinSnowModel;
using Physika::NeoHookean;
using Physika::StVK;
using Physika::IsotropicFixedCorotatedMaterial;
using Physika::IsotropicLinearElasticity;
using Physika::IsotropicHyperelasticMaterial;
using Physika::IsotropicHyperelasticMaterialInternal::ModulusType;
using Physika::IsotropicHyperelasticMaterialInternal::LAME_COEFFICIENTS;
typedef double Scalar;
const int Dim = 3;

//test if the gradient implementation of constitutive models is correct, i.e.,
// firstPiolaKirchhoffStress is gradient of energyDensity with respect to F
// firstPiolaKirchhoffStressDifferential is gradient of firstPiolaKirchhoffStress multiplies some delta
void testGradients(const IsotropicHyperelasticMaterial<Scalar,Dim> *material)
{
    SquareMatrix<Scalar,Dim> F(0.0);
    //randomly generate deformation gradient with entries in range [-1,1]
    srand((unsigned int)time(0));
    int lowest = -10, highest = 10;
    int range = highest - lowest;
    for(unsigned int row = 0; row < Dim; ++row)
        for(unsigned int col = 0; col < Dim; ++col)
        {
            int random_integer = lowest + rand()%range;
            F(row,col) = random_integer/10.0;
        }
    //randomly generate deformation gradient differential with entries in range [-1e-6,1e-6]
    SquareMatrix<Scalar,Dim> dF(0);
    for(unsigned int row = 0; row < Dim; ++row)
        for(unsigned int col = 0; col < Dim; ++col)
        {
            int random_integer = lowest + rand()%range;
            dF(row,col) = random_integer/1e7;
        }
    Scalar Psi1 = material->energyDensity(F);
    SquareMatrix<Scalar,Dim> P1 = material->firstPiolaKirchhoffStress(F);
    SquareMatrix<Scalar,Dim> dP1 = material->firstPiolaKirchhoffStressDifferential(F,dF);
    F = F + dF;
    Scalar Psi2 = material->energyDensity(F);
    SquareMatrix<Scalar,Dim> P2 = material->firstPiolaKirchhoffStress(F);
    SquareMatrix<Scalar,Dim> dP2 = material->firstPiolaKirchhoffStressDifferential(F,dF);
    std::cout<<"P rel_error: "<<((Psi2-Psi1)-(P2+P1).doubleContraction(dF)*0.5)/(Psi2-Psi1)<<"\n";
    std::cout<<"P_differential rel_error: "<<((P2-P1)-(dP2+dP1)/2).frobeniusNorm()/(P2-P1).frobeniusNorm()<<"\n";
}

int main()
{
    Scalar lambda = 1.0, mu =1.0;
    ModulusType par_type = LAME_COEFFICIENTS;
    NeoHookean<Scalar,Dim> neo_hookean_material(lambda,mu,par_type);
    StVK<Scalar,Dim> stvk_material(lambda,mu,par_type);
    IsotropicLinearElasticity<Scalar,Dim> linear_material(lambda,mu,par_type);
    IsotropicFixedCorotatedMaterial<Scalar,Dim> corotated_linear_material(lambda,mu,par_type);
    StomakhinSnowModel<Scalar,Dim> snow_material(lambda,mu,par_type);
    IsotropicHyperelasticMaterial<Scalar,Dim> *isotropic_hyperelastic_material;
    SquareMatrix<Scalar,Dim> F(0.0);
    //randomly generate deformation gradient with entries in range [-1,1]
    srand((unsigned int)time(0));
    int lowest = -10, highest = 10;
    int range = highest - lowest;
    for(unsigned int row = 0; row < Dim; ++row)
    for(unsigned int col = 0; col < Dim; ++col)
    {
        int random_integer = lowest + rand()%range;
        F(row,col) = random_integer/10.0;
    }
    cout<<"Deformation gradient: "<<endl;
    cout<<F<<endl;
    isotropic_hyperelastic_material = &neo_hookean_material;
    isotropic_hyperelastic_material->printInfo();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energyDensity(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    cout<<"E: "<<isotropic_hyperelastic_material->youngsModulus()<<", Nu: "<<isotropic_hyperelastic_material->poissonRatio()<<endl;
    cout<<"Setting Poisson Ratio to 0.4: "<<endl;
    isotropic_hyperelastic_material->setPoissonRatio(0.4);
    cout<<"E: "<<isotropic_hyperelastic_material->youngsModulus()<<", Nu: "<<isotropic_hyperelastic_material->poissonRatio()<<endl;
    cout<<"Lambda: "<<isotropic_hyperelastic_material->lambda()<<", Mu: "<<isotropic_hyperelastic_material->mu()<<endl;
    testGradients(isotropic_hyperelastic_material);

    isotropic_hyperelastic_material = &stvk_material;
    isotropic_hyperelastic_material->printInfo();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energyDensity(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    testGradients(isotropic_hyperelastic_material);

    isotropic_hyperelastic_material = &linear_material;
    isotropic_hyperelastic_material->printInfo();
    cout<<"Internal energy: "<<isotropic_hyperelastic_material->energyDensity(F)<<endl;
    cout<<"P: "<<isotropic_hyperelastic_material->firstPiolaKirchhoffStress(F)<<endl;
    cout<<"S: "<<isotropic_hyperelastic_material->secondPiolaKirchhoffStress(F)<<endl;
    cout<<"sigma: "<<isotropic_hyperelastic_material->cauchyStress(F)<<endl;
    testGradients(isotropic_hyperelastic_material);

    isotropic_hyperelastic_material = &snow_material;
    isotropic_hyperelastic_material->printInfo();
    testGradients(isotropic_hyperelastic_material);

    isotropic_hyperelastic_material = &corotated_linear_material;
    isotropic_hyperelastic_material->printInfo();
    testGradients(isotropic_hyperelastic_material);
    getchar();
    return 0;
}
