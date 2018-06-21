/*
 * @file stomakhin_snow_model.h
 * @brief the isotropic elastoplastic model Stomakhin et al. presented in their SIGGRAPH paper
 *            "A material point method for snow simulation"
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_STOMAKHIN_SNOW_MODEL_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_STOMAKHIN_SNOW_MODEL_H_

#include "Physika_Dynamics/Constitutive_Models/isotropic_fixed_corotated_material.h"

namespace Physika{

/*
 * the elastoplastic model is based on fixed corotated hyperelastic material model
 * reference:
 *           paper: a material point method for snow simulation
 *           tec note: http://hydra.math.ucla.edu/~craig/papers/2013-mpm-snow/tech-doc.pdf
 */

template <typename Scalar, int Dim>
class StomakhinSnowModel: public IsotropicFixedCorotatedMaterial<Scalar,Dim>
{
public:
    StomakhinSnowModel();
    //if par_type = YOUNG_AND_POISSON, then: par1 = young's modulus, par2 = poisson_ratio
    //if par_type = LAME_COEFFICIENTS, then: par1 = lambda, par2 = mu
    StomakhinSnowModel(Scalar par1, Scalar par2, IsotropicHyperelasticMaterialInternal::ModulusType par_type);
    StomakhinSnowModel(Scalar par1, Scalar par2, IsotropicHyperelasticMaterialInternal::ModulusType par_type,
                       Scalar stretching_yield, Scalar compression_yield, Scalar hardening_factor);
    StomakhinSnowModel(const StomakhinSnowModel<Scalar,Dim> &material);
    virtual ~StomakhinSnowModel();
    StomakhinSnowModel<Scalar,Dim>& operator= (const StomakhinSnowModel<Scalar,Dim> &material);
    virtual StomakhinSnowModel<Scalar,Dim>* clone() const;
    Scalar stretchingYield() const;
    void setStretchingYield(Scalar stretching_yield);
    Scalar compressionYield() const;
    void setCompressionYield(Scalar compression_yield);
    Scalar hardeningFactor() const;
    void setHardeningFactor(Scalar hardening_factor);
    virtual void printInfo() const;
    virtual Scalar energyDensity(const SquareMatrix<Scalar,Dim> &F) const;
    virtual SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const;
    virtual SquareMatrix<Scalar,Dim> secondPiolaKirchhoffStress(const SquareMatrix<Scalar,Dim> &F) const;
    virtual SquareMatrix<Scalar,Dim> cauchyStress(const SquareMatrix<Scalar,Dim> &F) const;
    //differential of first PiolaKirchhoff stress, for implicit time integration
    // \delta P = dP/dF : (\delta F)
    // \delta is differential
    virtual SquareMatrix<Scalar,Dim> firstPiolaKirchhoffStressDifferential(const SquareMatrix<Scalar,Dim> &F,
                                     const SquareMatrix<Scalar,Dim> &F_differential) const;
    //decompose deformation into elastic && plastic part
    void deformationDecomposition(const SquareMatrix<Scalar,Dim> &F,
                                  SquareMatrix<Scalar,Dim> &F_e,
                                  SquareMatrix<Scalar,Dim> &F_p) const;
protected:
    //compute related material paramters according to current deformation gradient
    void prepareParameters(const SquareMatrix<Scalar,Dim> &F,
                           SquareMatrix<Scalar,Dim> &F_e,   //elastic part
                           SquareMatrix<Scalar,Dim> &R_e,   //rotation part
                           SquareMatrix<Scalar,Dim> &F_p,   //plastic part
                           Scalar &lambda, Scalar &mu) const;
protected:
    Scalar stretching_yield_, compression_yield_; //the two thresholds in the paper
    Scalar hardening_factor_; //the epsilon in the paper
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_STOMAKHIN_SNOW_MODEL_H_
