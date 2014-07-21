/*
 * @file mpm_particle.h 
 * @Brief the particle class used in MPM
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_PARTICLE_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_PARTICLE_H_

#include "Physika_Dynamics/Particles/particle.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

/*
 * MPMParticle: particle used in MPM, contains basic information needed by MPM
 * It could be inherited to contain more information, e.g., fracture tendency, melting temporature, etc.
 * MPM drivers are designed to make use of the polymorphism of MPM particles hirerachy
 */

template <typename Scalar, int Dim>
class MPMParticle: public Particle<Scalar,Dim>
{
public:
    MPMParticle();
    MPMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol); //F set to identity, weight and weight gradient unset
    MPMParticle(const Vector<Scalar,Dim> &pos, const Vector<Scalar,Dim> &vel, Scalar mass, Scalar vol, const SquareMatrix<Scalar,Dim> &deform_grad, Scalar weight, const Vector<Scalar,Dim> &weight_grad);
    MPMParticle(const MPMParticle<Scalar,Dim> &particle);
    virtual ~MPMParticle(); //made virtual such that polymorphism is made use of in MPM drivers 
    MPMParticle<Scalar,Dim>& operator= (const MPMParticle<Scalar,Dim> &particle);
    const SquareMatrix<Scalar,Dim>& deformationGradient() const;
    void setDeformationGradient(const SquareMatrix<Scalar,Dim> &F);
    Scalar weight() const;
    void setWeight(Scalar weight);
    const Vector<Scalar,Dim>& weightGradient() const;
    void setWeightGradient(const Vector<Scalar,Dim> &weight_grad);
protected:
    SquareMatrix<Scalar,Dim> F_;
    Scalar weight_; //value of weight function with respect to grid
    Vector<Scalar,Dim> weight_grad_;  //gradient of weight function
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_PARTICLE_H_
