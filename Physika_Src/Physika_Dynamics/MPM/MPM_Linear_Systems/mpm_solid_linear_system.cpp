/*
 * @file mpm_solid_linear_system.cpp
 * @brief linear system for implicit integration of MPMSolid && CPDIMPMSolid driver
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

#include <typeinfo>
#include <vector>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/mpm_solid_linear_system.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidLinearSystem<Scalar,Dim>::MPMSolidLinearSystem(MPMSolid<Scalar,Dim> *mpm_solid_driver)
    :mpm_solid_driver_(mpm_solid_driver), active_obj_idx_(-1)
{
}

template <typename Scalar, int Dim>
MPMSolidLinearSystem<Scalar,Dim>::~MPMSolidLinearSystem()
{

}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::multiply(const GeneralizedVector<Scalar> &x,
                                                GeneralizedVector<Scalar> &result) const
{
    try
    {
        const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &xx = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>&>(x);
        UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &rr = dynamic_cast<UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>&>(result);
        energyHessianMultiply(xx, rr);
        Scalar dt_square = mpm_solid_driver_->computeTimeStep();
        dt_square *= dt_square;
        std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
        if (active_obj_idx_ == -1) //all objects solved together
        {
            mpm_solid_driver_->activeGridNodes(active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                rr[node_idx] = xx[node_idx] + dt_square*rr[node_idx] / mpm_solid_driver_->gridMass(node_idx);
            }
        }
        else  //solve for active object
        {
            mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                rr[node_idx] = xx[node_idx] + dt_square*rr[node_idx] / mpm_solid_driver_->gridMass(active_obj_idx_, node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument!");
    }
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::preconditionerMultiply(const GeneralizedVector<Scalar> &x,
                                                              GeneralizedVector<Scalar> &result) const
{
    try{
        const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &mpm_x = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>&>(x);
        UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &mpm_result = dynamic_cast<UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>&>(result);
        jacobiPreconditionerMultiply(mpm_x,mpm_result);
    }
    catch(std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
Scalar MPMSolidLinearSystem<Scalar, Dim>::innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const
{
    Scalar result = 0;
    try{
        const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &xx = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim>&>(x);
        const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &yy = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim>&>(y);
        std::vector<Vector<unsigned int, Dim> > active_nodes;
        if (active_obj_idx_ == -1) //solve for all objects
        {
            mpm_solid_driver_->activeGridNodes(active_nodes);
            for (unsigned int i = 0; i < active_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_nodes[i];
                result += xx[node_idx].dot(yy[node_idx]) * mpm_solid_driver_->gridMass(node_idx);
            }
        }
        else  //solve for active object
        {
            mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_nodes);
            for (unsigned int i = 0; i < active_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_nodes[i];
                result += xx[node_idx].dot(yy[node_idx]) * mpm_solid_driver_->gridMass(active_obj_idx_,node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return result;
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::setActiveObject(int obj_idx)
{
    active_obj_idx_ = obj_idx;
    //all negative values are set to -1
    active_obj_idx_ = active_obj_idx_ < 0 ? -1 : active_obj_idx_;
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar, Dim>::energyHessianMultiply(const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &x_diff,
                                                                  UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &result) const
{
    result.setValue(Vector<Scalar, Dim>(0));
    std::vector<unsigned int> active_objects;
    if (active_obj_idx_ == -1) //all objects solved together
    {
        active_objects.resize(mpm_solid_driver_->objectNum());
        for (unsigned int obj_idx = 0; obj_idx < active_objects.size(); ++obj_idx)
            active_objects[obj_idx] = obj_idx;
    }
    else  //solve for one active object
    {
        active_objects.resize(1);
        active_objects[0] = active_obj_idx_;
    }
    std::vector<Vector<unsigned int, Dim> > nodes_in_range;
    for (unsigned int active_idx = 0; active_idx < active_objects.size(); ++active_idx)
    {
        unsigned int obj_idx = active_objects[active_idx];
        for (unsigned int particle_idx = 0; particle_idx < mpm_solid_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SquareMatrix<Scalar, Dim> A_p(0);
            mpm_solid_driver_->gridNodesInRange(obj_idx, particle_idx, nodes_in_range);
            const SolidParticle<Scalar, Dim> &particle = mpm_solid_driver_->particle(obj_idx, particle_idx);
            for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
            {
                const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                Vector<Scalar, Dim> weight_gradient = mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                A_p += x_diff[node_idx].outerProduct(weight_gradient);
            }
            SquareMatrix<Scalar, Dim> particle_deform_grad = particle.deformationGradient();
            A_p = particle.constitutiveModel().firstPiolaKirchhoffStressDifferential(particle_deform_grad,A_p * particle_deform_grad);
            for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
            {
                const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                Vector<Scalar, Dim> weight_gradient = mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                result[node_idx] += mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p*particle_deform_grad.transpose()*weight_gradient;
            }
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar, Dim>::energyHessianDiagonalMultiply(const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &x_diff,
                                                                      UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &result) const
{
    result.setValue(Vector<Scalar, Dim>(0));
    std::vector<unsigned int> active_objects;
    if (active_obj_idx_ == -1) //all objects solved together
    {
        active_objects.resize(mpm_solid_driver_->objectNum());
        for (unsigned int obj_idx = 0; obj_idx < active_objects.size(); ++obj_idx)
            active_objects[obj_idx] = obj_idx;
    }
    else  //solve for one active object
    {
        active_objects.resize(1);
        active_objects[0] = active_obj_idx_;
    }
    std::vector<Vector<unsigned int, Dim> > nodes_in_range;
    for (unsigned int active_idx = 0; active_idx < active_objects.size(); ++active_idx)
    {
        unsigned int obj_idx = active_objects[active_idx];
        for (unsigned int particle_idx = 0; particle_idx < mpm_solid_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SquareMatrix<Scalar, Dim> A_p(0);
            mpm_solid_driver_->gridNodesInRange(obj_idx, particle_idx, nodes_in_range);
            const SolidParticle<Scalar, Dim> &particle = mpm_solid_driver_->particle(obj_idx, particle_idx);
            SquareMatrix<Scalar, Dim> particle_deform_grad = particle.deformationGradient();
            for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
            {
                const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                Vector<Scalar, Dim> weight_gradient = mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                A_p = x_diff[node_idx].outerProduct(weight_gradient);
                A_p = particle.constitutiveModel().firstPiolaKirchhoffStressDifferential(particle_deform_grad, A_p * particle_deform_grad);
                result[node_idx] += mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p*particle_deform_grad.transpose()*weight_gradient;
            }
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::jacobiPreconditionerMultiply(const UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &x,
                                                                    UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim> &result) const
{
    //in order to compute diag(A)^-1*x, we compute 1/(diag(A)*1/x)
    UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> rr = x;
    Scalar dt_square = mpm_solid_driver_->computeTimeStep();
    dt_square *= dt_square;
    std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
    const Scalar big_scalar = 1.0e7;
    if (active_obj_idx_ == -1) //all objects solved together
    {
        mpm_solid_driver_->activeGridNodes(active_grid_nodes);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            for (unsigned int dim = 0; dim < Dim; ++dim)
            {
                if (isEqual(rr[*iter][dim], static_cast<Scalar>(0.0)) == true)
                    rr[*iter][dim] = big_scalar;  //approximation
                else
                    rr[*iter][dim] = 1.0 / rr[*iter][dim];
            }
        energyHessianDiagonalMultiply(rr, result);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            result[*iter] = rr[*iter] + dt_square*result[*iter] / mpm_solid_driver_->gridMass(*iter);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            for (unsigned int dim = 0; dim < Dim; ++dim)
            {
                if (isEqual(result[*iter][dim], static_cast<Scalar>(0.0)) == true)
                    result[*iter][dim] = big_scalar; //approximation
                else
                    result[*iter][dim] = 1.0 / result[*iter][dim];
            }
    }
    else  //solve for one active object
    {
        mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            for (unsigned int dim = 0; dim < Dim; ++dim)
            {
                if (isEqual(rr[*iter][dim], static_cast<Scalar>(0.0)) == true)
                    rr[*iter][dim] = big_scalar;
                else
                    rr[*iter][dim] = 1.0 / rr[*iter][dim];
            }
        energyHessianDiagonalMultiply(rr, result);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            result[*iter] = rr[*iter] + dt_square*result[*iter] / mpm_solid_driver_->gridMass(active_obj_idx_,*iter);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
            for (unsigned int dim = 0; dim < Dim; ++dim)
            {
                if (isEqual(result[*iter][dim], static_cast<Scalar>(0.0)) == true)
                    result[*iter][dim] = big_scalar;  //approximation
                else
                    result[*iter][dim] = 1.0 / result[*iter][dim];
            }
    }
}

//explicit instantiations
template class MPMSolidLinearSystem<float,2>;
template class MPMSolidLinearSystem<float,3>;
template class MPMSolidLinearSystem<double,2>;
template class MPMSolidLinearSystem<double,3>;

}  //end of namespace Physika
