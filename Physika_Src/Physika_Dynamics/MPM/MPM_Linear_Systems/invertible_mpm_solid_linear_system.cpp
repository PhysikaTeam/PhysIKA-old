/*
* @file invertible_mpm_solid_linear_system.cpp
* @brief linear system for implicit integration of InvertibleMPMSolid driver
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
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model.h"
#include "Physika_Dynamics/MPM/invertible_mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/enriched_mpm_uniform_grid_generalized_vector.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/invertible_mpm_solid_linear_system.h"

namespace Physika{

template <typename Scalar, int Dim>
InvertibleMPMSolidLinearSystem<Scalar, Dim>::InvertibleMPMSolidLinearSystem(InvertibleMPMSolid<Scalar, Dim> *invertible_driver)
    :invertible_mpm_solid_driver_(invertible_driver), active_obj_idx_(-1)
{

}

template <typename Scalar, int Dim>
InvertibleMPMSolidLinearSystem<Scalar, Dim>::~InvertibleMPMSolidLinearSystem()
{

}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const
{
    try
    {
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &xx = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(x);
        EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &rr = dynamic_cast<EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(result);
        energyHessianMultiply(xx, rr);
        Scalar dt_square = invertible_mpm_solid_driver_->timeStep();
        dt_square *= dt_square;
        std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
        std::vector<unsigned int> enriched_particles;
        unsigned int corner_num = Dim == 2 ? 4 : 8;
        Scalar beta = invertible_mpm_solid_driver_->implicitSteppingFraction();
        if (active_obj_idx_ == -1)  //all objects solved together
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                rr[node_idx] = xx[node_idx] + beta*dt_square*rr[node_idx] / invertible_mpm_solid_driver_->gridMass(node_idx);
            }
            for (unsigned int obj_idx = 0; obj_idx < invertible_mpm_solid_driver_->objectNum(); ++obj_idx)
            {
                invertible_mpm_solid_driver_->enrichedParticles(obj_idx, enriched_particles);
                for (unsigned int i = 0; i < enriched_particles.size(); ++i)
                {
                    unsigned int particle_idx = enriched_particles[i];
                    for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                    {
                        Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(obj_idx, particle_idx, corner_idx);
                        rr(obj_idx, particle_idx, corner_idx) = xx(obj_idx, particle_idx, corner_idx) + beta*dt_square*rr(obj_idx, particle_idx, corner_idx) / corner_mass;
                    }
                }
            }
        }
        else //solve for active object
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                rr[node_idx] = xx[node_idx] + beta*dt_square*rr[node_idx] / invertible_mpm_solid_driver_->gridMass(active_obj_idx_, node_idx);
            }
            invertible_mpm_solid_driver_->enrichedParticles(active_obj_idx_, enriched_particles);
            for (unsigned int i = 0; i < enriched_particles.size(); ++i)
            {
                unsigned int particle_idx = enriched_particles[i];
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(active_obj_idx_, particle_idx, corner_idx);
                    rr(0, particle_idx, corner_idx) = xx(0, particle_idx, corner_idx) + beta*dt_square*rr(0, particle_idx, corner_idx) / corner_mass;
                }
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument!");
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const
{
    try
    {
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &xx = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(x);
        EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &rr = dynamic_cast<EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(result);
        jacobiPreconditionerMultiply(xx, rr);
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolidLinearSystem<Scalar, Dim>::innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const
{
    Scalar result = 0;
    try
    {
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &xx = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(x);
        const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &yy = dynamic_cast<const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(y);

        std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
        std::vector<unsigned int> enriched_particles;
        unsigned int corner_num = Dim == 2 ? 4 : 8;
        if (active_obj_idx_ == -1)  //all objects solved together
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                result += xx[node_idx].dot(yy[node_idx])*invertible_mpm_solid_driver_->gridMass(node_idx);
            }
            for (unsigned int obj_idx = 0; obj_idx < invertible_mpm_solid_driver_->objectNum(); ++obj_idx)
            {
                invertible_mpm_solid_driver_->enrichedParticles(obj_idx, enriched_particles);
                for (unsigned int i = 0; i < enriched_particles.size(); ++i)
                {
                    unsigned int particle_idx = enriched_particles[i];
                    for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                    {
                        Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(obj_idx, particle_idx, corner_idx);
                        result += xx(obj_idx, particle_idx, corner_idx).dot(yy(obj_idx, particle_idx, corner_idx))*corner_mass;
                    }
                }
            }
        }
        else //solve for active object
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                result += xx[node_idx].dot(yy[node_idx])*invertible_mpm_solid_driver_->gridMass(active_obj_idx_,node_idx);
            }
            invertible_mpm_solid_driver_->enrichedParticles(active_obj_idx_, enriched_particles);
            for (unsigned int i = 0; i < enriched_particles.size(); ++i)
            {
                unsigned int particle_idx = enriched_particles[i];
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(active_obj_idx_, particle_idx, corner_idx);
                    result += xx(0, particle_idx, corner_idx).dot(yy(0, particle_idx, corner_idx))*corner_mass;
                }
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
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::filter(GeneralizedVector<Scalar> &x) const
{
    try
    {
        EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &xx = dynamic_cast<EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(x);
        std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
        if (active_obj_idx_ == -1) //solve for all objects
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_grid_nodes);
            for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
                if (invertible_mpm_solid_driver_->isDirichletGridNode(*iter))
                    xx[*iter] = Vector<Scalar, Dim>(0.0);
        }
        else
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
            for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
                if (invertible_mpm_solid_driver_->isDirichletGridNode(active_obj_idx_, *iter))
                    xx[*iter] = Vector<Scalar, Dim>(0.0);
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::setActiveObject(int obj_idx)
{
    active_obj_idx_ = obj_idx;
    //all negative values are set to -1
    active_obj_idx_ = active_obj_idx_ < 0 ? -1 : active_obj_idx_;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::energyHessianMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x_diff,
                                                                        EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const
{
    result.setValue(Vector<Scalar, Dim>(0));
    std::vector<unsigned int> active_objects;
    if (active_obj_idx_ == -1) //all objects solved together
    {
        active_objects.resize(invertible_mpm_solid_driver_->objectNum());
        for (unsigned int obj_idx = 0; obj_idx < active_objects.size(); ++obj_idx)
            active_objects[obj_idx] = obj_idx;
    }
    else  //solve for one active object
    {
        active_objects.resize(1);
        active_objects[0] = active_obj_idx_;
    }
    std::vector<Vector<unsigned int, Dim> > nodes_in_range;
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    SquareMatrix<Scalar, Dim> particle_deform_grad, left_rotation, diag_deform_grad, right_rotation, rotated_dP;
    Scalar stretch_threshold = invertible_mpm_solid_driver_->principalStretchThreshold();
    for (unsigned int active_idx = 0; active_idx < active_objects.size(); ++active_idx)
    {
        unsigned int obj_idx = active_objects[active_idx];
        for (unsigned int particle_idx = 0; particle_idx < invertible_mpm_solid_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SquareMatrix<Scalar, Dim> A_p(0);
            invertible_mpm_solid_driver_->gridNodesInRange(obj_idx, particle_idx, nodes_in_range);
            const SolidParticle<Scalar, Dim> &particle = invertible_mpm_solid_driver_->particle(obj_idx, particle_idx);
            unsigned int enriched_corner_num = invertible_mpm_solid_driver_->enrichedDomainCornerNum(obj_idx, particle_idx);
            if (enriched_corner_num < corner_num)
            {//grid nodes
                for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
                {
                    const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                    Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                    A_p += x_diff[node_idx].outerProduct(weight_gradient);
                }
            }
            if (enriched_corner_num > 0)
            {//enriched domain corners
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    if (invertible_mpm_solid_driver_->isEnrichedDomainCorner(obj_idx, particle_idx, corner_idx))
                    {
                        Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->particleDomainCornerGradient(obj_idx, particle_idx, corner_idx);
                        if (active_idx == -1)
                            A_p += x_diff(obj_idx, particle_idx, corner_idx).outerProduct(weight_gradient);
                        else
                            A_p += x_diff(0, particle_idx, corner_idx).outerProduct(weight_gradient);
                    }
                }
            }
            //use technique from 'Teran et al. 05: Robust Quasistatic Finite Elements and Flesh Simulation' to ensure positive definiteness
            particle_deform_grad = particle.deformationGradient();
            invertible_mpm_solid_driver_->diagonalizedParticleDeformationGradient(obj_idx, particle_idx, left_rotation,diag_deform_grad,right_rotation);
            //clamp the principal stretch to the threshold if it's compressed too severely
            for (unsigned int row = 0; row < Dim; ++row)
                if (diag_deform_grad(row, row) < stretch_threshold)
                    diag_deform_grad(row, row) = stretch_threshold;
            rotated_dP = particle.constitutiveModel().firstPiolaKirchhoffStressDifferential(diag_deform_grad, left_rotation.transpose()*A_p*right_rotation);
            A_p = left_rotation*rotated_dP*right_rotation.transpose();
            if (enriched_corner_num < corner_num)
            {
                for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
                {
                    const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                    Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                    result[node_idx] += invertible_mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p*weight_gradient;
                }
            }
            if (enriched_corner_num > 0)
            {
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    if (invertible_mpm_solid_driver_->isEnrichedDomainCorner(obj_idx, particle_idx, corner_idx))
                    {
                        Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->particleDomainCornerGradient(obj_idx, particle_idx, corner_idx);
                        if (active_idx == -1)
                            result(obj_idx, particle_idx, corner_idx) += invertible_mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p*weight_gradient;
                        else
                            result(0, particle_idx, corner_idx) += invertible_mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p*weight_gradient;
                    }
                }
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::energyHessianDiagonal(EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &diagonals) const
{
    diagonals.setValue(Vector<Scalar, Dim>(0));
    std::vector<unsigned int> active_objects;
    if (active_obj_idx_ == -1) //all objects solved together
    {
        active_objects.resize(invertible_mpm_solid_driver_->objectNum());
        for (unsigned int obj_idx = 0; obj_idx < active_objects.size(); ++obj_idx)
            active_objects[obj_idx] = obj_idx;
    }
    else  //solve for one active object
    {
        active_objects.resize(1);
        active_objects[0] = active_obj_idx_;
    }
    std::vector<Vector<unsigned int, Dim> > nodes_in_range;
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    SquareMatrix<Scalar, Dim> particle_deform_grad, left_rotation, diag_deform_grad, right_rotation, rotated_dP;
    Scalar stretch_threshold = invertible_mpm_solid_driver_->principalStretchThreshold();
    for (unsigned int active_idx = 0; active_idx < active_objects.size(); ++active_idx)
    {
        unsigned int obj_idx = active_objects[active_idx];
        for (unsigned int particle_idx = 0; particle_idx < invertible_mpm_solid_driver_->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SquareMatrix<Scalar, Dim> A_p(0);
            invertible_mpm_solid_driver_->gridNodesInRange(obj_idx, particle_idx, nodes_in_range);
            const SolidParticle<Scalar, Dim> &particle = invertible_mpm_solid_driver_->particle(obj_idx, particle_idx);
            particle_deform_grad = particle.deformationGradient();
            invertible_mpm_solid_driver_->diagonalizedParticleDeformationGradient(obj_idx, particle_idx, left_rotation, diag_deform_grad, right_rotation);
            //clamp the principal stretch to the threshold if it's compressed too severely
            for (unsigned int row = 0; row < Dim; ++row)
                if (diag_deform_grad(row, row) < stretch_threshold)
                    diag_deform_grad(row, row) = stretch_threshold;
            unsigned int enriched_corner_num = invertible_mpm_solid_driver_->enrichedDomainCornerNum(obj_idx, particle_idx);
            if (enriched_corner_num < corner_num)
            {//grid nodes
                for (unsigned int idx = 0; idx < nodes_in_range.size(); ++idx)
                {
                    const Vector<unsigned int, Dim> &node_idx = nodes_in_range[idx];
                    Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->weightGradient(obj_idx, particle_idx, node_idx);
                    A_p = weight_gradient.outerProduct(weight_gradient);
                    rotated_dP = particle.constitutiveModel().firstPiolaKirchhoffStressDifferential(diag_deform_grad, left_rotation.transpose()*A_p*right_rotation);
                    A_p = left_rotation*rotated_dP*right_rotation.transpose();
                    A_p = invertible_mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p;
                    for (unsigned int i = 0; i < Dim; ++i)
                        diagonals[node_idx][i] += A_p(i, i);
                }
            }
            if (enriched_corner_num > 0)
            {//enriched domain corners
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    if (invertible_mpm_solid_driver_->isEnrichedDomainCorner(obj_idx, particle_idx, corner_idx))
                    {
                        Vector<Scalar, Dim> weight_gradient = invertible_mpm_solid_driver_->particleDomainCornerGradient(obj_idx, particle_idx, corner_idx);
                        unsigned int idx = active_obj_idx_ == -1 ? obj_idx : 0;
                        A_p = weight_gradient.outerProduct(weight_gradient);
                        rotated_dP = particle.constitutiveModel().firstPiolaKirchhoffStressDifferential(diag_deform_grad, left_rotation.transpose()*A_p*right_rotation);
                        A_p = left_rotation*rotated_dP*right_rotation.transpose();
                        A_p = invertible_mpm_solid_driver_->particleInitialVolume(obj_idx, particle_idx)*A_p;
                        for (unsigned int i = 0; i < Dim; ++i)
                            diagonals(idx, particle_idx, corner_idx)[i] += A_p(i, i);
                    }
                }
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::jacobiPreconditionerMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x,
                                                                               EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const
{
    Scalar dt_square = invertible_mpm_solid_driver_->timeStep();
    dt_square *= dt_square;
    std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
    std::vector<unsigned int> enriched_particles;
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    Scalar beta = invertible_mpm_solid_driver_->implicitSteppingFraction();
    EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > diagonals = x;
    if (active_obj_idx_ == -1)  //all objects solved together
    {
        invertible_mpm_solid_driver_->activeGridNodes(active_grid_nodes);
        energyHessianDiagonal(diagonals);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
        {
            SquareMatrix<Scalar, Dim> inv_diag(0.0);
            Scalar grid_mass = invertible_mpm_solid_driver_->gridMass(*iter);
            for (unsigned int i = 0; i < Dim; ++i)
                inv_diag(i, i) = 1.0 / (1 + beta*dt_square*diagonals[*iter][i] / grid_mass);
            result[*iter] = inv_diag*x[*iter];
        }
        for (unsigned int obj_idx = 0; obj_idx < invertible_mpm_solid_driver_->objectNum(); ++obj_idx)
        {
            invertible_mpm_solid_driver_->enrichedParticles(obj_idx, enriched_particles);
            for (std::vector<unsigned int>::iterator iter = enriched_particles.begin(); iter != enriched_particles.end(); ++iter)
            {
                unsigned int particle_idx = *iter;
                for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(obj_idx, particle_idx, corner_idx);
                    SquareMatrix<Scalar, Dim> inv_diag(0.0);
                    for (unsigned int i = 0; i < Dim; ++i)
                        inv_diag(i, i) = 1.0 / (1 + beta*dt_square*diagonals(obj_idx, particle_idx, corner_idx)[i] / corner_mass);
                    result(obj_idx, particle_idx, corner_idx) = inv_diag*x(obj_idx, particle_idx, corner_idx);
                }
            }
        }
    }
    else //solve for active object
    {
        invertible_mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
        invertible_mpm_solid_driver_->enrichedParticles(active_obj_idx_, enriched_particles);
        energyHessianDiagonal(diagonals);
        for (typename std::vector<Vector<unsigned int, Dim> >::iterator iter = active_grid_nodes.begin(); iter != active_grid_nodes.end(); ++iter)
        {
            SquareMatrix<Scalar, Dim> inv_diag(0.0);
            Scalar grid_mass = invertible_mpm_solid_driver_->gridMass(active_obj_idx_,*iter);
            for (unsigned int i = 0; i < Dim; ++i)
                inv_diag(i, i) = 1.0 / (1 + beta*dt_square*diagonals[*iter][i] / grid_mass);
            result[*iter] = inv_diag*x[*iter];
        }
        for (std::vector<unsigned int>::iterator iter = enriched_particles.begin(); iter != enriched_particles.end(); ++iter)
        {
            unsigned int particle_idx = *iter;
            for (unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                Scalar corner_mass = invertible_mpm_solid_driver_->domainCornerMass(active_obj_idx_, particle_idx, corner_idx);
                SquareMatrix<Scalar, Dim> inv_diag(0.0);
                for (unsigned int i = 0; i < Dim; ++i)
                    inv_diag(i, i) = 1.0 / (1 + beta*dt_square*diagonals(0, particle_idx, corner_idx)[i] / corner_mass);
                result(0, particle_idx, corner_idx) = inv_diag*x(0,particle_idx,corner_idx);
            }
        }
    }
}

//explicit instantiations
template class InvertibleMPMSolidLinearSystem<float, 2>;
template class InvertibleMPMSolidLinearSystem<float, 3>;
template class InvertibleMPMSolidLinearSystem<double, 2>;
template class InvertibleMPMSolidLinearSystem<double, 3>;

}  //end of namespace Physika