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
#include "Physika_Core/Utilities/physika_exception.h"
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
        Scalar dt_square = invertible_mpm_solid_driver_->computeTimeStep();
        dt_square *= dt_square;
        std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
        std::vector<Vector<unsigned int, 2> > enriched_domain_corners;
        if (active_obj_idx_ == -1)  //all objects solved together
        {
            invertible_mpm_solid_driver_->activeGridNodes(active_grid_nodes);
            //TO DO: get enriched domain corners
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                Vector<unsigned int, Dim> &node_idx = active_grid_nodes[i];
                rr[node_idx] = xx[node_idx] + dt_square*rr[node_idx] / invertible_mpm_solid_driver_->gridMass(node_idx);
            }
            //TO DO:
        }
        else //solve for active object
        {

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
    //TO DO
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolidLinearSystem<Scalar, Dim>::innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const
{
    //TO DO
    return 0;
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
    //TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolidLinearSystem<Scalar, Dim>::jacobiPreconditionerMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x,
                                                                               EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const
{
    //if (active_obj_idx_ == -1) //all objects solved together
    //{
    //    //TO DO
    //}
    //else  //solve for one active object
    //{
    //    //TO DO
    //}
    throw PhysikaException("Not implemented!");
}

//explicit instantiations
template class InvertibleMPMSolidLinearSystem<float, 2>;
template class InvertibleMPMSolidLinearSystem<float, 3>;
template class InvertibleMPMSolidLinearSystem<double, 2>;
template class InvertibleMPMSolidLinearSystem<double, 3>;

}  //end of namespace Physika