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
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/mpm_uniform_grid_generalized_vector.h"
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
        const MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &xx = dynamic_cast<const MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(x);
        MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &rr = dynamic_cast<MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >&>(result);
        internalForceDifferential(xx, rr);
        if (active_obj_idx_ == -1) //all objects solved together
        {

        }
        else  //solve for active object
        {
            std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
            mpm_solid_driver_->activeGridNodes(active_obj_idx_, active_grid_nodes);
            Scalar dt_square = mpm_solid_driver_->computeTimeStep();
            dt_square *= dt_square;
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
        const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &mpm_x = dynamic_cast<const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >&>(x);
        MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &mpm_result = dynamic_cast<MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >&>(result);
        jacobiPreconditionerMultiply(mpm_x,mpm_result);
    }
    catch(std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::setActiveObject(int obj_idx)
{
    active_obj_idx_ = obj_idx;
    //all negative values are set to -1
    active_obj_idx_ = active_obj_idx_ < 0 ? -1 : active_obj_idx_;
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar, Dim>::internalForceDifferential(const MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x_diff,
                                                                  MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const
{
    if (active_obj_idx_ == -1) //all objects solved together
    {

    }
    else  //solve for one active object
    {

    }
}

template <typename Scalar, int Dim>
void MPMSolidLinearSystem<Scalar,Dim>::jacobiPreconditionerMultiply(const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &x,
                                                                    MPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const
{
    if (active_obj_idx_ == -1) //all objects solved together
    {

    }
    else  //solve for one active object
    {

    }
}

//explicit instantiations
template class MPMSolidLinearSystem<float,2>;
template class MPMSolidLinearSystem<float,3>;
template class MPMSolidLinearSystem<double,2>;
template class MPMSolidLinearSystem<double,3>;

}  //end of namespace Physika
