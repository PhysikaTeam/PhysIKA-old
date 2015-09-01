/*
 * @file mpm_uniform_grid_generalized_vector.cpp
 * @brief generalized vector for mpm drivers with uniform grid
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
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/mpm_uniform_grid_generalized_vector.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>::MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size)
    :UniformGridGeneralizedVector<Scalar,Dim>(grid_size)
{
    unsigned int total_node_num = (this->data_).totalElementCount();
    active_node_mass_.resize(total_node_num,1);
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>::MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size,
                                                        const std::vector<Vector<unsigned int,Dim> > &active_grid_nodes,
                                                        const std::vector<Scalar> &active_node_mass)
    :UniformGridGeneralizedVector<Scalar,Dim>(grid_size,active_grid_nodes),active_node_mass_(active_node_mass)
{
    if(active_grid_nodes.size() != active_node_mass.size())
        throw PhysikaException("Active node mass pattern does not match active node pattern!");
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>::MPMUniformGridGeneralizedVector(const MPMUniformGridGeneralizedVector<Scalar,Dim> &vector)
    :UniformGridGeneralizedVector<Scalar,Dim>(vector),active_node_mass_(vector.active_node_mass_)
{

}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>::~MPMUniformGridGeneralizedVector()
{

}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>& MPMUniformGridGeneralizedVector<Scalar,Dim>::operator=
                                                   (const MPMUniformGridGeneralizedVector<Scalar,Dim> &vector)
{
    copy(vector);
    return *this;
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Scalar,Dim>* MPMUniformGridGeneralizedVector<Scalar,Dim>::clone() const
{
    return new MPMUniformGridGeneralizedVector<Scalar,Dim>(*this);
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Scalar,Dim>::norm() const
{
    return sqrt(normSquared());
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Scalar,Dim>::normSquared() const
{
    PHYSIKA_ASSERT((this->active_node_idx_).size() == active_node_mass_.size());
    Scalar norm_sqr = 0;
    for(unsigned int i = 0; i < (this->active_node_idx_).size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = (this->active_node_idx_)[i];
        norm_sqr += (this->data_)(node_idx)*active_node_mass_[i]*(this->data_)(node_idx);
    }
    return norm_sqr;
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Scalar,Dim>::dot(const GeneralizedVector<Scalar> &vector) const
{
    try{
        const MPMUniformGridGeneralizedVector<Scalar,Dim>& mpm_vec = dynamic_cast<const MPMUniformGridGeneralizedVector<Scalar,Dim>&>(vector);
        this->checkActivePattern(mpm_vec);
        Scalar result = 0;
        for(unsigned int i = 0; i < (this->active_node_idx_).size(); ++i)
        {
            Vector<unsigned int,Dim> node_idx = (this->active_node_idx_)[i];
            result += (this->data_)(node_idx)*active_node_mass_[i]*(mpm_vec.data_)(node_idx);
        }
        return result;
    }
    catch(std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
void MPMUniformGridGeneralizedVector<Scalar,Dim>::setActiveNodeMass(const std::vector<Scalar> &active_node_mass)
{
    if((this->active_node_idx_).size() != active_node_mass.size())
        throw PhysikaException("Active node mass size does not match active node number!");
    else
        active_node_mass_ = active_node_mass;
}

template <typename Scalar, int Dim>
void MPMUniformGridGeneralizedVector<Scalar,Dim>::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const MPMUniformGridGeneralizedVector<Scalar,Dim> &mpm_vec = dynamic_cast<const MPMUniformGridGeneralizedVector<Scalar,Dim>&>(vector);
        this->data_ = mpm_vec.data_;
        this->active_node_idx_ = mpm_vec.active_node_idx_;
        this->active_node_mass_ = mpm_vec.active_node_mass_;
    }
    catch(std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

//explicit instantiations
template class MPMUniformGridGeneralizedVector<float,2>;
template class MPMUniformGridGeneralizedVector<float,3>;
template class MPMUniformGridGeneralizedVector<double,2>;
template class MPMUniformGridGeneralizedVector<double,3>;

}  //end of namespace Physika
