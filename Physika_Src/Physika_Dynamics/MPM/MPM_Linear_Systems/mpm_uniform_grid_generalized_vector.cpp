/*
 * @file mpm_uniform_grid_generalized_vector.cpp
 * @brief generalized vector for mpm drivers with uniform grid
 *        defined for element type Vector<Scalar,Dim>
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
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size)
    :UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>(grid_size)
{
    unsigned int total_node_num = (this->data_).totalElementCount();
    active_node_mass_.resize(total_node_num,1);
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size,
                                                        const std::vector<Vector<unsigned int,Dim> > &active_grid_nodes,
                                                        const std::vector<Scalar> &active_node_mass)
    :UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>(grid_size,active_grid_nodes),active_node_mass_(active_node_mass)
{
    if(active_grid_nodes.size() != active_node_mass.size())
        throw PhysikaException("Active node mass pattern does not match active node pattern!");
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::MPMUniformGridGeneralizedVector(const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &vector)
    :UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>(vector),active_node_mass_(vector.active_node_mass_)
{

}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::~MPMUniformGridGeneralizedVector()
{

}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >& MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::operator=
                                                   (const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &vector)
{
    copy(vector);
    return *this;
}

template <typename Scalar, int Dim>
MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >* MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::clone() const
{
    return new MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >(*this);
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::norm() const
{
    return sqrt(normSquared());
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::normSquared() const
{
    PHYSIKA_ASSERT((this->active_node_idx_).size() == active_node_mass_.size());
    Scalar norm_sqr = 0;
    if (this->active_node_idx_.size() == this->data_.totalElementCount()) //all entries active
    {
        unsigned int i = 0;
        for (typename ArrayND<Vector<Scalar, Dim>, Dim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter, ++i)
            norm_sqr += (*iter).normSquared()*active_node_mass_[i];
    }
    else
    {
        for (unsigned int i = 0; i < (this->active_node_idx_).size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = (this->active_node_idx_)[i];
            norm_sqr += (this->data_)(node_idx).normSquared()*active_node_mass_[i];
        }
    }
    return norm_sqr;
}

template <typename Scalar, int Dim>
Scalar MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::dot(const GeneralizedVector<Scalar> &vector) const
{
    Scalar result = 0;
    try{
        const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >& mpm_vec = dynamic_cast<const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >&>(vector);
        bool same_pattern = this->checkActivePattern(mpm_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (this->active_node_idx_.size() == this->data_.totalElementCount()) //all entries active
        {
            unsigned int i = 0;
            for (typename ArrayND<Vector<Scalar, Dim>, Dim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter, ++i)
            {
                Vector<unsigned int, Dim> node_idx = iter.elementIndex();
                result += (*iter).dot(mpm_vec.data_(node_idx))*active_node_mass_[i];
            }
            
        }
        else
        {
            for (unsigned int i = 0; i < (this->active_node_idx_).size(); ++i)
            {
                Vector<unsigned int, Dim> node_idx = (this->active_node_idx_)[i];
                result += (this->data_)(node_idx).dot((mpm_vec.data_)(node_idx))*active_node_mass_[i];
            }
        }
    }
    catch(std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return result;
}

template <typename Scalar, int Dim>
void MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::setActiveNodeMass(const std::vector<Scalar> &active_node_mass)
{
    if((this->active_node_idx_).size() != active_node_mass.size())
        throw PhysikaException("Active node mass size does not match active node number!");
    else
        active_node_mass_ = active_node_mass;
}

template <typename Scalar, int Dim>
void MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &mpm_vec = dynamic_cast<const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >&>(vector);
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
template class MPMUniformGridGeneralizedVector<Vector<float,2> >;
template class MPMUniformGridGeneralizedVector<Vector<float,3> >;
template class MPMUniformGridGeneralizedVector<Vector<double,2> >;
template class MPMUniformGridGeneralizedVector<Vector<double,3> >;

}  //end of namespace Physika
