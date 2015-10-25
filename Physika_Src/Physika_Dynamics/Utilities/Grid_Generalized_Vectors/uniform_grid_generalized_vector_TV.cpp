/*
* @file uniform_grid_generalized_vector_TV.cpp
* @brief generalized vector for solving the linear system Ax = b on uniform grid
*        defined for Vector/VectorND element type
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

#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_TV.h"

namespace Physika{

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::UniformGridGeneralizedVector(const Vector<unsigned int, GridDim> &grid_size)
    :GeneralizedVector<Scalar>(), data_(grid_size,Vector<Scalar,EleDim>(0.0))
{
    //only size of active_node_idx_ makes sense if all nodes are active
    active_node_idx_.resize(data_.totalElementCount());
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::UniformGridGeneralizedVector(const Vector<unsigned int, GridDim> &grid_size, const std::vector<Vector<unsigned int, GridDim> > &active_grid_nodes)
    :GeneralizedVector<Scalar>(), data_(grid_size, Vector<Scalar, EleDim>(0.0)), active_node_idx_(active_grid_nodes)
{
    sortActiveNodes();
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::UniformGridGeneralizedVector(const UniformGridGeneralizedVector<Vector<Scalar,EleDim>,GridDim> &vector)
    :GeneralizedVector<Scalar>(vector), data_(vector.data_), active_node_idx_(vector.active_node_idx_)
{

}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::~UniformGridGeneralizedVector()
{

}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator=(const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& vector)
{
    data_ = vector.data_;
    active_node_idx_ = vector.active_node_idx_;
    return *this;
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>* UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::clone() const
{
    return new UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>(*this);
}

template <typename Scalar, int GridDim, int EleDim>
unsigned int UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::size() const
{
    return active_node_idx_.size();
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator+= (const GeneralizedVector<Scalar>& vector)
{
    try{
        const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
        {
            for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, GridDim> node_idx = iter.elementIndex();
                *iter += grid_vec.data_(node_idx);
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
                data_(node_idx) += grid_vec.data_(node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return *this;
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator-= (const GeneralizedVector<Scalar>& vector)
{
    try{
        const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
        {
            for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, GridDim> node_idx = iter.elementIndex();
                *iter -= grid_vec.data_(node_idx);
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
                data_(node_idx) -= grid_vec.data_(node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return *this;
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator *=(Scalar value)
{
    if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
    {
        for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter *= value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
            data_(node_idx) *= value;
        }
    }
    return *this;
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator /=(Scalar value)
{
    if (isEqual(value, static_cast<Scalar>(0.0)))
        throw PhysikaException("Divide by zero!");
    if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
    {
        for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter /= value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
            data_(node_idx) /= value;
        }
    }
    return *this;
}

template <typename Scalar, int GridDim, int EleDim>
Scalar UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::norm() const
{
    return sqrt(normSquared());
}

template <typename Scalar, int GridDim, int EleDim>
Scalar UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::normSquared() const
{
    Scalar norm_sqr = 0;
    if (active_node_idx_.size() == data_.totalElementCount())  //all entries active
    {
        for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
            norm_sqr += (*iter).normSquared();
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
            norm_sqr += data_(node_idx).normSquared();
        }
    }
    return norm_sqr;
}

template <typename Scalar, int GridDim, int EleDim>
Scalar UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::dot(const GeneralizedVector<Scalar> &vector) const
{
    Scalar result = 0;
    try{
        const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
        {
            for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, GridDim> node_idx = iter.elementIndex();
                result += (*iter).dot(grid_vec.data_(node_idx));
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
                result += data_(node_idx).dot(grid_vec.data_(node_idx));
            }
        }
    }
    catch (std::bad_cast& e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return result;
}

template <typename Scalar, int GridDim, int EleDim>
const Vector<Scalar, EleDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator[] (const Vector<unsigned int, GridDim> &idx) const
{
    return data_(idx);
}

template <typename Scalar, int GridDim, int EleDim>
Vector<Scalar, EleDim>& UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::operator[] (const Vector<unsigned int, GridDim> &idx)
{
    return data_(idx);
}

template <typename Scalar, int GridDim, int EleDim>
void UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::setValue(const Vector<Scalar, EleDim> &value)
{
    if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
    {
        for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter = value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
            data_(node_idx) = value;
        }
    }
}

template <typename Scalar, int GridDim, int EleDim>
void UniformGridGeneralizedVector < Vector<Scalar, EleDim>, GridDim >::setActivePattern(const std::vector<Vector<unsigned int, GridDim> > &active_grid_nodes)
{
    active_node_idx_ = active_grid_nodes;
    sortActiveNodes();
}

template <typename Scalar, int GridDim, int EleDim>
UniformGridGeneralizedVector<Scalar, GridDim> UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::vectorAtDim(unsigned int val_dim_idx) const
{
    if (val_dim_idx >= EleDim)
        throw PhysikaException("Element dimension out of range!");
    UniformGridGeneralizedVector<Scalar, GridDim> vec(data_.size(), active_node_idx_);
    if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
    {
        for (typename ArrayND<Vector<Scalar, EleDim>, GridDim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
        {
            Vector<unsigned int, GridDim> node_idx = iter.elementIndex();
            vec[node_idx] = data_(node_idx)[val_dim_idx];
        }
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, GridDim> node_idx = active_node_idx_[i];
            vec[node_idx] = data_(node_idx)[val_dim_idx];
        }
    }
    return vec;
}

template <typename Scalar, int GridDim, int EleDim>
unsigned int UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::valueDim() const
{
    return EleDim;
}

template <typename Scalar, int GridDim, int EleDim>
void UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>&>(vector);
        data_ = grid_vec.data_;
        active_node_idx_ = grid_vec.active_node_idx_;
    }
    catch (std::bad_cast& e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int GridDim, int EleDim>
bool UniformGridGeneralizedVector < Vector<Scalar, EleDim>, GridDim > ::checkActivePattern(const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim> &vector) const
{
    if (active_node_idx_.size() != vector.active_node_idx_.size())
        return false;
    else
    {
        if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
            return true;
        else
        {
            //active node idx is in order
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
                if (active_node_idx_[i] != vector.active_node_idx_[i])
                    return false;
            return true;
        }
    }
}

template <typename Scalar, int GridDim, int EleDim>
void UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>::sortActiveNodes()
{
    if (active_node_idx_.size() != data_.totalElementCount())
    {
        //helper struct for sortActiveNodes
        struct CompareVector
        {
            UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim> *master;
            bool operator()(const Vector<unsigned int, GridDim> &vec1, const Vector<unsigned int, GridDim> &vec2) const
            {
                Vector<unsigned int, GridDim> node_num = (master->data_).size();
                Vector<unsigned int, GridDim> tmp_vec1 = vec1, tmp_vec2 = vec2;
                unsigned int flat_vec1 = 0, flat_vec2 = 0;
                for (unsigned int i = 0; i < GridDim; ++i)
                {
                    for (unsigned int j = i + 1; j < GridDim; ++j)
                    {
                        tmp_vec1[i] *= node_num[j];
                        tmp_vec2[i] *= node_num[j];
                    }
                    flat_vec1 += tmp_vec1[i];
                    flat_vec2 += tmp_vec2[i];
                }
                return flat_vec1 < flat_vec2;
            }
        };
        CompareVector compare_vector;
        compare_vector.master = this;
        std::sort(active_node_idx_.begin(), active_node_idx_.end(), compare_vector);
    }
}

///////////////////////////////////////////////////////////

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::UniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size)
    :GeneralizedVector<Scalar>(), data_(grid_size, VectorND<Scalar>(0.0))
{
    //only size of active_node_idx_ makes sense if all nodes are active
    active_node_idx_.resize(data_.totalElementCount());
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::UniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size, const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes)
    :GeneralizedVector<Scalar>(), data_(grid_size, VectorND<Scalar>(0.0)), active_node_idx_(active_grid_nodes)
{
    sortActiveNodes();
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::UniformGridGeneralizedVector(const UniformGridGeneralizedVector<VectorND<Scalar>, Dim> &vector)
    :GeneralizedVector<Scalar>(vector), data_(vector.data_), active_node_idx_(vector.active_node_idx_)
{

}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::~UniformGridGeneralizedVector()
{

}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator=(const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& vector)
{
    data_ = vector.data_;
    active_node_idx_ = vector.active_node_idx_;
    return *this;
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>* UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::clone() const
{
    return new UniformGridGeneralizedVector<VectorND<Scalar>, Dim>(*this);
}

template <typename Scalar, int Dim>
unsigned int UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::size() const
{
    return active_node_idx_.size();
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator+= (const GeneralizedVector<Scalar>& vector)
{
    try{
        const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
        {
            for (typename ArrayND<VectorND<Scalar>, Dim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, Dim> node_idx = iter.elementIndex();
                *iter += grid_vec.data_(node_idx);
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
                data_(node_idx) += grid_vec.data_(node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return *this;
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator-= (const GeneralizedVector<Scalar>& vector)
{
    try{
        const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
        {
            for (typename ArrayND<VectorND<Scalar>, Dim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, Dim> node_idx = iter.elementIndex();
                *iter -= grid_vec.data_(node_idx);
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
                data_(node_idx) -= grid_vec.data_(node_idx);
            }
        }
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return *this;
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator *=(Scalar value)
{
    if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
    {
        for (typename ArrayND<VectorND<Scalar>, Dim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter *= value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
            data_(node_idx) *= value;
        }
    }
    return *this;
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator /=(Scalar value)
{
    if (isEqual(value, static_cast<Scalar>(0.0)))
        throw PhysikaException("Divide by zero!");
    if (active_node_idx_.size() == data_.totalElementCount()) //all entries active
    {
        for (typename ArrayND<VectorND<Scalar>, Dim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter /= value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
            data_(node_idx) /= value;
        }
    }
    return *this;
}

template <typename Scalar, int Dim>
Scalar UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::norm() const
{
    return sqrt(normSquared());
}

template <typename Scalar, int Dim>
Scalar UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::normSquared() const
{
    Scalar norm_sqr = 0;
    if (active_node_idx_.size() == data_.totalElementCount())  //all entries active
    {
        for (typename ArrayND<VectorND<Scalar>, Dim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
            norm_sqr += (*iter).normSquared();
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
            norm_sqr += data_(node_idx).normSquared();
        }
    }
    return norm_sqr;
}

template <typename Scalar, int Dim>
Scalar UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::dot(const GeneralizedVector<Scalar> &vector) const
{
    Scalar result = 0;
    try{
        const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>&>(vector);
        bool same_pattern = checkActivePattern(grid_vec);
        if (!same_pattern)
            throw PhysikaException("Active entry pattern does not match!");
        if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
        {
            for (typename ArrayND<VectorND<Scalar>, Dim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
            {
                Vector<unsigned int, Dim> node_idx = iter.elementIndex();
                result += (*iter).dot(grid_vec.data_(node_idx));
            }
        }
        else
        {
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
            {
                Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
                result += data_(node_idx).dot(grid_vec.data_(node_idx));
            }
        }
    }
    catch (std::bad_cast& e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
    return result;
}

template <typename Scalar, int Dim>
const VectorND<Scalar>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator[] (const Vector<unsigned int, Dim> &idx) const
{
    return data_(idx);
}

template <typename Scalar, int Dim>
VectorND<Scalar>& UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::operator[] (const Vector<unsigned int, Dim> &idx)
{
    return data_(idx);
}

template <typename Scalar, int Dim>
void UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::setValue(const VectorND<Scalar> &value)
{
    if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
    {
        for (typename ArrayND<VectorND<Scalar>, Dim>::Iterator iter = data_.begin(); iter != data_.end(); ++iter)
            *iter = value;
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
            data_(node_idx) = value;
        }
    }
}

template <typename Scalar, int Dim>
void UniformGridGeneralizedVector <VectorND<Scalar>, Dim>::setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes)
{
    active_node_idx_ = active_grid_nodes;
    sortActiveNodes();
}

template <typename Scalar, int Dim>
UniformGridGeneralizedVector<Scalar,Dim> UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::vectorAtDim(unsigned int val_dim_idx) const
{
    if (val_dim_idx >= valueDim())
        throw PhysikaException("Element dimension out of range!");
    UniformGridGeneralizedVector<Scalar, Dim> vec(data_.size(), active_node_idx_);
    if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
    {
        for (typename ArrayND<VectorND<Scalar>, Dim>::ConstIterator iter = data_.begin(); iter != data_.end(); ++iter)
        {
            Vector<unsigned int, Dim> node_idx = iter.elementIndex();
            vec[node_idx] = data_(node_idx)[val_dim_idx];
        }
    }
    else
    {
        for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
        {
            Vector<unsigned int, Dim> node_idx = active_node_idx_[i];
            vec[node_idx] = data_(node_idx)[val_dim_idx];
        }
    }
    return vec;
}

template <typename Scalar, int Dim>
unsigned int UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::valueDim() const
{
    if (active_node_idx_.empty())
        return 0;
    else //assume all entries of same dimension
        return data_(active_node_idx_[0]).dims();
}

template <typename Scalar, int Dim>
void UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& grid_vec = dynamic_cast<const UniformGridGeneralizedVector<VectorND<Scalar>, Dim>&>(vector);
        data_ = grid_vec.data_;
        active_node_idx_ = grid_vec.active_node_idx_;
    }
    catch (std::bad_cast& e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
bool UniformGridGeneralizedVector<VectorND<Scalar>, Dim > ::checkActivePattern(const UniformGridGeneralizedVector<VectorND<Scalar>, Dim> &vector) const
{
    if (active_node_idx_.size() != vector.active_node_idx_.size())
        return false;
    else
    {
        if (active_node_idx_.size() == data_.totalElementCount()) // all entries active
            return true;
        else
        {
            //active node idx is in order
            for (unsigned int i = 0; i < active_node_idx_.size(); ++i)
                if (active_node_idx_[i] != vector.active_node_idx_[i])
                    return false;
            return true;
        }
    }
}

template <typename Scalar, int Dim>
void UniformGridGeneralizedVector<VectorND<Scalar>, Dim>::sortActiveNodes()
{
    if (active_node_idx_.size() != data_.totalElementCount())
    {
        //helper struct for sortActiveNodes
        struct CompareVector
        {
            UniformGridGeneralizedVector<VectorND<Scalar>, Dim> *master;
            bool operator()(const Vector<unsigned int, Dim> &vec1, const Vector<unsigned int, Dim> &vec2) const
            {
                Vector<unsigned int, Dim> node_num = (master->data_).size();
                Vector<unsigned int, Dim> tmp_vec1 = vec1, tmp_vec2 = vec2;
                unsigned int flat_vec1 = 0, flat_vec2 = 0;
                for (unsigned int i = 0; i < Dim; ++i)
                {
                    for (unsigned int j = i + 1; j < Dim; ++j)
                    {
                        tmp_vec1[i] *= node_num[j];
                        tmp_vec2[i] *= node_num[j];
                    }
                    flat_vec1 += tmp_vec1[i];
                    flat_vec2 += tmp_vec2[i];
                }
                return flat_vec1 < flat_vec2;
            }
        };
        CompareVector compare_vector;
        compare_vector.master = this;
        std::sort(active_node_idx_.begin(), active_node_idx_.end(), compare_vector);
    }
}

//explicit instantiations
template class UniformGridGeneralizedVector < Vector<float, 1>, 2 > ;
template class UniformGridGeneralizedVector < Vector<float, 2>, 2 > ;
template class UniformGridGeneralizedVector < Vector<float, 3>, 2 > ;
template class UniformGridGeneralizedVector < Vector<float, 4>, 2 > ;
template class UniformGridGeneralizedVector < Vector<float, 2>, 3 > ;
template class UniformGridGeneralizedVector < Vector<float, 3>, 3 > ;
template class UniformGridGeneralizedVector < Vector<float, 4>, 3 > ;
template class UniformGridGeneralizedVector < Vector<double, 1>, 2 > ;
template class UniformGridGeneralizedVector < Vector<double, 2>, 2 > ;
template class UniformGridGeneralizedVector < Vector<double, 3>, 2 > ;
template class UniformGridGeneralizedVector < Vector<double, 4>, 2 > ;
template class UniformGridGeneralizedVector < Vector<double, 2>, 3 > ;
template class UniformGridGeneralizedVector < Vector<double, 3>, 3 > ;
template class UniformGridGeneralizedVector < Vector<double, 4>, 3 > ;
template class UniformGridGeneralizedVector < VectorND<float>, 2 > ;
template class UniformGridGeneralizedVector < VectorND<float>, 3 > ;
template class UniformGridGeneralizedVector < VectorND<double>, 2 > ;
template class UniformGridGeneralizedVector < VectorND<double>, 3 > ;

}  //end of namespace Physika