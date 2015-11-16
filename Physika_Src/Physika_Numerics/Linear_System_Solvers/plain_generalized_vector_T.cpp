/*
* @file plain_generalized_vector_T.cpp
* @brief definition of PlainGeneralizedVector for template type float/double
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

#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector.h"

namespace Physika{

template <typename Scalar>
PlainGeneralizedVector<Scalar>::PlainGeneralizedVector()
    :GeneralizedVector<Scalar>(),data_()
{

}

template <typename Scalar>
PlainGeneralizedVector<Scalar>::PlainGeneralizedVector(unsigned int size)
    :GeneralizedVector<Scalar>(),data_(size)
{

}

template <typename Scalar>
PlainGeneralizedVector<Scalar>::PlainGeneralizedVector(const VectorND<Scalar> &vector)
    :GeneralizedVector<Scalar>(),data_(vector)
{

}

template <typename Scalar>
PlainGeneralizedVector<Scalar>::PlainGeneralizedVector(unsigned int size, Scalar value)
    :GeneralizedVector<Scalar>(),data_(size,value)
{

}

template <typename Scalar>
PlainGeneralizedVector<Scalar>::PlainGeneralizedVector(const PlainGeneralizedVector<Scalar> &vector)
    :GeneralizedVector<Scalar>(vector),data_(vector.data_)
{

}

template <typename Scalar>
PlainGeneralizedVector<Scalar>& PlainGeneralizedVector<Scalar>::operator= (const PlainGeneralizedVector<Scalar> &vector)
{
    data_ = vector.data_;
    return *this;
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>::~PlainGeneralizedVector()
{

}

template <typename Scalar>
Scalar& PlainGeneralizedVector<Scalar>::operator[] (unsigned int idx)
{
    return data_[idx];
}

template <typename Scalar>
const Scalar& PlainGeneralizedVector<Scalar>::operator[] (unsigned int idx) const
{
    return data_[idx];
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>* PlainGeneralizedVector<Scalar>::clone() const
{
    return new PlainGeneralizedVector<Scalar>(*this);
}

template <typename Scalar>
unsigned int PlainGeneralizedVector<Scalar>::size() const
{
    return data_.dims();
}

template <typename Scalar>
void PlainGeneralizedVector<Scalar>::resize(unsigned int new_size)
{
    data_.resize(new_size);
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>& PlainGeneralizedVector<Scalar>::operator+= (const GeneralizedVector<Scalar> &vector)
{
    try{
        const PlainGeneralizedVector<Scalar> &plain_vector = dynamic_cast<const PlainGeneralizedVector<Scalar>&>(vector);
        data_ += plain_vector.data_;
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>& PlainGeneralizedVector<Scalar>::operator-= (const GeneralizedVector<Scalar> &vector)
{
    try{
        const PlainGeneralizedVector<Scalar> &plain_vector = dynamic_cast<const PlainGeneralizedVector<Scalar>&>(vector);
        data_ -= plain_vector.data_;
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>& PlainGeneralizedVector<Scalar>:: operator*= (Scalar value)
{
    data_ *= value;
    return *this;
}

template <typename Scalar>
PlainGeneralizedVector<Scalar>& PlainGeneralizedVector<Scalar>:: operator/= (Scalar value)
{
    if (isEqual(value, static_cast<Scalar>(0.0)) == true)
        throw PhysikaException("divide by zero!");
    data_ /= value;
    return *this;
}

template <typename Scalar>
void PlainGeneralizedVector<Scalar>::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const PlainGeneralizedVector<Scalar> &plain_vector = dynamic_cast<const PlainGeneralizedVector<Scalar>&>(vector);
        data_ = plain_vector.data_;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
void PlainGeneralizedVector<Scalar>::streamInfo(std::ostream &s) const
{
    s<<data_;
}

//explicit instantiations
template class PlainGeneralizedVector<float>;
template class PlainGeneralizedVector<double>;

}  //end of namespace Physika
