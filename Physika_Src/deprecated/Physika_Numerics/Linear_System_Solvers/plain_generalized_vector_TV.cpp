/*
* @file plain_generalized_vector_TV.cpp
* @brief definition of PlainGeneralizedVector for template type Vector/VectorND
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
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector_TV.h"

namespace Physika{

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::PlainGeneralizedVector()
    :GeneralizedVector<Scalar>(), data_()
{

}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::PlainGeneralizedVector(unsigned int size)
    :GeneralizedVector<Scalar>(), data_(size)
{

}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::PlainGeneralizedVector(const std::vector<Vector<Scalar, Dim> > &vector)
    :GeneralizedVector<Scalar>(), data_(vector)
{

}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::PlainGeneralizedVector(unsigned int size, const Vector<Scalar, Dim> &value)
    :GeneralizedVector<Scalar>(), data_(size, value)
{

}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::PlainGeneralizedVector(const PlainGeneralizedVector<Vector<Scalar,Dim> > &vector)
    :GeneralizedVector<Scalar>(vector), data_(vector.data_)
{

}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator= (const PlainGeneralizedVector<Vector<Scalar, Dim> > &vector)
{
    data_ = vector.data_;
    return *this;
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >::~PlainGeneralizedVector()
{

}

template <typename Scalar, int Dim>
unsigned int PlainGeneralizedVector<Vector<Scalar, Dim> >::valueDim() const
{
    return Dim;
}

template <typename Scalar, int Dim>
Vector<Scalar, Dim>& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator[](unsigned int idx)
{
    if (idx >= data_.size())
        throw PhysikaException("index out of range!");
    return data_[idx];
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim>& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator[](unsigned int idx) const
{
    if (idx >= data_.size())
        throw PhysikaException("index out of range!");
    return data_[idx];
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >* PlainGeneralizedVector<Vector<Scalar, Dim> >::clone() const
{
    return new PlainGeneralizedVector<Vector<Scalar, Dim> >(*this);
}

template <typename Scalar, int Dim>
unsigned int PlainGeneralizedVector<Vector<Scalar, Dim> >::size() const
{
    return data_.size();
}

template <typename Scalar, int Dim>
void PlainGeneralizedVector<Vector<Scalar, Dim> >::resize(unsigned int new_size)
{
    data_.resize(new_size);
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator+=(const GeneralizedVector<Scalar> &vector)
{
    if (vector.size() != this->size())
        throw PhysikaException("PlainGeneralizedVector size mismatch!");
    try{
        const PlainGeneralizedVector<Vector<Scalar, Dim> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<Vector<Scalar, Dim> >&>(vector);
        for (unsigned int i = 0; i < data_.size(); ++i)
            data_[i] += plain_vector.data_[i];
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator-=(const GeneralizedVector<Scalar> &vector)
{
    if (vector.size() != this->size())
        throw PhysikaException("PlainGeneralizedVector size mismatch!");
    try{
        const PlainGeneralizedVector<Vector<Scalar, Dim> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<Vector<Scalar, Dim> >&>(vector);
        for (unsigned int i = 0; i < data_.size(); ++i)
            data_[i] -= plain_vector.data_[i];
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator*=(Scalar value)
{
    for (unsigned int i = 0; i < data_.size(); ++i)
        data_[i] *= value;
    return *this;
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Vector<Scalar, Dim> >& PlainGeneralizedVector<Vector<Scalar, Dim> >::operator/=(Scalar value)
{
    if (isEqual(value, static_cast<Scalar>(0.0)) == true)
        throw PhysikaException("divide by zero!");
    for (unsigned int i = 0; i < data_.size(); ++i)
        data_[i] /= value;
    return *this;
}

template <typename Scalar, int Dim>
PlainGeneralizedVector<Scalar> PlainGeneralizedVector<Vector<Scalar, Dim> >::vectorAtDim(unsigned int val_dim_idx) const
{
    if (val_dim_idx >= Dim)
        throw PhysikaException("PlainGeneralizedVector value dimension index out of range!");
    PlainGeneralizedVector<Scalar> result(data_.size());
    for (unsigned int i = 0; i < data_.size(); ++i)
        result[i] = data_[i][val_dim_idx];
    return result;
}

template <typename Scalar, int Dim>
void PlainGeneralizedVector<Vector<Scalar, Dim> >::streamInfo(std::ostream &s) const
{
    s << "[";
    for (unsigned int i = 0; i < data_.size(); ++i)
    {
        s << data_[i];
        if(i<data_.size()-1)
            s<< ", ";
    }
    s << "]";
}

template <typename Scalar, int Dim>
void PlainGeneralizedVector<Vector<Scalar, Dim> >::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const PlainGeneralizedVector<Vector<Scalar, Dim> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<Vector<Scalar, Dim> >&>(vector);
        data_ = plain_vector.data_;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

/////////////////////////////////////////////////////////////////////////////


template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::PlainGeneralizedVector()
    :GeneralizedVector<Scalar>(), data_()
{

}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::PlainGeneralizedVector(unsigned int size)
    :GeneralizedVector<Scalar>(), data_(size)
{

}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::PlainGeneralizedVector(const std::vector<VectorND<Scalar> > &vector)
    :GeneralizedVector<Scalar>(), data_(vector)
{

}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::PlainGeneralizedVector(unsigned int size, const VectorND<Scalar> &value)
    :GeneralizedVector<Scalar>(), data_(size, value)
{

}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::PlainGeneralizedVector(const PlainGeneralizedVector<VectorND<Scalar> > &vector)
    :GeneralizedVector<Scalar>(vector), data_(vector.data_)
{

}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >& PlainGeneralizedVector<VectorND<Scalar> >::operator= (const PlainGeneralizedVector<VectorND<Scalar> > &vector)
{
    data_ = vector.data_;
    return *this;
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >::~PlainGeneralizedVector()
{

}

template <typename Scalar>
unsigned int PlainGeneralizedVector<VectorND<Scalar> >::valueDim() const
{
    if (data_.empty())
        return 0;
    else //assume all elements are the same dimension
        return data_[0].dims();
}

template <typename Scalar>
VectorND<Scalar>& PlainGeneralizedVector<VectorND<Scalar> >::operator[](unsigned int idx)
{
    if (idx >= data_.size())
        throw PhysikaException("index out of range!");
    return data_[idx];
}

template <typename Scalar>
const VectorND<Scalar>& PlainGeneralizedVector<VectorND<Scalar> >::operator[](unsigned int idx) const
{
    if (idx >= data_.size())
        throw PhysikaException("index out of range!");
    return data_[idx];
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >* PlainGeneralizedVector<VectorND<Scalar> >::clone() const
{
    return new PlainGeneralizedVector<VectorND<Scalar> >(*this);
}

template <typename Scalar>
unsigned int PlainGeneralizedVector<VectorND<Scalar> >::size() const
{
    return data_.size();
}

template <typename Scalar>
void PlainGeneralizedVector<VectorND<Scalar> >::resize(unsigned int new_size)
{
    data_.resize(new_size);
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >& PlainGeneralizedVector<VectorND<Scalar> >::operator+=(const GeneralizedVector<Scalar> &vector)
{
    if (vector.size() != this->size())
        throw PhysikaException("PlainGeneralizedVector size mismatch!");
    try{
        const PlainGeneralizedVector<VectorND<Scalar> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<VectorND<Scalar> >&>(vector);
        for (unsigned int i = 0; i < data_.size(); ++i)
            data_[i] += plain_vector.data_[i];
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >& PlainGeneralizedVector<VectorND<Scalar> >::operator-=(const GeneralizedVector<Scalar> &vector)
{
    if (vector.size() != this->size())
        throw PhysikaException("PlainGeneralizedVector size mismatch!");
    try{
        const PlainGeneralizedVector<VectorND<Scalar> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<VectorND<Scalar> >&>(vector);
        for (unsigned int i = 0; i < data_.size(); ++i)
            data_[i] -= plain_vector.data_[i];
        return *this;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >& PlainGeneralizedVector<VectorND<Scalar> >::operator*=(Scalar value)
{
    for (unsigned int i = 0; i < data_.size(); ++i)
        data_[i] *= value;
    return *this;
}

template <typename Scalar>
PlainGeneralizedVector<VectorND<Scalar> >& PlainGeneralizedVector<VectorND<Scalar> >::operator/=(Scalar value)
{
    if (isEqual(value, static_cast<Scalar>(0.0)) == true)
        throw PhysikaException("divide by zero!");
    for (unsigned int i = 0; i < data_.size(); ++i)
        data_[i] /= value;
    return *this;
}

template <typename Scalar>
Scalar PlainGeneralizedVector<VectorND<Scalar> >::norm() const
{
    return sqrt(normSquared());
}

template <typename Scalar>
Scalar PlainGeneralizedVector<VectorND<Scalar> >::normSquared() const
{
    Scalar norm_sqr = 0.0;
    for (unsigned int i = 0; i < data_.size(); ++i)
        norm_sqr += data_[i].normSquared();
    return norm_sqr;
}

template <typename Scalar>
Scalar PlainGeneralizedVector<VectorND<Scalar> >::dot(const GeneralizedVector<Scalar> &vector) const
{
    if (vector.size() != this->size())
        throw PhysikaException("PlainGeneralizedVector size mismatch!");
    try{
        const PlainGeneralizedVector<VectorND<Scalar> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<VectorND<Scalar> >&>(vector);
        Scalar result = 0.0;
        for (unsigned int i = 0; i < data_.size(); ++i)
            result += data_[i].dot(plain_vector[i]);
        return result;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

template <typename Scalar>
PlainGeneralizedVector<Scalar> PlainGeneralizedVector<VectorND<Scalar> >::vectorAtDim(unsigned int val_dim_idx) const
{
    if (val_dim_idx >= valueDim())
        throw PhysikaException("PlainGeneralizedVector value dimension index out of range!");
    PlainGeneralizedVector<Scalar> result(data_.size());
    for (unsigned int i = 0; i < data_.size(); ++i)
        result[i] = data_[i][val_dim_idx];
    return result;
}

template <typename Scalar>
void PlainGeneralizedVector<VectorND<Scalar> >::streamInfo(std::ostream &s) const
{
    s << "[";
    for (unsigned int i = 0; i < data_.size(); ++i)
    {
        s << data_[i];
        if (i < data_.size() - 1)
            s << ", ";
    }
    s << "]";
}

template <typename Scalar>
void PlainGeneralizedVector<VectorND<Scalar> >::copy(const GeneralizedVector<Scalar> &vector)
{
    try{
        const PlainGeneralizedVector<VectorND<Scalar> > &plain_vector = dynamic_cast<const PlainGeneralizedVector<VectorND<Scalar> >&>(vector);
        data_ = plain_vector.data_;
    }
    catch (std::bad_cast &e)
    {
        throw PhysikaException("Incorrect argument type!");
    }
}

//explicit instantiations
template class PlainGeneralizedVector < Vector<float, 1> >;
template class PlainGeneralizedVector < Vector<float, 2> >;
template class PlainGeneralizedVector < Vector<float, 3> >;
template class PlainGeneralizedVector < Vector<float, 4> >;
template class PlainGeneralizedVector < Vector<double, 1> >;
template class PlainGeneralizedVector < Vector<double, 2> >;
template class PlainGeneralizedVector < Vector<double, 3> >;
template class PlainGeneralizedVector < Vector<double, 4> >;
template class PlainGeneralizedVector < VectorND<float> >;
template class PlainGeneralizedVector < VectorND<double> >;

} //end of namespace Physika