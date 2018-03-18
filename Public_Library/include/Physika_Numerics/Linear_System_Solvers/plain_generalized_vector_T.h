/*
* @file plain_generalized_vector_T.h
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

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_T_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_T_H_

#include <iostream>
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"

namespace Physika{

template <typename Scalar>
class PlainGeneralizedVector: public GeneralizedVector<Scalar>
{
public:
    PlainGeneralizedVector();
    explicit PlainGeneralizedVector(unsigned int size);
    explicit PlainGeneralizedVector(const VectorND<Scalar> &vector); //initialize with VectorND
    PlainGeneralizedVector(unsigned int size, Scalar value);
    PlainGeneralizedVector(const PlainGeneralizedVector<Scalar> &vector);
    PlainGeneralizedVector<Scalar>& operator= (const PlainGeneralizedVector<Scalar> &vector);
    virtual ~PlainGeneralizedVector();

    //[] operator to get the elements
    Scalar& operator[] (unsigned int idx);
    const Scalar& operator[] (unsigned int idx) const;

    virtual PlainGeneralizedVector<Scalar>* clone() const;
    virtual unsigned int size() const;
    void resize(unsigned int new_size);
    //derived virtual methods
    //use the same parameter type and covariant return type compared to declaration in base class
    virtual PlainGeneralizedVector<Scalar>& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<Scalar>& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<Scalar>& operator*= (Scalar);
    virtual PlainGeneralizedVector<Scalar>& operator/= (Scalar);

    //advance accessors
    inline const VectorND<Scalar>& rawVector() const {return data_;}
    inline VectorND<Scalar>& rawVector() {return data_;}

    void streamInfo(std::ostream &s) const; //insert info into stream
protected:
    virtual void copy(const GeneralizedVector<Scalar> &vector);
protected:
    VectorND<Scalar> data_;
};

//override << for PlainGeneralizedVector
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const PlainGeneralizedVector<Scalar> &vec)
{
    vec.streamInfo(s);
    return s;
}

}  //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_T_H_
