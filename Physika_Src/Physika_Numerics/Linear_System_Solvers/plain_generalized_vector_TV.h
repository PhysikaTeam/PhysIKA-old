/*
* @file plain_generalized_vector_TV.h
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

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_TV_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_TV_H_

#include <iostream>
#include <vector>
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector_T.h"

namespace Physika{

/*
 * partial specialization for Vector element type
 */
template <typename Scalar, int Dim>
class PlainGeneralizedVector<Vector<Scalar,Dim> >: public GeneralizedVector<Scalar>
{
public:
    PlainGeneralizedVector();
    explicit PlainGeneralizedVector(unsigned int size);
    //initialize with a vector of Vector<Scalar,Dim>
    explicit PlainGeneralizedVector(const std::vector<Vector<Scalar, Dim> > &vector);
    PlainGeneralizedVector(unsigned int size, const Vector<Scalar, Dim> &value);
    PlainGeneralizedVector(const PlainGeneralizedVector<Vector<Scalar, Dim> > &vector);
    PlainGeneralizedVector<Vector<Scalar, Dim> >& operator= (const PlainGeneralizedVector<Vector<Scalar, Dim> > &vector);
    virtual ~PlainGeneralizedVector();

    unsigned int valueDim() const; //return dimension of each element
    //[] operator to get the element
    Vector<Scalar, Dim>& operator[] (unsigned int idx);
    const Vector<Scalar, Dim>& operator[] (unsigned int idx) const;

    virtual PlainGeneralizedVector<Vector<Scalar, Dim> >* clone() const;
    virtual unsigned int size() const;
    void resize(unsigned int new_size);
    //derived virtual methods
    //use the same parameter type and covariant return type compared to declaration in base class
    virtual PlainGeneralizedVector<Vector<Scalar,Dim> >& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<Vector<Scalar, Dim> >& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<Vector<Scalar, Dim> >& operator*= (Scalar);
    virtual PlainGeneralizedVector<Vector<Scalar, Dim> >& operator/= (Scalar);
    //wrap each dimension of values into a vector
    PlainGeneralizedVector<Scalar> vectorAtDim(unsigned int val_dim_idx) const;

    void streamInfo(std::ostream &s) const; //insert info into stream
protected:
    virtual void copy(const GeneralizedVector<Scalar> &vector);
protected:
    std::vector<Vector<Scalar, Dim> > data_;
};
//override << for PlainGeneralizedVector
template <typename Scalar, int Dim>
inline std::ostream& operator<< (std::ostream &s, const PlainGeneralizedVector<Vector<Scalar,Dim> > &vec)
{
    vec.streamInfo(s);
    return s;
}

/*
* partial specialization for VectorND element type
* all elements should be the same dimension
*/
template <typename Scalar>
class PlainGeneralizedVector<VectorND<Scalar> > : public GeneralizedVector<Scalar>
{
public:
    PlainGeneralizedVector();
    explicit PlainGeneralizedVector(unsigned int size);
    //initialize with a vector of VectorND
    explicit PlainGeneralizedVector(const std::vector<VectorND<Scalar> > &vector); 
    PlainGeneralizedVector(unsigned int size, const VectorND<Scalar> &value);
    PlainGeneralizedVector(const PlainGeneralizedVector<VectorND<Scalar> > &vector);
    PlainGeneralizedVector<VectorND<Scalar> >& operator= (const PlainGeneralizedVector<VectorND<Scalar> > &vector);
    virtual ~PlainGeneralizedVector();

    unsigned int valueDim() const; //return dimension of each element
    //[] operator to get the element
    VectorND<Scalar>& operator[] (unsigned int idx);
    const VectorND<Scalar>& operator[] (unsigned int idx) const;

    virtual PlainGeneralizedVector<VectorND<Scalar> >* clone() const;
    virtual unsigned int size() const;
    void resize(unsigned int new_size);
    //derived virtual methods
    //use the same parameter type and covariant return type compared to declaration in base class
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator+= (Scalar);
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator-= (Scalar);
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator*= (Scalar);
    virtual PlainGeneralizedVector<VectorND<Scalar> >& operator/= (Scalar);
    virtual Scalar norm() const;
    virtual Scalar normSquared() const;
    virtual Scalar dot(const GeneralizedVector<Scalar> &vector) const;
    //wrap each dimension of values into a vector
    PlainGeneralizedVector<Scalar> vectorAtDim(unsigned int val_dim_idx) const;

    void streamInfo(std::ostream &s) const; //insert info into stream
protected:
    virtual void copy(const GeneralizedVector<Scalar> &vector);
protected:
    std::vector<VectorND<Scalar> > data_;
};
//override << for PlainGeneralizedVector
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const PlainGeneralizedVector<VectorND<Scalar> > &vec)
{
    vec.streamInfo(s);
    return s;
}

}  //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_PLAIN_GENERALIZED_VECTOR_TV_H_