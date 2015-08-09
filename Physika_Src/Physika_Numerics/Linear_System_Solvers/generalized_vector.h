/*
 * @file generalized_vector.h
 * @brief generalized vector class to represent solution x and right hand side b in a linear system
 *        Ax = b. The generalized vector class may be derived for specific linear system for convenience.
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

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_VECTOR_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_VECTOR_H_

namespace Physika{

/*
 * GeneralizedVector: base class of linear-system dependent vector types to represent x and b.
 * Derived GeneralizedVector class generally comes along with specific LinearSystem class.
 * A derived example:
 * For a dim-dimensional simulation, A is matrix of size dim*n*dim*n, x and b are vectors of size dim*n,
 * where n is the number of free nodes/particles. With GeneralizedMatrix and GeneralizedVector, the matrix could
 * be defined as matrix of size n*n and vector is a vector of size n whose elements are dim*1 vectors. In this
 * way, we save memory and ease coding by eliminating the need for flatting vectors.
 */

/*
 * RawScalar is defined for float and double
 * the type of raw data in the vector
 *
 * GeneralizedVector makes extensive use of polymorphism,
 * one pitfall of this design is that implementation of LinearSystemSolvers
 * becomes a little inconvenient since they don't know the dynamic type of
 * vectors and have to use pointers/references to enable polymorphism.
 */
template <typename RawScalar>
class GeneralizedVector
{
public:
    GeneralizedVector(){}
    GeneralizedVector(const GeneralizedVector<RawScalar> &vector){}
    virtual ~GeneralizedVector(){}
    GeneralizedVector<RawScalar>& operator= (const GeneralizedVector<RawScalar> &vector);
    //virtual methods
    virtual GeneralizedVector<RawScalar>* clone() const = 0; //create a clone, prototype design pattern
    virtual unsigned int size() const = 0; //number of elements in the vector
    virtual GeneralizedVector<RawScalar>& operator+= (const GeneralizedVector<RawScalar> &vector) = 0;
    virtual GeneralizedVector<RawScalar>& operator-= (const GeneralizedVector<RawScalar> &vector) = 0;
    virtual GeneralizedVector<RawScalar>& operator+= (RawScalar) = 0;
    virtual GeneralizedVector<RawScalar>& operator-= (RawScalar) = 0;
    virtual GeneralizedVector<RawScalar>& operator*= (RawScalar) = 0;
    virtual GeneralizedVector<RawScalar>& operator/= (RawScalar) = 0;
    virtual RawScalar norm() const = 0;
    virtual RawScalar normSquared() const = 0;
    virtual RawScalar dot(const GeneralizedVector<RawScalar> &vector) const = 0;
protected:
    //the actual engine of assign operator, so that assign operator needs not to be made virtual while polymorphism
    //is still enabled when calling assignment operator with pointer/reference to base class
    //subclass of GeneralizedVector implements this method and call this method assign operator implementation
    virtual void copy(const GeneralizedVector<RawScalar> &vector) = 0;
};

}  //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_VECTOR_H_
