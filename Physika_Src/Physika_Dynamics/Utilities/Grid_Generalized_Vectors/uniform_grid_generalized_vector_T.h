/*
 * @file uniform_grid_generalized_vector_T.h
 * @brief generalized vector for solving the linear system Ax = b on uniform grid
 *        defined for float/double element type
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

#ifndef PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_T_H_
#define PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_T_H_

#include <vector>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * UniformGridGeneralizedVector: for linear system Ax = b on uniform cartesian grids, this
 * generalized vector represents the x and b. It's a high level vector whose entries can be accessed via grid node
 * index.
 * UniformGridGeneralizedVector can be viewed as a wrraper of ArrayND, only that some entries can be inactive
 */

template <typename Scalar, int Dim>
class UniformGridGeneralizedVector: public GeneralizedVector<Scalar>
{
public:
    explicit UniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size); //all grid nodes are active
    UniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size, const std::vector<Vector<unsigned int,Dim> > &active_grid_nodes);
    UniformGridGeneralizedVector(const UniformGridGeneralizedVector<Scalar,Dim> &vector);
    ~UniformGridGeneralizedVector();
    UniformGridGeneralizedVector<Scalar,Dim> & operator= (const UniformGridGeneralizedVector<Scalar,Dim> &vector);
    //virtual methods
    virtual UniformGridGeneralizedVector<Scalar,Dim>* clone() const;
    virtual unsigned int size() const; //the actual number of valid elements in the vector
    virtual UniformGridGeneralizedVector<Scalar,Dim>& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<Scalar,Dim>& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<Scalar,Dim>& operator*= (Scalar);
    virtual UniformGridGeneralizedVector<Scalar, Dim>& operator/= (Scalar);
    //setters && getters
    const Scalar& operator[](const Vector<unsigned int,Dim> &idx) const;
    Scalar& operator[](const Vector<unsigned int,Dim> &idx);
    void setValue(Scalar value); //set one value for all entries
    void setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes);
    bool checkActivePattern(const UniformGridGeneralizedVector<Scalar, Dim> &vector) const; //check if active grid node pattern matches
protected:
    UniformGridGeneralizedVector(); //default constructor made protected
    virtual void copy(const GeneralizedVector<Scalar> &vector);
    void sortActiveNodes(); //sort the active node index in ascending order
protected:
    ArrayND<Scalar,Dim> data_;
    std::vector<Vector<unsigned int,Dim> > active_node_idx_; //the portion of entries in data_ that are active
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_T_H_
