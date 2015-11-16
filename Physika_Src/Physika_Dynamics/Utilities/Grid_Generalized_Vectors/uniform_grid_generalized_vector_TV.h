/*
* @file uniform_grid_generalized_vector_TV.h
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

#ifndef PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_TV_H_
#define PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_TV_H_

#include <vector>
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_vector.h"
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_T.h"

namespace Physika{

/*
 * partial specialization for Vector<Scalar,Dim> element type
 */

template <typename Scalar, int GridDim, int EleDim>
class UniformGridGeneralizedVector<Vector<Scalar,EleDim>,GridDim>: public GeneralizedVector<Scalar>
{
public:
    explicit UniformGridGeneralizedVector(const Vector<unsigned int, GridDim> &grid_size); //all grid nodes are active
    UniformGridGeneralizedVector(const Vector<unsigned int, GridDim> &grid_size, const std::vector<Vector<unsigned int, GridDim> > &active_grid_nodes);
    UniformGridGeneralizedVector(const UniformGridGeneralizedVector<Vector<Scalar,EleDim>,GridDim> &vector);
    ~UniformGridGeneralizedVector();
    UniformGridGeneralizedVector<Vector<Scalar, EleDim>,GridDim>& operator= (const UniformGridGeneralizedVector<Vector<Scalar, EleDim>,GridDim> &vector);
    //virtual methods
    virtual UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>* clone() const;
    virtual unsigned int size() const; //the actual number of valid elements in the vector
    virtual UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& operator*= (Scalar);
    virtual UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim>& operator/= (Scalar);
    //setters && getters
    const Vector<Scalar, EleDim>& operator[](const Vector<unsigned int, GridDim> &idx) const;
    Vector<Scalar, EleDim>& operator[](const Vector<unsigned int, GridDim> &idx);
    void setValue(const Vector<Scalar, EleDim>& value); //set one value for all entries
    void setActivePattern(const std::vector<Vector<unsigned int, GridDim> > &active_grid_nodes);
    //wrap each dimension of vector value into a grid vector
    UniformGridGeneralizedVector<Scalar, GridDim> vectorAtDim(unsigned int val_dim_idx) const;
    unsigned int valueDim() const;
protected:
    UniformGridGeneralizedVector(); //default constructor made protected
    virtual void copy(const GeneralizedVector<Scalar> &vector);
    bool checkActivePattern(const UniformGridGeneralizedVector<Vector<Scalar, EleDim>, GridDim> &vector) const; //check if active grid node pattern matches
    void sortActiveNodes(); //sort the active node index in ascending order
protected:
    ArrayND<Vector<Scalar,EleDim>, GridDim> data_;
    std::vector<Vector<unsigned int, GridDim> > active_node_idx_; //the portion of entries in data_ that are active
};

/*
 * partial specialization for VectorND<Scalar> element type
 * all entries should be of the same dimension
 */
template <typename Scalar, int Dim>
class UniformGridGeneralizedVector<VectorND<Scalar>, Dim> : public GeneralizedVector < Scalar >
{
public:
    explicit UniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size); //all grid nodes are active
    UniformGridGeneralizedVector(const Vector<unsigned int, Dim> &grid_size, const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes);
    UniformGridGeneralizedVector(const UniformGridGeneralizedVector<VectorND<Scalar>, Dim> &vector);
    ~UniformGridGeneralizedVector();
    UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& operator= (const UniformGridGeneralizedVector<VectorND<Scalar>, Dim> &vector);
    //virtual methods
    virtual UniformGridGeneralizedVector<VectorND<Scalar>, Dim>* clone() const;
    virtual unsigned int size() const; //the actual number of valid elements in the vector
    virtual UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& operator+= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& operator-= (const GeneralizedVector<Scalar> &vector);
    virtual UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& operator*= (Scalar);
    virtual UniformGridGeneralizedVector<VectorND<Scalar>, Dim>& operator/= (Scalar);
    virtual Scalar norm() const;
    virtual Scalar normSquared() const;
    virtual Scalar dot(const GeneralizedVector<Scalar> &vector) const;
    //setters && getters
    const VectorND<Scalar>& operator[](const Vector<unsigned int, Dim> &idx) const;
    VectorND<Scalar>& operator[](const Vector<unsigned int, Dim> &idx);
    void setValue(const VectorND<Scalar>& value); //set one value for all entries
    void setActivePattern(const std::vector<Vector<unsigned int, Dim> > &active_grid_nodes);
    //wrap each dimension of vector value into a grid vector
    UniformGridGeneralizedVector<Scalar, Dim> vectorAtDim(unsigned int val_dim_idx) const;
    unsigned int valueDim() const;
protected:
    UniformGridGeneralizedVector(); //default constructor made protected
    virtual void copy(const GeneralizedVector<Scalar> &vector);
    bool checkActivePattern(const UniformGridGeneralizedVector<VectorND<Scalar>, Dim> &vector) const; //check if active grid node pattern matches
    void sortActiveNodes(); //sort the active node index in ascending order
protected:
    ArrayND<VectorND<Scalar>, Dim> data_;
    std::vector<Vector<unsigned int, Dim> > active_node_idx_; //the portion of entries in data_ that are active
};
}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_UTILITIES_GRID_GENERALIZED_VECTORS_UNIFORM_GRID_GENERALIZED_VECTOR_TV_H_