/*
 * @file mpm_uniform_grid_generalized_vector.h
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_

#include <vector>
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_TV.h"

namespace Physika{

/*
 * MPMUniformGridGeneralizedVector: a special UniformGridGeneralizedVector whose inner-product
 * is scaled with mass at corresponding node
 * the element is Vector<Scalar,Dim> where Dim = 2/3 
 */

//default template, constructor made private to prohibit instance
template <typename Scalar>
class MPMUniformGridGeneralizedVector
{
private:
    MPMUniformGridGeneralizedVector();
};

//partial specialization for Vector<Scalar,Dim>
template <typename Scalar, int Dim>
class MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >: public UniformGridGeneralizedVector<Vector<Scalar,Dim>,Dim>
{
public:
    //all grid nodes active, node mass set to 1
    explicit MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size);
    MPMUniformGridGeneralizedVector(const Vector<unsigned int,Dim> &grid_size,
                                    const std::vector<Vector<unsigned int,Dim> > &active_grid_nodes,
                                    const std::vector<Scalar> &active_node_mass);
    MPMUniformGridGeneralizedVector(const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &vector);
    ~MPMUniformGridGeneralizedVector();
    MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >& operator= (const MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> > &vector);
    virtual MPMUniformGridGeneralizedVector<Vector<Scalar,Dim> >* clone() const;
    virtual Scalar norm() const;
    virtual Scalar normSquared() const;
    virtual Scalar dot(const GeneralizedVector<Scalar> &vector) const;
    void setActiveNodeMass(const std::vector<Scalar> &active_node_mass);
protected:
    virtual void copy(const GeneralizedVector<Scalar> &vector);
protected:
    std::vector<Scalar> active_node_mass_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_UNIFORM_GRID_GENERALIZED_VECTOR_H_
