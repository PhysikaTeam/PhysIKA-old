/*
 * @file deformation_diagonalization.h 
 * @brief diagonalization of the deformation gradient via a modified SVD,
 *        implementation of the technique proposed in the sca04 paper:
 *        <Invertible Finite Elements For Robust Simulation of Large Deformation>.
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

#ifndef PHYSIKA_DYNAMICS_UTILITIES_DEFORMATION_DIAGONALIZATION_DEFORMATION_DIAGONALIZATION_H_
#define PHYSIKA_DYNAMICS_UTILITIES_DEFORMATION_DIAGONALIZATION_DEFORMATION_DIAGONALIZATION_H_

#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"

namespace Physika{

template <typename Scalar, int Dim>
class DeformationDiagonalization
{
public:
    //type definition, for convenience
    typedef struct{
        SquareMatrix<Scalar,Dim> left_rotation, diag_deform_grad, right_rotation;
    }DiagonalizedDeformation;
public:
    DeformationDiagonalization();  //initialize with default epsilon
    explicit DeformationDiagonalization(Scalar epsilon); //initialize with given epsilon
    ~DeformationDiagonalization();
    Scalar epsilon() const;
    void setEpsilon(Scalar epsilon);
	void diagonalizeDeformationGradient(const SquareMatrix<Scalar,Dim> &deform_grad, SquareMatrix<Scalar,Dim> &left_rotation,
		                                SquareMatrix<Scalar,Dim> &diag_deform_grad, SquareMatrix<Scalar,Dim> &right_rotation) const;
    void diagonalizeDeformationGradient(const SquareMatrix<Scalar,Dim> &deform_grad, DiagonalizedDeformation &diagonalized_deformation) const;
protected:
    //trait method for different dimension
	void diagonalizationTrait(const SquareMatrix<Scalar,2> &deform_grad, SquareMatrix<Scalar,2> &left_rotation,
                              SquareMatrix<Scalar,2> &diag_deform_grad, SquareMatrix<Scalar,2> &right_rotation) const;
	void diagonalizationTrait(const SquareMatrix<Scalar,3> &deform_grad, SquareMatrix<Scalar,3> &left_rotation,
                              SquareMatrix<Scalar,3> &diag_deform_grad, SquareMatrix<Scalar,3> &right_rotation) const;
protected:
    Scalar epsilon_; //the epsilon value used to determine if some value is close to zero
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_UTILITIES_DEFORMATION_DIAGONALIZATION_DEFORMATION_DIAGONALIZATION_H_
