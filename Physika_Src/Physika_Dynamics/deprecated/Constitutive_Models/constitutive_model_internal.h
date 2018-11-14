/*
 * @file constitutive_model_internal.h
 * @brief common structures and methods used by the constitutive models
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_INTERNAL_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_INTERNAL_H_

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

namespace ConstitutiveModelInternal{

/*
 * methods for computing differentials
 * implicit time integration requires computing differentials of stress
 * the ***Differentials() methods are helper methods
 */

//differential of J*F.inverse().transpose(), defined for float/double
//useful for models that involve J in energy density
//derivative of J with respect to F is J*F.inverse().transpose() and implicit time integration
//requires differentiate again, where this method will be of use
template <typename Scalar>
SquareMatrix<Scalar,2> cofactorMatrixDifferential(const SquareMatrix<Scalar,2> &F,
      const SquareMatrix<Scalar,2> &F_differential);
template <typename Scalar>
SquareMatrix<Scalar,3> cofactorMatrixDifferential(const SquareMatrix<Scalar,3> &F,
      const SquareMatrix<Scalar,3> &F_differential);

//differential of rotation part of F
//useful for fixed corotated constitutive models
//F and R are given
template <typename Scalar>
SquareMatrix<Scalar,2> rotationDifferential(const SquareMatrix<Scalar,2> &F,
                       const SquareMatrix<Scalar,2> &R,
                       const SquareMatrix<Scalar,2> &F_differential);
template <typename Scalar>
SquareMatrix<Scalar,3> rotationDifferential(const SquareMatrix<Scalar,3> &F,
                       const SquareMatrix<Scalar,3> &R,
                       const SquareMatrix<Scalar,3> &F_differential);

} //end of namespace ConstitutiveModelInternal

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_INTERNAL_H_
