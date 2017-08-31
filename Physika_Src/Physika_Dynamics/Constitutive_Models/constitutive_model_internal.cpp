/*
 * @file constitutive_model_internal.cpp
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/constitutive_model_internal.h"

namespace Physika{

namespace ConstitutiveModelInternal{

template <typename Scalar>
SquareMatrix<Scalar,2> cofactorMatrixDifferential(const SquareMatrix<Scalar,2> &F,
                                                  const SquareMatrix<Scalar,2> &F_differential)
{
    return SquareMatrix<Scalar,2>(F_differential(1,1),-F_differential(1,0),-F_differential(0,1),F_differential(0,0));
}
template <typename Scalar>
SquareMatrix<Scalar,3> cofactorMatrixDifferential(const SquareMatrix<Scalar,3> &F,
                                                  const SquareMatrix<Scalar,3> &F_differential)
{
    return SquareMatrix<Scalar,3>(F_differential(1,1)*F(2,2)+F(1,1)*F_differential(2,2)-F_differential(2,1)*F(1,2)-F(2,1)*F_differential(1,2),
                                  F_differential(2,0)*F(1,2)+F(2,0)*F_differential(1,2)-F_differential(1,0)*F(2,2)-F(1,0)*F_differential(2,2),
                                  F_differential(1,0)*F(2,1)+F(1,0)*F_differential(2,1)-F_differential(2,0)*F(1,1)-F(2,0)*F_differential(1,1),
                                  F_differential(2,1)*F(0,2)+F(2,1)*F_differential(0,2)-F_differential(0,1)*F(2,2)-F(0,1)*F_differential(2,2),
                                  F_differential(0,0)*F(2,2)+F(0,0)*F_differential(2,2)-F_differential(2,0)*F(0,2)-F(2,0)*F_differential(0,2),
                                  F_differential(2,0)*F(0,1)+F(2,0)*F_differential(0,1)-F_differential(0,0)*F(2,1)-F(0,0)*F_differential(2,1),
                                  F_differential(0,1)*F(1,2)+F(0,1)*F_differential(1,2)-F_differential(1,1)*F(0,2)-F(1,1)*F_differential(0,2),
                                  F_differential(1,0)*F(0,2)+F(1,0)*F_differential(0,2)-F_differential(0,0)*F(1,2)-F(0,0)*F_differential(1,2),
                                  F_differential(0,0)*F(1,1)+F(0,0)*F_differential(1,1)-F_differential(1,0)*F(0,1)-F(1,0)*F_differential(0,1));
}
//explicit instantiations
template SquareMatrix<float,2> cofactorMatrixDifferential<float>(const SquareMatrix<float,2> &F,
                                                                 const SquareMatrix<float,2> &F_differential);
template SquareMatrix<double,2> cofactorMatrixDifferential<double>(const SquareMatrix<double,2> &F,
                                                                 const SquareMatrix<double,2> &F_differential);
template SquareMatrix<float,3> cofactorMatrixDifferential<float>(const SquareMatrix<float,3> &F,
                                                                 const SquareMatrix<float,3> &F_differential);
template SquareMatrix<double,3> cofactorMatrixDifferential<double>(const SquareMatrix<double,3> &F,
                                                                  const SquareMatrix<double,3> &F_differential);


template <typename Scalar>
SquareMatrix<Scalar,2> rotationDifferential(const SquareMatrix<Scalar,2> &F,
                                            const SquareMatrix<Scalar,2> &R,
                                            const SquareMatrix<Scalar,2> &F_differential)
{
    SquareMatrix<Scalar,2> M = R.transpose()*F_differential;
    SquareMatrix<Scalar,2> S = R.transpose()*F;
    if(S(1,1)+S(0,0)<std::numeric_limits<Scalar>::epsilon())
      return SquareMatrix<Scalar,2>(0);
    Scalar a = (M(0,1)-M(1,0))/(S(1,1)+S(0,0));
    SquareMatrix<Scalar,2> RTdR(0,a,-a,0);
    return R*RTdR;
}
template <typename Scalar>
SquareMatrix<Scalar,3> rotationDifferential(const SquareMatrix<Scalar,3> &F,
                                            const SquareMatrix<Scalar,3> &R,
                                            const SquareMatrix<Scalar,3> &F_differential)
{
    SquareMatrix<Scalar,3> M = R.transpose()*F_differential;
    SquareMatrix<Scalar,3> S = R.transpose()*F;
    SquareMatrix<Scalar,3> K(S(1,1)+S(0,0),S(2,1),-S(2,0),S(2,1),S(2,2)+S(0,0),S(1,0),-S(2,0),S(1,0),S(2,2)+S(1,1));
    Vector<Scalar,3> L(M(1,0)-M(0,1),-M(0,2)+M(2,0),-M(1,2)+M(2,1));
    Vector<Scalar,3> RV = K.inverse()*L; //solve K*x=L
    SquareMatrix<Scalar,3> RTdR(0,-RV[0],-RV[1],RV[0],0,-RV[2],RV[1],RV[2],0);
    return R*RTdR;
}
//explicit instantiations
template SquareMatrix<float,2> rotationDifferential<float>(const SquareMatrix<float,2> &F,
                                                           const SquareMatrix<float,2> &R,
                                                           const SquareMatrix<float,2> &F_differential);
template SquareMatrix<double,2> rotationDifferential<double>(const SquareMatrix<double,2> &F,
                                                             const SquareMatrix<double,2> &R,
                                                             const SquareMatrix<double,2> &F_differential);
template SquareMatrix<float,3> rotationDifferential<float>(const SquareMatrix<float,3> &F,
                                                           const SquareMatrix<float,3> &R,
                                                           const SquareMatrix<float,3> &F_differential);
template SquareMatrix<double,3> rotationDifferential<double>(const SquareMatrix<double,3> &F,
                                                             const SquareMatrix<double,3> &R,
                                                             const SquareMatrix<double,3> &F_differential);

}  //end of namespace ConstitutiveModelInternal

}  //end of namespace Physika
