/*
 * @file linear_system_solvers_test.cpp
 * @brief test implementations of linear system solvers.
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

#include <iostream>
#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"
#include "Physika_Numerics/Linear_System_Solvers/plain_generalized_vector.h"
#include "Physika_Numerics/Linear_System_Solvers/generalized_matrix.h"
#include "Physika_Numerics/Linear_System_Solvers/conjugate_gradient_solver.h"
#include "Physika_Dependency/Eigen/Eigen"

using Physika::MatrixMxN;
using Physika::VectorND;
using Physika::LinearSystem;
using Physika::PlainGeneralizedVector;
using Physika::GeneralizedMatrix;
using Physika::ConjugateGradientSolver;
using namespace std;
typedef float Scalar;

int main()
{
    MatrixMxN<Scalar> A_data(2,2);
    A_data(0,0) = 2; A_data(0,1) = -1; A_data(1,0) = -1; A_data(1,1) = 2;
    GeneralizedMatrix<Scalar> A(A_data);
    LinearSystem<Scalar> linear_system(A);
    PlainGeneralizedVector<Scalar> b(2);
    b[0] = 1; b[1] = 0;
    PlainGeneralizedVector<Scalar> x(2,0);
    //CG
    ConjugateGradientSolver<Scalar> cg_solver;
    cg_solver.solve(linear_system,b,x);
    cout<<"Physika CG solver terminated in "<<cg_solver.iterationsUsed()<<" iterations\n";
    cout<<"Norm of residual :"<<cg_solver.residualMagnitude()<<"\n";
    cout<<"Solution: "<<x<<"\n";
    //PCG with jacobi preconditioner
    linear_system.computeJacobiPreconditioner();
    if(linear_system.preconditioner())
        std::cout<<"Yes\n";
    else
        std::cout<<"No\n";
    cg_solver.reset();
    cg_solver.enablePreconditioner();
    x[0] = 2; x[1] = 0;
    cg_solver.solve(linear_system,b,x);
    cout<<"Physika PCG solver with Jacobi preconditioner terminated in "<<cg_solver.iterationsUsed()<<" iterations\n";
    cout<<"Norm of residual :"<<cg_solver.residualMagnitude()<<"\n";
    cout<<"Solution: "<<x<<"\n";
    return 0;
}
