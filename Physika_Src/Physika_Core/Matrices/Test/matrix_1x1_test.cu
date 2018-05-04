/*
 * @file matrix_1x1_test.cu
 * @brief cuda test for SquareMatrix<Scalar, 1>.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "cuda_runtime.h"

#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Matrices/matrix_1x1.h"
#include "matrix_1x1_test.h"

namespace Physika{

namespace matrix_1x1_test{

__device__ inline void print(Vector1f vec)
{
    printf("(%f)\n", vec[0]);
}

__device__ inline void print(Matrix1f mat)
{
    printf("(%f)\n", mat(0, 0));
}

__device__ void test_ctor()
{
    printf("test_ctor:\n");

    Matrix1f default_mat;
    print(default_mat);
    printf("\n");

    Matrix1f one_mat(1.0);
    print(one_mat);
    printf("\n");

    Matrix1f ascend_mat(1.0);
    print(ascend_mat);
    printf("\n");

    printf("rows: %d\n", Matrix1f::rows());
    printf("cols: %d\n", Matrix1f::cols());

    printf("\n");
}

__device__ void test_operator_asscess()
{
    printf("test_operator_asscess:\n");

    Matrix1f ascend_mat(1.0);
    printf("(%f)\n", ascend_mat(0, 0));
    printf("\n");
}

__device__ void test_row_and_col_vector()
{
    printf("test_row_and_col_vector:\n");
    Matrix1f ascend_mat(1.0);
    print(ascend_mat.rowVector(0));
    printf("\n");
    print(ascend_mat.colVector(0));
    printf("\n");
}

__device__ void test_operator_add()
{
    printf("test_operator_add:\n");

    Matrix1f one_mat(1.0);
    Matrix1f two_mat(2.0);

    print(one_mat + two_mat);
    printf("\n");
    print(one_mat += two_mat);
    printf("\n");
}

__device__ void test_operator_minus()
{
    printf("test_operator_minus:\n");

    Matrix1f one_mat(1.0);
    Matrix1f two_mat(2.0);

    print(one_mat - two_mat);
    printf("\n");
    print(one_mat -= two_mat);
    printf("\n");
}

__device__ void test_operator_assign()
{
    printf("test_operator_assign:\n");

    Matrix1f one_mat(1.0);
    Matrix1f two_mat(2.0);

    print(one_mat = two_mat);
    printf("\n");
}

__device__ void test_operator_equal_not_equal()
{
    printf("test_operator_equal_not_equal:\n");

    Matrix1f one_mat(1.0);
    Matrix1f two_mat(2.0);
    printf("one_mat == one_mat: %d\n", one_mat == one_mat);
    printf("one_mat == two_mat: %d\n", one_mat == two_mat);
    printf("one_mat != two_mat: %d\n", one_mat != two_mat);
    printf("\n");
}

/*
__device__ void test_operator_add_scalar()
{
    printf("test_operator_add_scalar:\n");

    Matrix1f one_mat(1.0);
    print(one_mat + 1.0f);
    print(one_mat += 1.0f);
    printf("\n");
}

__device__ void test_operator_minus_scalar()
{
    printf("test_operator_minus_scalar:\n");

    Matrix1f one_mat(1.0);
    print(one_mat - 1.0f);
    print(one_mat -= 1.0f);
    printf("\n");
}
*/

__device__ void test_operator_multi_scalar()
{
    printf("test_operator_multi_scalar:\n");

    Matrix1f one_mat(1.0);
    print(one_mat * 2.0f);
    printf("\n");
    print(2.0f * one_mat);
    printf("\n");
    print(one_mat *= 2.0f);
    printf("\n");
}

__device__ void test_operator_multi_vector()
{
    printf("test_operator_multi_vector:\n");
    Matrix1f one_mat(1.0);
    Vector1f one_vec(1.0);
    print(one_mat * one_vec);
    printf("\n");
}

__device__ void test_operator_multi_matrix()
{
    printf("test_operator_multi_matrix:\n");
    Matrix1f one_mat(1.0);
    Matrix1f two_mat(2.0);
    print(one_mat * two_mat);
    printf("\n");
    print(one_mat *= two_mat);
    printf("\n");
}

__device__ void test_operator_sub_scalar()
{
    printf("test_operator_sub_scalar:\n");

    Matrix1f one_mat(1.0);
    print(one_mat / 2.0f);
    printf("\n");
    print(one_mat /= 2.0f);
    printf("\n");
}

__device__ void test_operator_pre_minus()
{
    printf("test_operator_pre_minus:\n");

    Matrix1f one_mat(1.0);
    print(-one_mat);
    printf("\n");
}

__device__ void test_transpose()
{
    printf("test_transpose:\n");

    Matrix1f ascend_mat(1.0);
    print(ascend_mat.transpose());
    printf("\n");
}

__device__ void test_inverse()
{
    printf("test_inverse:\n");

    Matrix1f ascend_mat(1.0);
    print(ascend_mat.inverse());
    printf("\n");
}

__device__ void test_determinant()
{
    printf("test_determinant:\n");

    Matrix1f ascend_mat(1.0);
    printf("trace: %f\n", ascend_mat.determinant());
    printf("\n");
}

__device__ void test_trace()
{
    printf("test_trace:\n");

    Matrix1f ascend_mat(1.0);
    printf("trace: %f\n", ascend_mat.trace());
    printf("\n");
}

__device__ void test_double_contraction()
{
    printf("test_double_contraction:\n");

    Matrix1f ascend_mat(1.0);
    printf("trace: %f\n", ascend_mat.doubleContraction(ascend_mat));
    printf("\n");
}

__device__ void test_frobenius_norm()
{
    printf("test_frobenius_norm:\n");

    Matrix1f ascend_mat(1.0);
    printf("trace: %f\n", ascend_mat.frobeniusNorm());
    printf("\n");
}

__global__ void test_matrix_1x1()
{
    test_ctor();
    test_operator_asscess();
    test_row_and_col_vector();
    test_operator_add();
    test_operator_minus();
    test_operator_assign();
    test_operator_equal_not_equal();
    //test_operator_add_scalar();
    //test_operator_minus_scalar();
    test_operator_multi_vector();
    test_operator_multi_matrix();
    test_operator_multi_scalar();
    test_operator_sub_scalar();
    test_operator_pre_minus();

    test_transpose();
    test_inverse();

    test_determinant();
    test_trace();
    test_double_contraction();
    test_frobenius_norm();
}
    
}//end of namespace matrix_1x1_test

void testMatrix1x1()
{
    matrix_1x1_test::test_matrix_1x1 << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

}//end of namespace Physika