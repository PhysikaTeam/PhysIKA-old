/*
 * @file matrix_2x2_test.cu
 * @brief cuda test for SquareMatrix<Scalar, 2>.
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

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "matrix_2x2_test.h"

namespace Physika{

namespace matrix_2x2_test{

__device__ inline void print(Vector2f vec)
{
    printf("(%f, %f)\n", vec[0], vec[1]);
}

__device__ inline void print(Matrix2f mat)
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
            printf("%f, ", mat(i, j));
        printf("\n");
    }
}

__device__ void test_ctor()
{
    printf("test_ctor:\n");

    Matrix2f default_mat;
    print(default_mat);
    printf("\n");

    Matrix2f one_mat(1.0);
    print(one_mat);
    printf("\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    print(ascend_mat);
    printf("\n");

    Vector2f row1(1.0f, 2.0f);
    Vector2f row2(3.0f, 4.0f);
    Matrix2f ascend_mat_2(row1, row2);
    print(ascend_mat_2);
    printf("\n");

    printf("rows: %d\n", Matrix2f::rows());
    printf("cols: %d\n", Matrix2f::cols());

    printf("\n");
}

__device__ void test_operator_asscess()
{
    printf("test_operator_asscess:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    printf("(%f, %f, %f, %f)\n", ascend_mat(0, 0), ascend_mat(0, 1), ascend_mat(1, 0), ascend_mat(1, 1));
    printf("\n");
}

__device__ void test_row_and_col_vector()
{
    printf("test_row_and_col_vector:\n");
    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    print(ascend_mat.rowVector(0));
    print(ascend_mat.rowVector(1));
    printf("\n");
    print(ascend_mat.colVector(0));
    print(ascend_mat.colVector(1));
    printf("\n");
}

__device__ void test_operator_add()
{
    printf("test_operator_add:\n");

    Matrix2f one_mat(1.0);
    Matrix2f two_mat(2.0);

    print(one_mat + two_mat);
    printf("\n");
    print(one_mat += two_mat);
    printf("\n");
}

__device__ void test_operator_minus()
{
    printf("test_operator_minus:\n");

    Matrix2f one_mat(1.0);
    Matrix2f two_mat(2.0);

    print(one_mat - two_mat);
    printf("\n");
    print(one_mat -= two_mat);
    printf("\n");
}

__device__ void test_operator_assign()
{
    printf("test_operator_assign:\n");

    Matrix2f one_mat(1.0);
    Matrix2f two_mat(2.0);

    print(one_mat = two_mat);
    printf("\n");
}

__device__ void test_operator_equal_not_equal()
{
    printf("test_operator_equal_not_equal:\n");

    Matrix2f one_mat(1.0);
    Matrix2f two_mat(2.0);
    printf("one_mat == one_mat: %d\n", one_mat == one_mat);
    printf("one_mat == two_mat: %d\n", one_mat == two_mat);
    printf("one_mat != two_mat: %d\n", one_mat != two_mat);
    printf("\n");
}

/*
__device__ void test_operator_add_scalar()
{
    printf("test_operator_add_scalar:\n");

    Matrix2f one_mat(1.0);
    print(one_mat + 1.0f);
    print(one_mat += 1.0f);
    printf("\n");
}

__device__ void test_operator_minus_scalar()
{
    printf("test_operator_minus_scalar:\n");

    Matrix2f one_mat(1.0);
    print(one_mat - 1.0f);
    print(one_mat -= 1.0f);
    printf("\n");
}
*/

__device__ void test_operator_multi_scalar()
{
    printf("test_operator_multi_scalar:\n");

    Matrix2f one_mat(1.0);
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
    Matrix2f one_mat(1.0);
    Vector2f one_vec(1.0);
    print(one_mat * one_vec);
    printf("\n");
}

__device__ void test_operator_multi_matrix()
{
    printf("test_operator_multi_matrix:\n");
    Matrix2f one_mat(1.0);
    Matrix2f two_mat(2.0);
    print(one_mat * two_mat);
    printf("\n");
    print(one_mat *= two_mat);
    printf("\n");
}

__device__ void test_operator_sub_scalar()
{
    printf("test_operator_sub_scalar:\n");

    Matrix2f one_mat(1.0);
    print(one_mat / 2.0f);
    printf("\n");
    print(one_mat /= 2.0f);
    printf("\n");
}

__device__ void test_operator_pre_minus()
{
    printf("test_operator_pre_minus:\n");

    Matrix2f one_mat(1.0);
    print(-one_mat);
    printf("\n");
}

__device__ void test_transpose()
{
    printf("test_transpose:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    print(ascend_mat.transpose());
    printf("\n");
}

__device__ void test_inverse()
{
    printf("test_inverse:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    print(ascend_mat.inverse());
    printf("\n");
}

__device__ void test_determinant()
{
    printf("test_determinant:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    printf("trace: %f\n", ascend_mat.determinant());
    printf("\n");
}

__device__ void test_trace()
{
    printf("test_trace:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    printf("trace: %f\n", ascend_mat.trace());
    printf("\n");
}

__device__ void test_double_contraction()
{
    printf("test_double_contraction:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    printf("trace: %f\n", ascend_mat.doubleContraction(ascend_mat));
    printf("\n");
}

__device__ void test_frobenius_norm()
{
    printf("test_frobenius_norm:\n");

    Matrix2f ascend_mat(1.0, 2.0, 3.0, 4.0);
    printf("trace: %f\n", ascend_mat.frobeniusNorm());
    printf("\n");
}

__global__ void test_matrix_2x2()
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
    
}//end of namespace matrix_2x2_test

void testMatrix2x2()
{
    matrix_2x2_test::test_matrix_2x2 << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

}//end of namespace Physika