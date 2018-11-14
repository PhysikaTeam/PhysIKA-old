/*
 * @file matrix_4x4_test.cu
 * @brief cuda test for SquareMatrix<Scalar, 3>.
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

#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Matrices/matrix_4x4.h"
#include "matrix_4x4_test.h"

namespace Physika{

namespace matrix_4x4_test{

__device__ inline void print(Vector4f vec)
{
    printf("(%f, %f, %f, %f)\n", vec[0], vec[1], vec[2], vec[3]);
}

__device__ inline void print(Matrix4f mat)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
            printf("%f, ", mat(i, j));
        printf("\n");
    }
}

__device__ void test_ctor()
{
    printf("test_ctor:\n");

    Matrix4f default_mat;
    print(default_mat);
    printf("\n");

    Matrix4f one_mat(1.0);
    print(one_mat);
    printf("\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    print(ascend_mat);
    printf("\n");

    Vector4f row1(1.0f, 2.0f, 3.0f, 4.0f);
    Vector4f row2(5.0f, 6.0f, 7.0f, 8.0f);
    Vector4f row3(9.0f, 10.0f, 11.0f, 12.0f);
    Vector4f row4(13.0f, 14.0f, 15.0f, 16.0f);
    Matrix4f ascend_mat_2(row1, row2, row3, row4);
    print(ascend_mat_2);
    printf("\n");

    printf("rows: %d\n", Matrix4f::rows());
    printf("cols: %d\n", Matrix4f::cols());

    printf("\n");
}

__device__ void test_operator_asscess()
{
    printf("test_operator_asscess:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    printf("(%f, %f, %f, %f, \
             %f, %f, %f, %f, \
             %f, %f, %f, %f,\
             %f, %f, %f, %f)\n", ascend_mat(0, 0), ascend_mat(0, 1), ascend_mat(0, 2), ascend_mat(0, 3),
                                 ascend_mat(1, 0), ascend_mat(1, 1), ascend_mat(1, 2), ascend_mat(1, 3),
                                 ascend_mat(2, 0), ascend_mat(2, 1), ascend_mat(2, 2), ascend_mat(2, 3),
                                 ascend_mat(3, 0), ascend_mat(3, 1), ascend_mat(3, 2), ascend_mat(3, 3));
    printf("\n");
}

__device__ void test_row_and_col_vector()
{
    printf("test_row_and_col_vector:\n");
    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    print(ascend_mat.rowVector(0));
    print(ascend_mat.rowVector(1));
    print(ascend_mat.rowVector(2));
    print(ascend_mat.rowVector(3));
    printf("\n");
    print(ascend_mat.colVector(0));
    print(ascend_mat.colVector(1));
    print(ascend_mat.colVector(2));
    print(ascend_mat.colVector(3));
    printf("\n");
}

__device__ void test_operator_add()
{
    printf("test_operator_add:\n");

    Matrix4f one_mat(1.0);
    Matrix4f two_mat(2.0);

    print(one_mat + two_mat);
    printf("\n");
    print(one_mat += two_mat);
    printf("\n");
}

__device__ void test_operator_minus()
{
    printf("test_operator_minus:\n");

    Matrix4f one_mat(1.0);
    Matrix4f two_mat(2.0);

    print(one_mat - two_mat);
    printf("\n");
    print(one_mat -= two_mat);
    printf("\n");
}

__device__ void test_operator_assign()
{
    printf("test_operator_assign:\n");

    Matrix4f one_mat(1.0);
    Matrix4f two_mat(2.0);

    print(one_mat = two_mat);
    printf("\n");
}

__device__ void test_operator_equal_not_equal()
{
    printf("test_operator_equal_not_equal:\n");

    Matrix4f one_mat(1.0);
    Matrix4f two_mat(2.0);
    printf("one_mat == one_mat: %d\n", one_mat == one_mat);
    printf("one_mat == two_mat: %d\n", one_mat == two_mat);
    printf("one_mat != two_mat: %d\n", one_mat != two_mat);
    printf("\n");
}

/*
__device__ void test_operator_add_scalar()
{
    printf("test_operator_add_scalar:\n");

    Matrix4f one_mat(1.0);
    print(one_mat + 1.0f);
    print(one_mat += 1.0f);
    printf("\n");
}

__device__ void test_operator_minus_scalar()
{
    printf("test_operator_minus_scalar:\n");

    Matrix4f one_mat(1.0);
    print(one_mat - 1.0f);
    print(one_mat -= 1.0f);
    printf("\n");
}
*/

__device__ void test_operator_multi_scalar()
{
    printf("test_operator_multi_scalar:\n");

    Matrix4f one_mat(1.0);
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
    Matrix4f one_mat(1.0);
    Vector4f one_vec(1.0);
    print(one_mat * one_vec);
    printf("\n");
}

__device__ void test_operator_multi_matrix()
{
    printf("test_operator_multi_matrix:\n");
    Matrix4f one_mat(1.0);
    Matrix4f two_mat(2.0);
    print(one_mat * two_mat);
    printf("\n");
    print(one_mat *= two_mat);
    printf("\n");
}

__device__ void test_operator_sub_scalar()
{
    printf("test_operator_sub_scalar:\n");

    Matrix4f one_mat(1.0);
    print(one_mat / 2.0f);
    printf("\n");
    print(one_mat /= 2.0f);
    printf("\n");
}

__device__ void test_operator_pre_minus()
{
    printf("test_operator_pre_minus:\n");

    Matrix4f one_mat(1.0);
    print(-one_mat);
    printf("\n");
}

__device__ void test_transpose()
{
    printf("test_transpose:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    print(ascend_mat.transpose());
    printf("\n");
}

__device__ void test_inverse()
{
    printf("test_inverse:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    print(ascend_mat.inverse());
    printf("\n");
}

__device__ void test_determinant()
{
    printf("test_determinant:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    printf("trace: %f\n", ascend_mat.determinant());
    printf("\n");
}

__device__ void test_trace()
{
    printf("test_trace:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    printf("trace: %f\n", ascend_mat.trace());
    printf("\n");
}

__device__ void test_double_contraction()
{
    printf("test_double_contraction:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    printf("trace: %f\n", ascend_mat.doubleContraction(ascend_mat));
    printf("\n");
}

__device__ void test_frobenius_norm()
{
    printf("test_frobenius_norm:\n");

    Matrix4f ascend_mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    printf("trace: %f\n", ascend_mat.frobeniusNorm());
    printf("\n");
}

__global__ void test_matrix_4x4()
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
    
}//end of namespace matrix_4x4_test

void testMatrix4x4()
{
    matrix_4x4_test::test_matrix_4x4 << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

}//end of namespace Physika