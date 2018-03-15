/*
 * @file vector_2d_test.cu
 * @brief cuda test for Vector<Scalar, 2>.
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
#include "vector_2d_test.h"

namespace Physika{

namespace vector_2d_test{

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

    Vector2f default_vec;
    print(default_vec);

    Vector2f one_vec(1.0);
    print(one_vec);

    Vector2f ascend_vec(1.0, 2.0);
    print(ascend_vec);

    printf("dims: %d\n", Vector2f::dims());
    printf("\n");
}

__device__ void test_operator_asscess()
{
    printf("test_operator_asscess:\n");

    Vector2f ascend_vec(1.0, 2.0);
    printf("(%f, %f)\n", ascend_vec[0], ascend_vec[1]);
    printf("\n");
}

__device__ void test_operator_add()
{
    printf("test_operator_add:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    print(one_vec + two_vec);
    print(one_vec += two_vec);
    printf("\n");
}

__device__ void test_operator_minus()
{
    printf("test_operator_minus:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    print(one_vec - two_vec);
    print(one_vec -= two_vec);
    printf("\n");
}

__device__ void test_operator_assign()
{
    printf("test_operator_assign:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    print(one_vec = two_vec);
    printf("\n");
}

__device__ void test_operator_equal_not_equal()
{
    printf("test_operator_equal_not_equal:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);
    printf("one_vec == one_vec: %d\n", one_vec == one_vec);
    printf("one_vec == two_vec: %d\n", one_vec == two_vec);
    printf("one_vec != two_vec: %d\n", one_vec != two_vec);
    printf("\n");
}

__device__ void test_operator_add_scalar()
{
    printf("test_operator_add_scalar:\n");

    Vector2f one_vec(1.0);
    print(one_vec + 1.0f);
    print(one_vec += 1.0f);
    printf("\n");
}

__device__ void test_operator_minus_scalar()
{
    printf("test_operator_minus_scalar:\n");

    Vector2f one_vec(1.0);
    print(one_vec - 1.0f);
    print(one_vec -= 1.0f);
    printf("\n");
}

__device__ void test_operator_multi_scalar()
{
    printf("test_operator_multi_scalar:\n");

    Vector2f one_vec(1.0);
    print(one_vec * 2.0f);
    print(2.0f * one_vec);
    print(one_vec *= 2.0f);
    printf("\n");
}

__device__ void test_operator_sub_scalar()
{
    printf("test_operator_sub_scalar:\n");

    Vector2f one_vec(1.0);
    print(one_vec / 2.0f);
    print(one_vec /= 2.0f);
    printf("\n");
}

__device__ void test_operator_pre_minus()
{
    printf("test_operator_pre_minus:\n");

    Vector2f one_vec(1.0);
    print(-one_vec);
    printf("\n");
}

__device__ void test_norm()
{
    printf("test_norm:\n");

    Vector2f one_vec(1.0);
    printf("one_vec_norm: %f\n", one_vec.norm());

    Vector2f two_vec(2.0);
    printf("two_vec_norm: %f\n", two_vec.norm());

    printf("\n");
}

__device__ void test_norm_squared()
{
    printf("test_norm_squared:\n");

    Vector2f one_vec(1.0);
    printf("one_vec_normsquared: %f\n", one_vec.normSquared());

    Vector2f two_vec(2.0);
    printf("two_vec_normsquared: %f\n", two_vec.normSquared());

    printf("\n");
}

__device__ void test_normalize()
{
    printf("test_normalize:\n");

    Vector2f one_vec(1.0);
    print(one_vec.normalize());

    Vector2f two_vec(2.0);
    print(two_vec.normalize());

    printf("\n");
}

__device__ void test_cross()
{
    printf("test_cross:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    printf("one_vec.cross(two_vec): %f\n", one_vec.cross(two_vec));

    printf("\n");
}

__device__ void test_dot()
{
    printf("test_dot:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    printf("one_vec.dot(two_vec): %f\n", one_vec.dot(two_vec));

    printf("\n");
}

__device__ void test_outer_product()
{
    printf("test_outer_product:\n");

    Vector2f one_vec(1.0);
    Vector2f two_vec(2.0);

    printf("one_vec.outerProduct(two_vec):\n");
    print(one_vec.outerProduct(two_vec));

    printf("\n");
}

__global__ void test_vector_2d()
{
    test_ctor();
    test_operator_asscess();
    test_operator_add();
    test_operator_minus();
    test_operator_assign();
    test_operator_equal_not_equal();
    test_operator_add_scalar();
    test_operator_minus_scalar();
    test_operator_multi_scalar();
    test_operator_sub_scalar();
    test_operator_pre_minus();

    test_norm();
    test_norm_squared();
    test_normalize();
    test_cross();
    test_dot();
    test_outer_product();
}

}//end of namespace vector_2d_test

void testVector2d()
{
    vector_2d_test::test_vector_2d << <1, 1 >> > ();
    cudaDeviceSynchronize();
}

}//end of namespace Physika