/*
* @file vector_Nd_test.cpp
* @brief unit test for vector Nd.
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

#include <cmath>
#include <iostream>

#include "gtest/gtest.h"

#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_MxN.h"

using namespace Physika;

template <typename Scalar>
class Vector_Nd_Test: public testing::Test
{
public:

    VectorND<Scalar> default_vector_0d;

    VectorND<Scalar> default_vector_5d{ 5 };
    VectorND<Scalar> zero_vector_5d{ 5, 0 };
    VectorND<Scalar> one_vector_5d{ 5, 1 };
    VectorND<Scalar> two_vector_5d{ 5, 2 };

    VectorND<Scalar> default_vector_6d{ 6 };
    VectorND<Scalar> zero_vector_6d{ 6, 0 };
    VectorND<Scalar> one_vector_6d{ 6, 1 };
    VectorND<Scalar> two_vector_6d{ 6, 2 };
    
};

using TestTypes = testing::Types<float, double, long double>;
TYPED_TEST_CASE(Vector_Nd_Test, TestTypes);

TYPED_TEST(Vector_Nd_Test, ctor)
{
    //explicit ctor
    for (auto i = 0u; i < this->one_vector_5d.dims(); ++i)
        EXPECT_EQ(this->one_vector_5d[i], 1);

    //copy ctor
    auto result = this->one_vector_5d;
    EXPECT_EQ(result, this->one_vector_5d);

    //default ctor for Nd
    EXPECT_EQ(this->default_vector_5d, this->zero_vector_5d);

    //default ctor for 0d
    EXPECT_EQ(this->default_vector_0d.dims(), 0);
}


TYPED_TEST(Vector_Nd_Test, dims)
{
    EXPECT_EQ(this->default_vector_0d.dims(), 0);
    EXPECT_EQ(this->default_vector_5d.dims(), 5);
    EXPECT_EQ(this->default_vector_6d.dims(), 6);
}

TYPED_TEST(Vector_Nd_Test, resize)
{
    this->default_vector_0d.resize(5);
    EXPECT_EQ(this->default_vector_0d, this->zero_vector_5d);
}

TYPED_TEST(Vector_Nd_Test, operator_get_item)
{
    for (auto i = 0u; i < this->one_vector_5d.dims(); i++)
        EXPECT_EQ(this->one_vector_5d[i], 1);
    EXPECT_ANY_THROW(this->one_vector_5d[this->one_vector_5d.dims()]);

    //const operator[]
    const auto const_one_vector_5d = this->one_vector_5d;
    EXPECT_EQ(const_one_vector_5d[0], 1);
    EXPECT_ANY_THROW(const_one_vector_5d[const_one_vector_5d.dims()]);

}

TYPED_TEST(Vector_Nd_Test, operator_add_vector)
{
    //operator +
    auto result = this->one_vector_5d + this->one_vector_5d;
    auto expect_result = VectorND<TypeParam>{5, 2};
    EXPECT_EQ(result, expect_result);

    //operator +=
    result += this->one_vector_5d;
    expect_result = VectorND<TypeParam>{5, 3};
    EXPECT_EQ(result, expect_result);

    //exception
    EXPECT_ANY_THROW(this->one_vector_5d + this->one_vector_6d);
    EXPECT_ANY_THROW(this->one_vector_5d += this->one_vector_6d);

}

TYPED_TEST(Vector_Nd_Test, operator_substract_vector)
{
    //operator -
    auto result = this->one_vector_5d - this->one_vector_5d;
    EXPECT_EQ(result, this->zero_vector_5d);

    //operator -=
    this->two_vector_5d -= this->one_vector_5d;
    EXPECT_EQ(this->two_vector_5d, this->one_vector_5d);

    //exception
    EXPECT_ANY_THROW(this->one_vector_5d - this->one_vector_6d);
    EXPECT_ANY_THROW(this->one_vector_5d -= this->one_vector_6d);
}

TYPED_TEST(Vector_Nd_Test, assignment)
{
    this->one_vector_5d = this->zero_vector_5d;
    EXPECT_EQ(this->one_vector_5d, this->zero_vector_5d);

    this->one_vector_5d = this->zero_vector_6d;
    EXPECT_EQ(this->one_vector_5d, this->zero_vector_6d);
}

TYPED_TEST(Vector_Nd_Test, operator_eq_neq)
{
    EXPECT_EQ(this->zero_vector_5d == this->zero_vector_5d, true);

    auto another_zero_vector_5d = VectorND<TypeParam>{5, 0};
    EXPECT_EQ(this->zero_vector_5d == another_zero_vector_5d, true);

    EXPECT_EQ(this->zero_vector_5d != this->one_vector_5d, true);
    EXPECT_EQ(this->zero_vector_5d != this->zero_vector_6d, true);
}

TYPED_TEST(Vector_Nd_Test, operator_add_with_scalar)
{
    //operator +
    auto result = this->zero_vector_5d + 1;
    EXPECT_EQ(result, this->one_vector_5d);

    //operator +=
    result += 1;
    EXPECT_EQ(result, this->two_vector_5d);

    //operator + for 0d
    result = this->default_vector_0d + 1;
    EXPECT_EQ(this->default_vector_0d, result);

    //operator += for 0d
    result += 1;
    EXPECT_EQ(this->default_vector_0d, result);

    
}

TYPED_TEST(Vector_Nd_Test, operator_substract_with_scalar)
{
    //operator -
    auto result = this->one_vector_5d - 1;
    EXPECT_EQ(result, this->zero_vector_5d);

    //operator -=
    this->two_vector_5d -= 1;
    EXPECT_EQ(this->two_vector_5d, this->one_vector_5d);

    //operator + for 0d
    result = this->default_vector_0d - 1;
    EXPECT_EQ(this->default_vector_0d, result);

    //operator += for 0d
    result -= 1;
    EXPECT_EQ(this->default_vector_0d, result);
}

TYPED_TEST(Vector_Nd_Test, operator_multi_with_scalar)
{
    //operator *
    auto result = this->one_vector_5d * 2;
    EXPECT_EQ(result, this->two_vector_5d);

    result = 2 * this->one_vector_5d;
    EXPECT_EQ(result, this->two_vector_5d);

    //operator *=
    result *= 0;
    EXPECT_EQ(result, this->zero_vector_5d);

    //operator * for 0d
    result = this->default_vector_0d * 2;
    EXPECT_EQ(this->default_vector_0d, result);

    result = 2 * this->default_vector_0d;
    EXPECT_EQ(this->default_vector_0d, result);

    //operator *= for 0d
    result *= 2;
    EXPECT_EQ(this->default_vector_0d, result);
}

TYPED_TEST(Vector_Nd_Test, operator_subdivide_with_scalar)
{
    //operator /
    auto result = this->two_vector_5d / 2;
    EXPECT_EQ(result, this->one_vector_5d);

    EXPECT_ANY_THROW(result / 0);
    EXPECT_ANY_THROW(this->default_vector_0d / 0);

    //operator /=
    result /= 1;
    EXPECT_EQ(result, this->one_vector_5d);

    EXPECT_ANY_THROW(result /= 0);
    EXPECT_ANY_THROW(this->default_vector_0d /= 0);
}

TYPED_TEST(Vector_Nd_Test, norm)
{
    EXPECT_EQ(this->default_vector_0d.norm(), 0);

    EXPECT_EQ(this->zero_vector_5d.norm(), 0);
    EXPECT_NEAR(this->one_vector_5d.norm(), sqrt(1*5), 1.0e-6);
    EXPECT_NEAR(this->two_vector_5d.norm(), sqrt(4*5), 1.0e-6);

    EXPECT_EQ(this->zero_vector_6d.norm(), 0);
    EXPECT_NEAR(this->one_vector_6d.norm(), sqrt(1 * 6), 1.0e-6);
    EXPECT_NEAR(this->two_vector_6d.norm(), sqrt(4 * 6), 1.0e-6);

    auto negative_one_vector_5d = VectorND<TypeParam>{5, static_cast<TypeParam>(-1)};
    EXPECT_NEAR(negative_one_vector_5d.norm(), sqrt(1*5), 1.0e-6);
}

TYPED_TEST(Vector_Nd_Test, norm_squared)
{
    EXPECT_EQ(this->two_vector_5d.normSquared(), 4*5);
    EXPECT_EQ(this->one_vector_5d.normSquared(), 1*5);
    EXPECT_EQ(this->zero_vector_5d.normSquared(), 0);

    EXPECT_EQ(this->two_vector_6d.normSquared(), 4*6);
    EXPECT_EQ(this->one_vector_6d.normSquared(), 1*6);
    EXPECT_EQ(this->zero_vector_6d.normSquared(), 0);

    auto  negative_two_vector_5d = VectorND<TypeParam>{5, static_cast<TypeParam>(-2)};
    EXPECT_EQ(negative_two_vector_5d.normSquared(), 4*5);
}

TYPED_TEST(Vector_Nd_Test, normalize)
{
    EXPECT_EQ(this->default_vector_0d.normalize(), this->default_vector_0d);

    EXPECT_EQ(this->two_vector_5d.normalize(), this->one_vector_5d.normalize());

    //need further consideration
    EXPECT_EQ(this->zero_vector_5d.normalize(), this->zero_vector_5d);

}


TYPED_TEST(Vector_Nd_Test, operator_pre_substract)
{
    auto  negative_one_vector_5d = VectorND<TypeParam>{ 5, static_cast<TypeParam>(-1) };
    EXPECT_EQ(-negative_one_vector_5d, this->one_vector_5d);
    EXPECT_EQ(-this->one_vector_5d, negative_one_vector_5d);
}

TYPED_TEST(Vector_Nd_Test, dot)
{
    EXPECT_EQ(this->two_vector_5d.dot(this->two_vector_5d), 4*5);
    EXPECT_EQ(this->one_vector_5d.dot(this->two_vector_5d), 2*5);
    EXPECT_EQ(this->two_vector_5d.dot(this->one_vector_5d), 2*5);
    EXPECT_EQ(this->two_vector_5d.dot(this->zero_vector_5d), 0);

    EXPECT_ANY_THROW(this->default_vector_0d.dot(this->default_vector_5d));
    EXPECT_ANY_THROW(this->default_vector_5d.dot(this->default_vector_6d));

}

TYPED_TEST(Vector_Nd_Test, outer_product)
{
    auto expect_result = MatrixMxN<TypeParam>{ 5, 5, 2 };
    EXPECT_EQ(this->one_vector_5d.outerProduct(this->two_vector_5d), expect_result);

    //note: static_cast is required in the following codes, because compiler would deduce {5, 5, x} to initializer_list<int>,
    //      and we does not define a ctor that accept initializer_list<int>, so the compiler would reject !
    
    //need further consideration, may be we should just use () to initialize instead of {}

    expect_result = MatrixMxN<TypeParam>{ 5, 5, static_cast<TypeParam>(0) };
    EXPECT_EQ(this->one_vector_5d.outerProduct(this->zero_vector_5d), expect_result);

    expect_result = MatrixMxN<TypeParam>{ 5, 6, static_cast<TypeParam>(2) };
    EXPECT_EQ(this->one_vector_5d.outerProduct(this->two_vector_6d), expect_result);

    expect_result = MatrixMxN<TypeParam>{ 6, 5, static_cast<TypeParam>(2) };
    EXPECT_EQ(this->one_vector_6d.outerProduct(this->two_vector_5d), expect_result);
}