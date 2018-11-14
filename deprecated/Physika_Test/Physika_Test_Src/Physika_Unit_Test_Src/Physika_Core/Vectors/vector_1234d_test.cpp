/*
* @file Vector_1234d_Test.cpp
* @brief unit test for vector 1234d.
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

#include "gtest/gtest.h"

#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Matrices/matrix_1x1.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

using namespace Physika;

//trait class
template <typename SCALAR, int DIM>
class VectorType
{
public:
    using Scalar = SCALAR;
    static constexpr int Dim = DIM;
};

template <typename VectorType>
class Vector_Test: public testing::Test
{
public:
    Vector_Test<VectorType>();

public:
    Vector<typename VectorType::Scalar, VectorType::Dim> default_vector;
    Vector<typename VectorType::Scalar, VectorType::Dim> zero_vector{ 0 };
    Vector<typename VectorType::Scalar, VectorType::Dim> one_vector { 1 };
    Vector<typename VectorType::Scalar, VectorType::Dim> two_vector { 2 };

    Vector<typename VectorType::Scalar, VectorType::Dim> ascend_vector;
};

template <typename VectorType>
Vector_Test<VectorType>::Vector_Test<VectorType>()
{
    for (auto i = 0u; i < VectorType::Dim; ++i)
        this->ascend_vector[i] = i;
}

TYPED_TEST_CASE_P(Vector_Test);

using TestTypes_1d = testing::Types< VectorType<unsigned char, 1>,
                                     VectorType<unsigned short, 1>,
                                     VectorType<unsigned int, 1>,
                                     VectorType<unsigned long, 1>,
                                     VectorType<unsigned long, 1>,
                                     VectorType<long, 1>,
                                     VectorType<signed char, 1>,
                                     VectorType<short, 1>,
                                     VectorType<int, 1>,
                                     VectorType<long, 1>,
                                     VectorType<long long, 1>,
                                     VectorType<float, 1>,
                                     VectorType<double, 1>,
                                     VectorType<long double, 1>
                                   > ;

using TestTypes_2d = testing::Types< VectorType<unsigned char, 2>,
                                     VectorType<unsigned short, 2>,
                                     VectorType<unsigned int, 2>,
                                     VectorType<unsigned long, 2>,
                                     VectorType<unsigned long, 2>,
                                     VectorType<long, 2>,
                                     VectorType<signed char, 2>,
                                     VectorType<short, 2>,
                                     VectorType<int, 2>,
                                     VectorType<long, 2>,
                                     VectorType<long long, 2>,
                                     VectorType<float, 2>,
                                     VectorType<double, 2>,
                                     VectorType<long double, 2>
                                   >;

using TestTypes_3d = testing::Types< VectorType<unsigned char, 3>,
                                     VectorType<unsigned short, 3>,
                                     VectorType<unsigned int, 3>,
                                     VectorType<unsigned long, 3>,
                                     VectorType<unsigned long, 3>,
                                     VectorType<long, 3>,
                                     VectorType<signed char, 3>,
                                     VectorType<short, 3>,
                                     VectorType<int, 3>,
                                     VectorType<long, 3>,
                                     VectorType<long long, 3>,
                                     VectorType<float, 3>,
                                     VectorType<double, 3>,
                                     VectorType<long double, 3>
                                   >;

using TestTypes_4d = testing::Types< VectorType<unsigned char, 4>,
                                     VectorType<unsigned short, 4>,
                                     VectorType<unsigned int, 4>,
                                     VectorType<unsigned long, 4>,
                                     VectorType<unsigned long, 4>,
                                     VectorType<long, 4>,
                                     VectorType<signed char, 4>,
                                     VectorType<short, 4>,
                                     VectorType<int, 4>,
                                     VectorType<long, 4>,
                                     VectorType<long long, 4>,
                                     VectorType<float, 4>,
                                     VectorType<double, 4>,
                                     VectorType<long double, 4>
                                   >;

//TYPED_TEST_CASE(Vector_Test, TestTypes_1d);
//TYPED_TEST_CASE(Vector_Test, TestTypes_2d);

TYPED_TEST_P(Vector_Test, ctor)
{
    //explicit ctor
    for (auto i = 0u; i < TypeParam::Dim; ++i)
        EXPECT_EQ(this->one_vector[i], 1);

    for (auto i = 0u; i < TypeParam::Dim; ++i)
        EXPECT_EQ(this->ascend_vector[i], i);

    //copy ctor
    auto result = this->one_vector;
    EXPECT_EQ(result, this->one_vector);

    //default ctor
    EXPECT_EQ(this->default_vector, this->zero_vector);
}

TYPED_TEST_P(Vector_Test, dims)
{
    EXPECT_EQ(this->default_vector.dims(), TypeParam::Dim);
}

TYPED_TEST_P(Vector_Test, operator_get_item)
{
    for(auto i = 0u; i < TypeParam::Dim; ++i)
        EXPECT_EQ(this->ascend_vector[i], i);

    EXPECT_ANY_THROW(this->one_vector[TypeParam::Dim]);

    //const operator[]
    const auto const_ascend_vector = this->ascend_vector;
    for(auto i = 0u; i < TypeParam::Dim; ++i)
        EXPECT_EQ(const_ascend_vector[i], i);
    EXPECT_ANY_THROW(const_ascend_vector[TypeParam::Dim]);

}

TYPED_TEST_P(Vector_Test, operator_add_vector)
{
    //operator +
    auto result = this->one_vector + this->one_vector;
    auto expect_result = Vector<typename TypeParam::Scalar, TypeParam::Dim>{2};
    EXPECT_EQ(result, expect_result);

    //operator +=
    result += this->one_vector;
    expect_result = Vector<typename TypeParam::Scalar, TypeParam::Dim>{3};
    EXPECT_EQ(result, expect_result);
}

TYPED_TEST_P(Vector_Test, operator_substract_vector)
{
    //operator -
    auto result = this->one_vector - this->one_vector;
    EXPECT_EQ(result, this->zero_vector);

    //operator -=
    this->two_vector -= this->one_vector;
    EXPECT_EQ(this->two_vector, this->one_vector);
}

TYPED_TEST_P(Vector_Test, assignment)
{
    this->one_vector = this->zero_vector;
    EXPECT_EQ(this->one_vector, this->zero_vector);
}

TYPED_TEST_P(Vector_Test, operator_eq_neq)
{
    EXPECT_EQ(this->zero_vector == this->zero_vector, true);

    auto another_zero_vector = Vector<typename TypeParam::Scalar, TypeParam::Dim>{0};
    EXPECT_EQ(this->zero_vector == another_zero_vector, true);

    EXPECT_EQ(this->zero_vector != this->one_vector, true);
}

TYPED_TEST_P(Vector_Test, operator_add_with_scalar)
{
    //operator +
    auto result = this->zero_vector + 1;
    EXPECT_EQ(result, this->one_vector);

    //operator +=
    result += 1;
    EXPECT_EQ(result, this->two_vector);

    
}

TYPED_TEST_P(Vector_Test, operator_substract_with_scalar)
{
    //operator -
    auto result = this->one_vector - 1;
    EXPECT_EQ(result, this->zero_vector);

    //operator -=
    this->two_vector -= 1;
    EXPECT_EQ(this->two_vector, this->one_vector);
}

TYPED_TEST_P(Vector_Test, operator_multi_with_scalar)
{
    //operator *
    auto result = this->one_vector * 2;
    EXPECT_EQ(result, this->two_vector);

    result = 2 * this->one_vector;
    EXPECT_EQ(result, this->two_vector);

    //operator *=
    result *= 0;
    EXPECT_EQ(result, this->zero_vector);
}

TYPED_TEST_P(Vector_Test, operator_subdivide_with_scalar)
{
    //operator /
    auto result = this->two_vector / 2;
    EXPECT_EQ(result, this->one_vector);

    EXPECT_ANY_THROW(result / 0);

    //operator /=
    result /= 1;
    EXPECT_EQ(result, this->one_vector);

    EXPECT_ANY_THROW(result /= 0);
}

TYPED_TEST_P(Vector_Test, norm)
{
    using Scalar = typename TypeParam::Scalar;
    constexpr int Dim = TypeParam::Dim;

    EXPECT_EQ(this->zero_vector.norm(), 0);
    EXPECT_EQ(this->one_vector.norm(), static_cast<Scalar>(sqrt(Dim)));
    EXPECT_EQ(this->two_vector.norm(), static_cast<Scalar>(sqrt(Dim * 4)));

    auto negative_one_vector = Vector<Scalar, Dim>{static_cast<Scalar>(-1)};
    EXPECT_EQ(negative_one_vector.norm(), static_cast<Scalar>(sqrt(Dim)));
}

TYPED_TEST_P(Vector_Test, norm_squared)
{
    EXPECT_EQ(this->two_vector.normSquared(), TypeParam::Dim * 4);
    EXPECT_EQ(this->one_vector.normSquared(), TypeParam::Dim);
    EXPECT_EQ(this->zero_vector.normSquared(), 0);

    auto  negative_two_vector = Vector<typename TypeParam::Scalar, TypeParam::Dim>{static_cast<typename TypeParam::Scalar>(-2)};
    EXPECT_EQ(negative_two_vector.normSquared(), TypeParam::Dim * 4);
}

TYPED_TEST_P(Vector_Test, normalize)
{
    EXPECT_EQ(this->two_vector.normalize(), this->one_vector.normalize());

    //need further consideration
    EXPECT_EQ(this->zero_vector.normalize(), this->zero_vector);

}

///////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
void cross_test(const Vector<Scalar, 1> & lhs, const Vector<Scalar, 1> & rhs)
{
    EXPECT_EQ(lhs.cross(rhs), 0);
}

template <typename Scalar>
void cross_test(const Vector<Scalar, 2> & lhs, const Vector<Scalar, 2> & rhs)
{
    Scalar expect_result = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    EXPECT_EQ(lhs.cross(rhs), expect_result);
}

template <typename Scalar>
void cross_test(const Vector<Scalar, 3> & lhs, const Vector<Scalar, 3> & rhs)
{
    Scalar x = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    Scalar y = -(lhs[0] * rhs[2] - lhs[2] * rhs[0]);
    Scalar z = lhs[0] * rhs[1] - lhs[1] * rhs[0];

    auto expect_result = Vector<Scalar, 3>{ x, y, z };
    EXPECT_EQ(lhs.cross(rhs), expect_result);
}

template <typename Scalar>
void cross_test(const Vector<Scalar, 4> & one_vector, const Vector<Scalar, 4> & two_vector)
{
    //no cross operator provided for 4-dim vector
}

TYPED_TEST_P(Vector_Test, cross)
{
    cross_test(this->one_vector, this->two_vector);
    cross_test(this->one_vector, this->zero_vector);
    cross_test(this->zero_vector, this->two_vector);

    cross_test(this->ascend_vector, this->zero_vector);
    cross_test(this->ascend_vector, this->one_vector);
    cross_test(this->ascend_vector, this->two_vector);
}


TYPED_TEST_P(Vector_Test, operator_pre_substract)
{
    auto  negative_one_vector = Vector<typename TypeParam::Scalar, TypeParam::Dim>{static_cast<typename TypeParam::Scalar>(-1)};
    EXPECT_EQ(-negative_one_vector, this->one_vector);
    EXPECT_EQ(-this->one_vector, negative_one_vector);
}

TYPED_TEST_P(Vector_Test, dot)
{
    EXPECT_EQ(this->two_vector.dot(this->two_vector), 4 * TypeParam::Dim);
    EXPECT_EQ(this->one_vector.dot(this->two_vector), 2 * TypeParam::Dim);
    EXPECT_EQ(this->two_vector.dot(this->one_vector), 2 * TypeParam::Dim);
    EXPECT_EQ(this->two_vector.dot(this->zero_vector), 0);
}

TYPED_TEST_P(Vector_Test, outer_product)
{
    auto expect_result = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{ 2 };
    EXPECT_EQ(this->one_vector.outerProduct(this->two_vector), expect_result);

    expect_result = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{0};
    EXPECT_EQ(this->one_vector.outerProduct(this->zero_vector), expect_result);
}

REGISTER_TYPED_TEST_CASE_P(Vector_Test, ctor, dims, operator_get_item, operator_add_vector,
                                        operator_substract_vector, assignment, operator_eq_neq,
                                        operator_add_with_scalar, operator_substract_with_scalar,
                                        operator_multi_with_scalar, operator_subdivide_with_scalar,
                                        norm, norm_squared, normalize, cross, operator_pre_substract, 
                                        dot, outer_product
                          );

INSTANTIATE_TYPED_TEST_CASE_P(Dim_1_, Vector_Test, TestTypes_1d);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_2_, Vector_Test, TestTypes_2d);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_3_, Vector_Test, TestTypes_3d);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_4_, Vector_Test, TestTypes_4d);