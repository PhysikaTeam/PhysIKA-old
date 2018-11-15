/*
* @file matrix_MxN_test.cpp
* @brief unit test for matrix MxN.
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

#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Matrices/matrix_MxN.h"

using namespace Physika;

template <typename Scalar>
class Matrix_MxN_Test: public testing::Test
{
public:
    MatrixMxN<Scalar> default_matrix;

    MatrixMxN<Scalar> default_matrix_5x5{5, 5};
    MatrixMxN<Scalar> zero_matrix_5x5{5, 5, static_cast<Scalar>(0)};
    MatrixMxN<Scalar> one_matrix_5x5{5, 5, static_cast<Scalar>(1)};
    MatrixMxN<Scalar> two_matrix_5x5{5, 5, static_cast<Scalar>(2)};

    MatrixMxN<Scalar> default_matrix_5x6{ 5, 6 };
    MatrixMxN<Scalar> zero_matrix_5x6{ 5, 6, static_cast<Scalar>(0) };
    MatrixMxN<Scalar> one_matrix_5x6{ 5, 6, static_cast<Scalar>(1) };
    MatrixMxN<Scalar> two_matrix_5x6{ 5, 6, static_cast<Scalar>(2) };
    
};

typedef testing::Types<float, double, long double> TestTypes;
TYPED_TEST_CASE(Matrix_MxN_Test, TestTypes);

TYPED_TEST(Matrix_MxN_Test, ctor)
{
    //explicit ctor
    EXPECT_EQ(this->one_matrix_5x5(0,0), 1);

    //copy ctor
    auto result = this->one_matrix_5x5;
    EXPECT_EQ(result, this->one_matrix_5x5);

    //default ctor for MxN
    EXPECT_EQ(this->default_matrix_5x5, this->zero_matrix_5x5);

    //default ctor for 0x0
    EXPECT_EQ(this->default_matrix.rows(), 0);
    EXPECT_EQ(this->default_matrix.cols(), 0);
}


TYPED_TEST(Matrix_MxN_Test, dims)
{
    EXPECT_EQ(this->default_matrix_5x5.rows(), 5);
    EXPECT_EQ(this->default_matrix_5x5.cols(), 5);

    EXPECT_EQ(this->default_matrix.rows(), 0);
    EXPECT_EQ(this->default_matrix.cols(), 0);

}

TYPED_TEST(Matrix_MxN_Test, resize)
{
    this->default_matrix.resize(7, 8);
    EXPECT_EQ(this->default_matrix.rows(), 7);
    EXPECT_EQ(this->default_matrix.cols(), 8);

    auto expect_result = MatrixMxN<TypeParam>{ 7, 8 };
    EXPECT_EQ(this->default_matrix, expect_result);

}

TYPED_TEST(Matrix_MxN_Test, operator_get_item)
{
    for (auto i = 0u; i < this->one_matrix_5x5.rows(); ++i)
        for (auto j = 0u; j < this->one_matrix_5x5.cols(); ++j)
            EXPECT_EQ(this->one_matrix_5x5(i, j), 1);

    EXPECT_ANY_THROW(this->one_matrix_5x5(0, this->one_matrix_5x5.cols()));
    EXPECT_ANY_THROW(this->one_matrix_5x5(this->one_matrix_5x5.rows(), 0));
    EXPECT_ANY_THROW(this->one_matrix_5x5(this->one_matrix_5x5.rows(), this->one_matrix_5x5.cols()));

    //const operator[]
    const auto const_one_matrix_5x5 = this->one_matrix_5x5;
    for (auto i = 0u; i < const_one_matrix_5x5.rows(); ++i)
        for (auto j = 0u; j < const_one_matrix_5x5.cols(); ++j)
            EXPECT_EQ(const_one_matrix_5x5(i, j), 1);

    EXPECT_ANY_THROW(const_one_matrix_5x5(0, const_one_matrix_5x5.cols()));
    EXPECT_ANY_THROW(const_one_matrix_5x5(const_one_matrix_5x5.rows(), 0));
    EXPECT_ANY_THROW(const_one_matrix_5x5(const_one_matrix_5x5.rows(), const_one_matrix_5x5.cols()));
}


TYPED_TEST(Matrix_MxN_Test, row_and_col_vector)
{
    auto one_vector_1d_row = this->one_matrix_5x5.rowVector(0);
    auto one_vector_1d_col = this->one_matrix_5x5.colVector(0);
    auto expect_result = VectorND<TypeParam>{5, 1};
    EXPECT_EQ(one_vector_1d_row, expect_result);
    EXPECT_EQ(one_vector_1d_col, expect_result);

    EXPECT_ANY_THROW(this->one_matrix_5x5.rowVector(this->one_matrix_5x5.rows()));
    EXPECT_ANY_THROW(this->one_matrix_5x5.colVector(this->one_matrix_5x5.cols()));
}

TYPED_TEST(Matrix_MxN_Test, operator_add_matrix)
{
    //operator +
    auto result = this->one_matrix_5x5 + this->one_matrix_5x5;
    auto  expect_result = MatrixMxN<TypeParam>{5, 5, 2};
    EXPECT_EQ(result, expect_result);

    //operator +=
    result += this->one_matrix_5x5;
    expect_result = MatrixMxN<TypeParam>{5, 5, 3};
    EXPECT_EQ(result, expect_result);
}

TYPED_TEST(Matrix_MxN_Test, operator_substract_matrix)
{
    //operator -
    auto result = this->one_matrix_5x5 - this->one_matrix_5x5;
    EXPECT_EQ(result, this->zero_matrix_5x5);

    //operator -=
    this->two_matrix_5x5 -= this->one_matrix_5x5;
    EXPECT_EQ(this->two_matrix_5x5, this->one_matrix_5x5);
}

TYPED_TEST(Matrix_MxN_Test, assignment)
{
    this->one_matrix_5x5 = this->zero_matrix_5x5;
    EXPECT_EQ(this->one_matrix_5x5, this->zero_matrix_5x5);
}

TYPED_TEST(Matrix_MxN_Test, operator_eq_neq)
{
    EXPECT_EQ(this->zero_matrix_5x5 == this->zero_matrix_5x5, true);

    auto  another_zero_matrix_5x5 = MatrixMxN<TypeParam>{ 5, 5, static_cast<TypeParam>(0) };
    EXPECT_EQ(this->zero_matrix_5x5 == another_zero_matrix_5x5, true);

    EXPECT_EQ(this->zero_matrix_5x5 != this->one_matrix_5x5, true);
    EXPECT_EQ(this->zero_matrix_5x5 != this->zero_matrix_5x6, true);
}

TYPED_TEST(Matrix_MxN_Test, operator_add_with_scalar)
{
    //to do

    /*
    //operator +
    Vector<TypeParam, 1> result = this->zero_matrix_5x5 + 1;
    EXPECT_EQ(result, this->one_matrix_5x5);

    //operator +=
    result += 1;
    EXPECT_EQ(result, this->two_matrix_5x5);
    */
    
}

TYPED_TEST(Matrix_MxN_Test, operator_substract_with_scalar)
{
    //to do

    /*
    //operator -
    Vector<TypeParam, 1> result = this->one_matrix_5x5 - 1;
    EXPECT_EQ(result, this->zero_matrix_5x5);

    //operator -=
    this->two_matrix_5x5 -= 1;
    EXPECT_EQ(this->two_matrix_5x5, this->one_matrix_5x5);
    */
}

TYPED_TEST(Matrix_MxN_Test, operator_multi_with_scalar)
{
    //operator *
    auto result = this->one_matrix_5x5 * 2;
    EXPECT_EQ(result, this->two_matrix_5x5);

    result = 2 * this->one_matrix_5x5;
    EXPECT_EQ(result, this->two_matrix_5x5);

    //operator *=
    result *= 0;
    EXPECT_EQ(result, this->zero_matrix_5x5);
}

/*
TYPED_TEST(Matrix_MxN_Test, operator_multi_with_vector)
{
    //operator *
    auto result = this->one_matrix_5x5 * VectorND<TypeParam>{5, 2};
    auto expect_result = VectorND<TypeParam>{ 5, 2 };
    EXPECT_EQ(result, expect_result);

    result = 2 * this->one_matrix_5x5 * VectorND<TypeParam>{5, 0};
    expect_result = VectorND<TypeParam>{5, 0 };
    EXPECT_EQ(result, expect_result);

    EXPECT_ANY_THROW(this->default_matrix_5x5*VectorND<TypeParam>{6, 0});
}
*/

TYPED_TEST(Matrix_MxN_Test, operator_multi_with_matrix)
{
    //operator *
    auto result = this->one_matrix_5x5 * this->two_matrix_5x5;
    auto expect_result = MatrixMxN<TypeParam>{ 5, 5, static_cast<TypeParam>(2 * this->one_matrix_5x5.cols()) };
    EXPECT_EQ(result, expect_result);

    result = 2 * this->one_matrix_5x5 * this->zero_matrix_5x5;
    expect_result = MatrixMxN<TypeParam>{ 5, 5, static_cast<TypeParam>(0) };
    EXPECT_EQ(result, this->zero_matrix_5x5);

    result = this->one_matrix_5x5 * this->two_matrix_5x6;
    expect_result = MatrixMxN<TypeParam>( 5, 6, 2 * this->one_matrix_5x5.cols() );
    EXPECT_EQ(result, expect_result);
}

TYPED_TEST(Matrix_MxN_Test, operator_subdivide_with_scalar)
{
    //operator /
    auto result = this->two_matrix_5x5 / 2;
    EXPECT_EQ(result, this->one_matrix_5x5);

    EXPECT_ANY_THROW(result / 0);

    //operator /=
    result /= 1;
    EXPECT_EQ(result, this->one_matrix_5x5);

    EXPECT_ANY_THROW(result /= 0);
}

TYPED_TEST(Matrix_MxN_Test, transpose)
{
    EXPECT_EQ(this->zero_matrix_5x5.transpose(), this->zero_matrix_5x5);
    EXPECT_EQ(this->one_matrix_5x5.transpose(), this->one_matrix_5x5);
    EXPECT_EQ(this->two_matrix_5x5.transpose(), this->two_matrix_5x5);

    EXPECT_NE(this->zero_matrix_5x6.transpose(), this->zero_matrix_5x6);

}

TYPED_TEST(Matrix_MxN_Test, inverse)
{
    EXPECT_ANY_THROW(this->zero_matrix_5x5.inverse());
    EXPECT_ANY_THROW(this->one_matrix_5x5.inverse());
    EXPECT_ANY_THROW(this->two_matrix_5x5.inverse());

    EXPECT_ANY_THROW(this->zero_matrix_5x6.inverse());
}

TYPED_TEST(Matrix_MxN_Test, determinant)
{ 
    EXPECT_EQ(this->zero_matrix_5x5.determinant(), 0);
    EXPECT_EQ(this->one_matrix_5x5.determinant(), 0);
    EXPECT_EQ(this->two_matrix_5x5.determinant(), 0);

    EXPECT_ANY_THROW(this->zero_matrix_5x6.determinant());
    EXPECT_ANY_THROW(this->one_matrix_5x6.determinant());
    EXPECT_ANY_THROW(this->two_matrix_5x6.determinant());
}

TYPED_TEST(Matrix_MxN_Test, trace)
{
    EXPECT_EQ(this->zero_matrix_5x5.trace(), 0);
    EXPECT_EQ(this->one_matrix_5x5.trace(), this->one_matrix_5x5.rows());
    EXPECT_EQ(this->two_matrix_5x5.trace(), 2*this->one_matrix_5x5.rows());

    EXPECT_ANY_THROW(this->zero_matrix_5x6.trace());
    EXPECT_ANY_THROW(this->one_matrix_5x6.trace());
    EXPECT_ANY_THROW(this->two_matrix_5x6.trace());

}

TYPED_TEST(Matrix_MxN_Test, double_contraction)
{
    EXPECT_EQ(this->zero_matrix_5x5.doubleContraction(this->one_matrix_5x5), 0);
    EXPECT_EQ(this->one_matrix_5x5.doubleContraction(this->two_matrix_5x5), 2*this->one_matrix_5x5.rows()*this->one_matrix_5x5.cols());

    EXPECT_ANY_THROW(this->one_matrix_5x5.doubleContraction(this->two_matrix_5x6));
}

TYPED_TEST(Matrix_MxN_Test, frobenius_norm)
{
    EXPECT_EQ(this->zero_matrix_5x5.frobeniusNorm(), 0);
    EXPECT_NEAR(this->one_matrix_5x5.frobeniusNorm(), sqrt(this->one_matrix_5x5.rows()*this->one_matrix_5x5.cols()), 1.0e-6);
    EXPECT_NEAR(this->two_matrix_5x5.frobeniusNorm(), sqrt(4*this->one_matrix_5x5.rows()*this->one_matrix_5x5.cols()), 1.0e-6);

    auto  negative_one_matrix_5x5 = MatrixMxN<TypeParam>{5, 5, static_cast<TypeParam>(-1) };
    EXPECT_NEAR(negative_one_matrix_5x5.frobeniusNorm(), sqrt(this->one_matrix_5x5.rows()*this->one_matrix_5x5.cols()), 1.0e-6);

    EXPECT_NEAR(this->one_matrix_5x6.frobeniusNorm(), sqrt(this->one_matrix_5x6.rows()*this->one_matrix_5x6.cols()), 1.0e-6);
}

TYPED_TEST(Matrix_MxN_Test, operator_pre_substract)
{
    //to do

    /*
    Vector<TypeParam, 1> negative_one_matrix_5x5{static_cast<TypeParam>(-1)};
    EXPECT_EQ(-negative_one_matrix_5x5, this->one_matrix_5x5);
    EXPECT_EQ(-this->one_matrix_5x5, negative_one_matrix_5x5);
    */
}

TYPED_TEST(Matrix_MxN_Test, singular_value_decomposition)
{
    // to do
}

TYPED_TEST(Matrix_MxN_Test, eigen_decomposition)
{
    // to do
}
