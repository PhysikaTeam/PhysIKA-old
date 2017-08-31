/*
* @file matrix_test.cpp
* @brief unit test for matrix.
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
#include <limits>

#include "gtest/gtest.h"

#include "Physika_Core/Utilities/physika_assert.h"

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
class MatrixType
{
public:
    using Scalar = SCALAR;
    static constexpr int Dim = DIM;
};

template <typename MatrixType>
class Matrix_Test: public testing::Test
{
public:
    Matrix_Test();

    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> default_matrix;
    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> zero_matrix{0};
    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> one_matrix{1};
    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> two_matrix{2};

    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> ascend_matrix;

    SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim> identity_matrix = SquareMatrix<typename MatrixType::Scalar, MatrixType::Dim>::identityMatrix();
    
};

template<typename MatrixType>
Matrix_Test<MatrixType>::Matrix_Test()
{
    for (auto i = 0u; i < MatrixType::Dim; ++i)
        for (auto j = 0u; j < MatrixType::Dim; ++j)
            this->ascend_matrix(i, j) = i * MatrixType::Dim + j;
}

TYPED_TEST_CASE_P(Matrix_Test);

using MatrixTypes_1x1 = testing::Types< MatrixType<unsigned char, 1>,
                                        MatrixType<unsigned short, 1>,
                                        MatrixType<unsigned int, 1>,
                                        MatrixType<unsigned long, 1>,
                                        MatrixType<unsigned long, 1>,
                                        MatrixType<long, 1>,
                                        MatrixType<signed char, 1>,
                                        MatrixType<short, 1>,
                                        MatrixType<int, 1>,
                                        MatrixType<long, 1>,
                                        MatrixType<long long, 1>,
                                        MatrixType<float, 1>,
                                        MatrixType<double, 1>,
                                        MatrixType<long double, 1>
                                      >;

using MatrixTypes_2x2 = testing::Types< MatrixType<unsigned char, 2>,
                                        MatrixType<unsigned short, 2>,
                                        MatrixType<unsigned int, 2>,
                                        MatrixType<unsigned long, 2>,
                                        MatrixType<unsigned long, 2>,
                                        MatrixType<long, 2>,
                                        MatrixType<signed char, 2>,
                                        MatrixType<short, 2>,
                                        MatrixType<int, 2>,
                                        MatrixType<long, 2>,
                                        MatrixType<long long, 2>,
                                        MatrixType<float, 2>,
                                        MatrixType<double, 2>,
                                        MatrixType<long double, 2>
                                      >;

using MatrixTypes_3x3 = testing::Types< MatrixType<unsigned char, 3>,
                                        MatrixType<unsigned short, 3>,
                                        MatrixType<unsigned int, 3>,
                                        MatrixType<unsigned long, 3>,
                                        MatrixType<unsigned long, 3>,
                                        MatrixType<long, 3>,
                                        MatrixType<signed char, 3>,
                                        MatrixType<short, 3>,
                                        MatrixType<int, 3>,
                                        MatrixType<long, 3>,
                                        MatrixType<long long, 3>,
                                        MatrixType<float, 3>,
                                        MatrixType<double, 3>,
                                        MatrixType<long double, 3>
                                      >;

using MatrixTypes_4x4 = testing::Types< MatrixType<unsigned char, 4>,
                                        MatrixType<unsigned short, 4>,
                                        MatrixType<unsigned int, 4>,
                                        MatrixType<unsigned long, 4>,
                                        MatrixType<unsigned long, 4>,
                                        MatrixType<long, 4>,
                                        MatrixType<signed char, 4>,
                                        MatrixType<short, 4>,
                                        MatrixType<int, 4>,
                                        MatrixType<long, 4>,
                                        MatrixType<long long, 4>,
                                        MatrixType<float, 4>,
                                        MatrixType<double, 4>,
                                        MatrixType<long double, 4>
                                      >;

TYPED_TEST_P(Matrix_Test, ctor)
{
    //explicit ctor
    for (auto i = 0u; i < TypeParam::Dim; ++i)
        for (auto j = 0u; j < TypeParam::Dim; ++j)
            EXPECT_EQ(this->ascend_matrix(i, j), i*TypeParam::Dim+j);
    

    //copy ctor
    auto result = this->one_matrix;
    EXPECT_EQ(result, this->one_matrix);

    //default ctor
    EXPECT_EQ(this->default_matrix, this->zero_matrix);
}


TYPED_TEST_P(Matrix_Test, dims)
{
    EXPECT_EQ(this->default_matrix.rows(), TypeParam::Dim);
    EXPECT_EQ(this->default_matrix.cols(), TypeParam::Dim);
}

TYPED_TEST_P(Matrix_Test, operator_get_item)
{
    for (auto i = 0u; i < TypeParam::Dim; ++i)
        for (auto j = 0u; j < TypeParam::Dim; ++j)
            EXPECT_EQ(this->ascend_matrix(i, j), TypeParam::Dim*i + j);

    EXPECT_ANY_THROW(this->one_matrix(0, this->one_matrix.cols()));
    EXPECT_ANY_THROW(this->one_matrix(this->one_matrix.rows(), 0));
    EXPECT_ANY_THROW(this->one_matrix(this->one_matrix.rows(), this->one_matrix.cols()));

    //const operator(i, j)
    const auto const_ascend_matrix = this->ascend_matrix;

    for (auto i = 0u; i < TypeParam::Dim; ++i)
        for (auto j = 0u; j < TypeParam::Dim; ++j)
            EXPECT_EQ(const_ascend_matrix(i, j), TypeParam::Dim * i + j);

    EXPECT_ANY_THROW(const_ascend_matrix(0, const_ascend_matrix.cols()));
    EXPECT_ANY_THROW(const_ascend_matrix(const_ascend_matrix.rows(), 0));
    EXPECT_ANY_THROW(const_ascend_matrix(const_ascend_matrix.rows(), const_ascend_matrix.cols()));
}


TYPED_TEST_P(Matrix_Test, row_and_col_vector)
{
    auto one_vector_row = this->one_matrix.rowVector(0);
    auto one_vector_col = this->one_matrix.colVector(0);
    auto expect_result = Vector<typename TypeParam::Scalar, TypeParam::Dim>{ 1 };
    EXPECT_EQ(one_vector_row, expect_result);
    EXPECT_EQ(one_vector_col, expect_result);

    EXPECT_ANY_THROW(this->one_matrix.rowVector(TypeParam::Dim));
    EXPECT_ANY_THROW(this->one_matrix.colVector(TypeParam::Dim));
}

TYPED_TEST_P(Matrix_Test, operator_add_matrix)
{
    //operator +
    auto result = this->one_matrix + this->one_matrix;
    auto  expect_result = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{2};
    EXPECT_EQ(result, expect_result);

    //operator +=
    result += this->one_matrix;
    expect_result = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{3};
    EXPECT_EQ(result, expect_result);
}

TYPED_TEST_P(Matrix_Test, operator_substract_matrix)
{
    //operator -
    auto result = this->one_matrix - this->one_matrix;
    EXPECT_EQ(result, this->zero_matrix);

    //operator -=
    this->two_matrix -= this->one_matrix;
    EXPECT_EQ(this->two_matrix, this->one_matrix);
}

TYPED_TEST_P(Matrix_Test, assignment)
{
    this->one_matrix = this->zero_matrix;
    EXPECT_EQ(this->one_matrix, this->zero_matrix);
}

TYPED_TEST_P(Matrix_Test, operator_eq_neq)
{
    EXPECT_EQ(this->zero_matrix == this->zero_matrix, true);

    auto  another_zero_matrix = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{ 0 };
    EXPECT_EQ(this->zero_matrix== another_zero_matrix, true);

    EXPECT_EQ(this->zero_matrix != this->one_matrix, true);
}

TYPED_TEST_P(Matrix_Test, operator_add_with_scalar)
{
    //to do

    /*
    //operator +
    Vector<TypeParam, 1> result = this->zero_matrix + 1;
    EXPECT_EQ(result, this->one_matrix);

    //operator +=
    result += 1;
    EXPECT_EQ(result, this->two_matrix);
    */
    
}

TYPED_TEST_P(Matrix_Test, operator_substract_with_scalar)
{
    //to do

    /*
    //operator -
    Vector<TypeParam, 1> result = this->one_matrix - 1;
    EXPECT_EQ(result, this->zero_matrix);

    //operator -=
    this->two_matrix -= 1;
    EXPECT_EQ(this->two_matrix, this->one_matrix);
    */
}

TYPED_TEST_P(Matrix_Test, operator_multi_with_scalar)
{
    //operator *
    auto result = this->one_matrix * 2;
    EXPECT_EQ(result, this->two_matrix);

    result = 2 * this->one_matrix;
    EXPECT_EQ(result, this->two_matrix);

    //operator *=
    result *= 0;
    EXPECT_EQ(result, this->zero_matrix);
}

TYPED_TEST_P(Matrix_Test, operator_multi_with_vector)
{
    //operator *
    auto result = this->one_matrix * Vector<typename TypeParam::Scalar, TypeParam::Dim>{2};
    auto expect_result = Vector<typename TypeParam::Scalar, TypeParam::Dim>{ 2 * TypeParam::Dim };
    EXPECT_EQ(result, expect_result);

    result = 2 * this->one_matrix * Vector<typename TypeParam::Scalar, TypeParam::Dim>{0};
    expect_result = Vector<typename TypeParam::Scalar, TypeParam::Dim>{ 0 };
    EXPECT_EQ(result, expect_result);
}

TYPED_TEST_P(Matrix_Test, operator_multi_with_matrix)
{
    //operator *
    auto result = this->one_matrix * this->two_matrix;
    auto expect_result = SquareMatrix<typename TypeParam::Scalar, TypeParam::Dim>{ 2 * TypeParam::Dim };
    EXPECT_EQ(result, expect_result);

    result = 2 * this->one_matrix * this->zero_matrix;
    EXPECT_EQ(result, this->zero_matrix);
}

TYPED_TEST_P(Matrix_Test, operator_subdivide_with_scalar)
{
    //operator /
    auto result = this->two_matrix / 2;
    EXPECT_EQ(result, this->one_matrix);

    EXPECT_ANY_THROW(result / 0);

    //operator /=
    result /= 1;
    EXPECT_EQ(result, this->one_matrix);

    EXPECT_ANY_THROW(result /= 0);
}

TYPED_TEST_P(Matrix_Test, transpose)
{
    EXPECT_EQ(this->zero_matrix.transpose(), this->zero_matrix);
    EXPECT_EQ(this->one_matrix.transpose(), this->one_matrix);
    EXPECT_EQ(this->two_matrix.transpose(), this->two_matrix);
    EXPECT_EQ(this->identity_matrix.transpose(), this->identity_matrix);

    if (TypeParam::Dim == 1)
        EXPECT_EQ(this->ascend_matrix.transpose(), this->ascend_matrix);
    else
        EXPECT_NE(this->ascend_matrix.transpose(), this->ascend_matrix);
    

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
void inverse_test(const Matrix_Test<MatrixType<Scalar, 1> > * test)
{
    auto  expect_result = SquareMatrix<Scalar, 1>{ static_cast<Scalar>(0.5) };
    EXPECT_EQ(test->two_matrix.inverse(), expect_result);
    EXPECT_EQ(test->one_matrix.inverse(), test->one_matrix);
}

template <typename Scalar>
void inverse_test(const Matrix_Test<MatrixType<Scalar, 2> > * test)
{
    EXPECT_ANY_THROW(test->zero_matrix.inverse());
    EXPECT_ANY_THROW(test->one_matrix.inverse());
    EXPECT_ANY_THROW(test->two_matrix.inverse());

    auto expect_result = SquareMatrix<Scalar, 2>(-1.5, 0.5, 1, 0);
    EXPECT_EQ(test->ascend_matrix.inverse(), expect_result);

    EXPECT_EQ(test->identity_matrix.inverse(), test->identity_matrix);
}

template <typename Scalar, int Dim>
void inverse_test(const Matrix_Test<MatrixType<Scalar, Dim> > * test)
{
    PHYSIKA_STATIC_ASSERT(Dim == 3 || Dim == 4, "inverse_test requires Dim = 1||2||3||4");
    
    EXPECT_ANY_THROW(test->zero_matrix.inverse());
    EXPECT_ANY_THROW(test->one_matrix.inverse());
    EXPECT_ANY_THROW(test->two_matrix.inverse());
    EXPECT_ANY_THROW(test->ascend_matrix.inverse());

    EXPECT_EQ(test->identity_matrix.inverse(), test->identity_matrix);
}


TYPED_TEST_P(Matrix_Test, inverse)
{
    inverse_test(this);

    EXPECT_EQ(this->identity_matrix.inverse(), this->identity_matrix);
    EXPECT_ANY_THROW(this->zero_matrix.inverse());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TYPED_TEST_P(Matrix_Test, determinant)
{
    if (TypeParam::Dim == 1)
    {
        EXPECT_EQ(this->zero_matrix.determinant(), 0);
        EXPECT_EQ(this->one_matrix.determinant(), 1);
        EXPECT_EQ(this->two_matrix.determinant(), 2);
        EXPECT_EQ(this->identity_matrix.determinant(), 1);
    }
    else
    {
        EXPECT_EQ(this->zero_matrix.determinant(), 0);
        EXPECT_EQ(this->one_matrix.determinant(), 0);
        EXPECT_EQ(this->two_matrix.determinant(), 0);
        EXPECT_EQ(this->identity_matrix.determinant(), 1);
    }
    
}

TYPED_TEST_P(Matrix_Test, trace)
{
    EXPECT_EQ(this->zero_matrix.trace(), 0);
    EXPECT_EQ(this->one_matrix.trace(), 1*TypeParam::Dim);
    EXPECT_EQ(this->two_matrix.trace(), 2*TypeParam::Dim);

}

TYPED_TEST_P(Matrix_Test, double_contraction)
{
    EXPECT_EQ(this->zero_matrix.doubleContraction(this->one_matrix), 0);
    EXPECT_EQ(this->one_matrix.doubleContraction(this->two_matrix), 2*TypeParam::Dim*TypeParam::Dim);
}

TYPED_TEST_P(Matrix_Test, frobenius_norm)
{
    using Scalar = typename TypeParam::Scalar;
    constexpr int Dim = TypeParam::Dim;

    EXPECT_EQ(this->zero_matrix.frobeniusNorm(), 0);
    EXPECT_EQ(this->one_matrix.frobeniusNorm(), static_cast<Scalar>(sqrt(Dim*Dim)));
    EXPECT_EQ(this->two_matrix.frobeniusNorm(), static_cast<Scalar>(sqrt(4 * Dim*Dim)));

    auto  negative_one_matrix = SquareMatrix<Scalar, Dim>{ static_cast<Scalar>(-1) };
    EXPECT_EQ(negative_one_matrix.frobeniusNorm(), static_cast<Scalar>(sqrt(Dim*Dim)));
}

TYPED_TEST_P(Matrix_Test, operator_pre_substract)
{
    //to do

    /*
    Vector<TypeParam, 1> negative_one_matrix{static_cast<TypeParam>(-1)};
    EXPECT_EQ(-negative_one_matrix, this->one_matrix);
    EXPECT_EQ(-this->one_matrix, negative_one_matrix);
    */
}

TYPED_TEST_P(Matrix_Test, identity_matrix_test)
{
    for (auto i = 0u; i < TypeParam::Dim; ++i)
        for (auto j = 0u; j < TypeParam::Dim; ++j)
        {
            auto value = this->identity_matrix(i, j);
            if (i == j)
                EXPECT_EQ(value, 1);
            else
                EXPECT_EQ(value, 0);
        }
    
}

REGISTER_TYPED_TEST_CASE_P(Matrix_Test, ctor, dims, operator_get_item, row_and_col_vector, 
                                        operator_add_matrix, operator_substract_matrix, assignment, 
                                        operator_eq_neq, operator_add_with_scalar, operator_substract_with_scalar,
                                        operator_multi_with_scalar, operator_multi_with_vector, operator_multi_with_matrix,
                                        operator_subdivide_with_scalar, transpose, inverse, determinant,
                                        trace, double_contraction, frobenius_norm, operator_pre_substract, identity_matrix_test
                          );

INSTANTIATE_TYPED_TEST_CASE_P(Dim_1x1_, Matrix_Test, MatrixTypes_1x1);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_2x2_, Matrix_Test, MatrixTypes_2x2);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_3x3_, Matrix_Test, MatrixTypes_3x3);
INSTANTIATE_TYPED_TEST_CASE_P(Dim_4x4_, Matrix_Test, MatrixTypes_4x4);


