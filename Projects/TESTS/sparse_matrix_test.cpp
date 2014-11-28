/*
 * @file sparse_matrix_test.cpp 
 * @brief Test the SparseMatrix class. when you want to use the test file to test the SparseMatrix class,
 * the best way is to use the cmd to excute the PhysikaTestDebug.exe
 * @author Liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <vector>
#include <ctime>
#include "Physika_Dependency/Eigen/Eigen"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Core/Timer/timer.h"
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"
#include "Physika_Core/Matrices/matrix_MxN.h"

#define Max1 2000
#define Max2 2000
#define Max3 156
#define Max  10000
#define Maxv 10
#define mytype double

using namespace std;
using std::cout;

template <class Scalar>
void print(vector<Scalar> &v)
{
    for(int i=0; i<v.size();++i)
        cout<<v[i]<<" ";
    cout<<endl;
}
void compare(const Physika::SparseMatrix<mytype> &a, const Eigen::SparseMatrix<mytype> &b)
{
    /*if (a.nonZeros() != b.nonZeros())
    {
        cout<<"a.nonzeros:"<<a.nonZeros()<<endl;
        cout << "b.nonZeros:" << b.nonZeros() << endl;
        cout << "correctness bad!" << endl;
        return;
    }*/
        std::vector<Physika::Trituple<mytype>> v;
        bool correctness = true;
        for(unsigned int i = 0;i<a.rows();++i)
        {
            v = a.getRowElements(i);
            for(unsigned int j=0;j<v.size();++j)
            {
                int row = v[j].row(), col = v[j].col();
                mytype value = v[j].value();
                if(b.coeff(row,col) != value)
                {
                    cout<<"eror: "<<row<<' '<<col<<" value: psm "<<value<<" "<<b.coeff(row,col)<<endl;
                    correctness = false;
                    break;
                }
            }
            for (Eigen::SparseMatrix<mytype>::InnerIterator it(b, i); it; ++it)
            {
                if (it.value() != a(it.row(), it.col()))
                {
                    cout << "eror: " << it.row() << ' ' << it.col() << " value: psm " << a(it.row(),it.col()) << " " << it.value() << endl;
                    correctness = false;
                    break;
                }
            }
        }
        if(correctness) cout<<"correctness OK!"<<endl;
        else cout<<"correctness bad!"<<endl;
}
void compare(const Physika::VectorND<mytype> &a, const Eigen::SparseVector<mytype> &b)
{
    for (unsigned int i = 0; i < a.dims(); ++i)
    {
        if (a[i] != b.coeff(i))
        {
            cout << "uncorrectly vector multiply a sparsematrix" << endl;
            return;
        }
    }
    cout << "correctly multiply" << endl;
    return ;
}
int main()
{
    //construct function check
    Physika::SparseMatrix<mytype> m11(5, 5, false);
    m11.setEntry(0, 0, 1);
    m11.setEntry(1, 1, 1);
    m11.setEntry(2, 2, 1);
    m11.setEntry(3, 3, 1);
    m11.setEntry(4, 4, 1);
    cout << "SparseMatrix<mytype> m11 as follow:" << endl;
    cout << m11 << endl;
    Physika::SparseMatrix<mytype> ps1(Max1, Max2, false);
    Eigen::SparseMatrix<mytype> es1(Max1, Max2);
    Physika::Timer timer;
    srand(time(NULL));
    cout << "correctness of insert operation tests:";
    for (unsigned int i = 0; i<Max; ++i)
    {
        unsigned int row = rand() % Max1;
        unsigned int col = rand() % Max2;
        mytype v = rand() % Max + 1;
        ps1.setEntry(row, col, v);
        es1.coeffRef(row, col) = v;
    }
    compare(ps1, es1);
    Physika::SparseMatrix<mytype> ps2(ps1);
    cout << "correctness of operator=:";
    compare(ps2, es1);
    //get rowElements and colElements function test
    //iterator test
    for (Physika::SparseMatrixIterator<mytype> it(ps1, Max1 / 2); it; ++it)
    {
        it.row();
        it.value();
        it.col();
        cout <<"<"<< it.row();
        cout << ", " << it.col();
        cout << ", " << it.value() <<"> ";
    }
    cout << endl;
    print(ps1.getRowElements(Max1 / 2));
    //remove function test
    for (unsigned int i = 0; i<Max/2; ++i)
    {
        unsigned int row = rand() % Max1;
        unsigned int col = rand() % Max2;
        ps1.remove(row, col);
        es1.coeffRef(row, col) = 0;
    }
    cout << "remove function correctness:";
    compare(ps1, es1);
    //overall test
    /*
    {
        cout <<"overall test"<< endl;
        Physika::SparseMatrix<mytype> m1(5, 5);
        m1.setEntry(0, 0, 1);
        m1.setEntry(1, 1, 1);
        m1.setEntry(2, 2, 1);
        m1.setEntry(3, 3, 1);
        m1.setEntry(4, 4, 1);
        cout << "SparseMatrix<mytype> m1 as follow:" << endl;
        cout << m1 << endl;
        Physika::SparseMatrix<mytype> m2(m1);
        m2.setEntry(0, 4, 5);
        cout << "SparseMatrix<mytype> m2 as follow:" << endl;
        cout << m2 << endl;
        cout << "m1 rows:" << m1.rows() << endl;
        cout << "m1 cols:" << m1.cols() << endl;
        cout << "m1 nonzeros:" << m1.nonZeros() << endl;
        cout << "m1(3,3):" << m1(3, 3) << endl;
        cout << "m1 + m2:" << endl;
        cout << m1 + m2 << endl;
        cout << "m1 += m2:" << endl;
        cout << (m1 += m2) << endl;
        cout << "m1 - m2:" << endl;
        cout << m1 - m2 << endl;
        cout << "m1 -= m2:" << endl;
        cout << (m1 -= m2) << endl;
        cout << "m1 == m2? " << (m1 == m2) << endl;
        cout << "copy m1 to m2" << endl;
        cout << (m2 = m1) << endl;
        cout << "m1 == m2? " << (m1 == m2) << endl;
        cout << "m1 != m2? " << (m1 != m2) << endl;
        cout << "m1 * 3" << endl;
        cout << (m1 * 3) << endl;
        cout << "m1*=3" << endl;
        cout << (m1 *= 3) << endl;
        cout << "m1 / 4" << endl;
        cout << (m1 / 4) << endl;
        cout << "m1/=4" << endl;
        cout << (m1 /= 4) << endl;
        cout << "m1.getRowElements(2):";
        print(m1.getRowElements(2));
        cout << "m1.getColElements(1):";
        print(m1.getColElements(1));
        m1.remove(3, 3);
        cout << "remove (3,3):" << endl;
        cout << m1 << endl;
        Physika::SparseMatrix<mytype> m3(5, 4);
        m3.setEntry(1, 1, 1);
        m3.setEntry(2, 2, 1);
        m3.setEntry(3, 3, 1);
        m3.setEntry(0, 0, 1);
        m3.setEntry(4, 3, 1);
        cout << "m3 as follow" << endl;
        cout << m3 << endl;
        cout << "m1 * m3 =" << endl;
        cout << m1*m3 << endl;
        cout << "vec[5]:" << endl;
        Physika::VectorND<mytype> vec(5, 2.0);
        cout << vec << endl;
        cout << (m1*vec) << endl;
        m1.setEntry(1, 2, 5);
        cout << "new m1" << endl;
        cout << m1 << endl;
        cout << "m1.transpose():" << endl;
        cout << m1.transpose() << endl;
        cout << "m1" << endl;
        cout << m1 << endl;
        cout << "vec * m1" << endl;
        cout << (vec * m1) << endl;
        //getchar();
    }
    */
    //efficiency test
    //insert psm esm
    
    cout << "insert in order" << endl;
    Physika::SparseMatrix<mytype> psm(Max1, Max2);
    Eigen::SparseMatrix<mytype> esm(Max1, Max2);
    timer.startTimer();
    for (unsigned int i = 0; i < Max1; ++i)
    {
        vector<Physika::Trituple<mytype>> vec_1 = ps1.getRowElements(i);
        for (unsigned int j = 0; j < vec_1.size(); ++j)
        {
            psm.setEntry(vec_1[j].row(), vec_1[j].col(), vec_1[j].value());
        }
    }
    timer.stopTimer();
    cout << "psm insert time: " << timer.getElapsedTime() << endl;
    for (unsigned int i = 0; i < Max1; ++i)
    {
        vector<Physika::Trituple<mytype>> vec_1 = ps1.getRowElements(i);
        for (unsigned int j = 0; j < vec_1.size(); ++j)
        {
            esm.insert(vec_1[j].row(), vec_1[j].col()) = vec_1[j].value();
        }
    }
    timer.stopTimer();
    cout << "esm insert time: " << timer.getElapsedTime() << endl;
    compare(psm, esm);
    //transpose efficiency psm1 esm1
    Physika::SparseMatrix<mytype> psm1(0, 0, 0);
    Eigen::SparseMatrix<mytype> esm1;
    timer.startTimer();
    psm1 = psm.transpose();
    timer.stopTimer();
    cout << "psm transpose time: " << timer.getElapsedTime() << endl;
    
    timer.startTimer(); 
    psm.transpose();
    timer.stopTimer();
    cout <<"only pushback the elements: "<< timer.getElapsedTime() << endl;
    
    timer.startTimer();
    esm1 = esm.transpose();
    timer.stopTimer();
    cout << "esm transpose time: " << timer.getElapsedTime() << endl;
    compare(psm1, esm1);
    //multiply efficiency psm2,esm2
    Physika::SparseMatrix<mytype> psm2;
    Eigen::SparseMatrix<mytype> esm2;
    timer.startTimer();
    esm2 = esm*esm1;
    timer.stopTimer();
    cout << "esm multiply time: " << timer.getElapsedTime() << endl;
    timer.startTimer();
    psm2 = psm*psm1;
    timer.stopTimer();
    cout << "psm multiply time: " << timer.getElapsedTime() << endl;
    compare(psm2, esm2);
    return 0;
}