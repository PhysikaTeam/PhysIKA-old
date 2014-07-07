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
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Vectors/vector_Nd.h"
using namespace std;
using Physika::SparseMatrix;

template <typename Scalar>
void print(vector<Scalar> &v)
{
    for(int i=0; i<v.size();++i)
        cout<<v[i]<<" ";
    cout<<endl;
}

int main()
{
    SparseMatrix<float> m1(5,5);
    m1.setEntry(0,0,1);
    m1.setEntry(1,1,1);
    m1.setEntry(2,2,1);
    m1.setEntry(3,3,1);
    m1.setEntry(4,4,1);
    cout<<"SparseMatrix<float> m1 as follow:"<<endl;
    cout<<m1<<endl;
    SparseMatrix<float> m2(m1);
    m2.setEntry(0,4,5);
    cout<<"SparseMatrix<float> m2 as follow:"<<endl;
    cout<<m2<<endl;
    cout<<"m1 rows:"<<m1.rows()<<endl;
    cout<<"m1 cols:"<<m1.cols()<<endl;
    cout<<"m1 nonzeros:"<<m1.nonZeros()<<endl;
    cout<<"m1(3,3):"<<m1(3,3)<<endl;
    cout<<"m1 + m2:"<<endl;
    cout<<m1+m2<<endl;
    cout<<"m1 += m2:"<<endl;
    cout<<(m1 += m2)<<endl;
    cout<<"m1 - m2:"<<endl;
    cout<<m1 - m2<<endl;
    cout<<"m1 -= m2:"<<endl;
    cout<<(m1 -= m2)<<endl;

    cout<<"m1 == m2? "<<(m1 == m2) <<endl;
    cout<<"copy m1 to m2"<<endl;
    cout<<(m2=m1)<<endl;
    cout<<"m1 == m2? "<<(m1 == m2)<<endl;
    cout<<"m1 != m2? "<<(m1 != m2)<<endl;

    cout<<"m1 * 3"<<endl;
    cout<<(m1*3)<<endl;
    cout<<"m1*=3"<<endl;
    cout<<(m1*=3)<<endl;

    cout<<"m1 / 4"<<endl;
    cout<<(m1/4)<<endl;
    cout<<"m1/=4"<<endl;
    cout<<(m1/=4)<<endl;

    cout<<"m1.getRowElements(2):";
    print(m1.getRowElements(2));

    cout<<"m1.getColElements(1):";
    print(m1.getColElements(1));
    
    m1.remove(3,3);
    cout<<"remove (3,3):"<<endl;
    cout<<m1<<endl;

    SparseMatrix<float> m3(5,4);
    m3.setEntry(1,1,1);
    m3.setEntry(2,2,1);
    m3.setEntry(3,3,1);
    m3.setEntry(0,0,1);
    m3.setEntry(4,3,1);
    cout<<"m3 as follow"<<endl;
    cout<<m3<<endl;

    cout<<"m1 * m3 ="<<endl;
    cout<<m1*m3<<endl;

    cout<<"vec[5]:"<<endl;
    Physika::VectorND<float> vec(5,2.0);
    cout<<vec<<endl;
    cout<<(m1*vec)<<endl;
    //getchar();
    return 0;
}