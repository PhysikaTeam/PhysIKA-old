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
#include <ctime>
#include "Physika_Dependency/Eigen/Sparse"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#define Max 10000

using namespace std;

template <typename Scalar>
void print(vector<Scalar> &v)
{
    for(int i=0; i<v.size();++i)
        cout<<v[i]<<" ";
    cout<<endl;
}

int main()
{
	/*cout<<"基本功能测试"<<endl;
	Physika::SparseMatrix<float> m1(5,5);
    m1.setEntry(0,0,1);
    m1.setEntry(1,1,1);
    m1.setEntry(2,2,1);
    m1.setEntry(3,3,1);
    m1.setEntry(4,4,1);
    cout<<"SparseMatrix<float> m1 as follow:"<<endl;
    cout<<m1<<endl;
    Physika::SparseMatrix<float> m2(m1);
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
    Physika::SparseMatrix<float> m3(5,4);
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
	m1.setEntry(1,2,5);
	cout<<"new m1"<<endl;
	cout<<m1<<endl;
	cout<<"m1.transpose():"<<endl;
	cout<<m1.transpose()<<endl;
	cout<<(vec * m1)<<endl;
    //getchar();
	*/


	srand(time(NULL));
	cout<<"特定功能 高级测试"<<endl;

	cout<<"insert effectiveness"<<endl;
	Physika::SparseMatrix<float> psm(Max,Max);
	Eigen::SparseMatrix<float> esm(Max, Max);
	clock_t start = clock();
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max;
		unsigned int col = rand()%Max;
		float v = rand()%Max+1;
		psm.setEntry(row,col,v);
		//esm.coeffRef(row,col) = v;
	}
	clock_t end = clock();
	cout<<"psm insert:"<<static_cast<double>(end-start)<<endl;
	start = clock();
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max;
		unsigned int col = rand()%Max;
		float v = rand()%Max+1;
		//psm.setEntry(row,col,v);
		esm.coeffRef(row,col) = v;
	}
	end = clock();
	cout<<"esm insert:"<<static_cast<double>(end-start)<<endl;

		cout<<"测量矩阵转置正确性及时间效率"<<endl;
	start = clock();
	esm.transpose();
	end = clock();
	cout<<"esm time consuming:"<<(static_cast<double>(end - start))<<endl;
	start = clock();
	psm.transpose();
	 end = clock();
	cout<<"psm time consuming:"<<(static_cast<double>(end - start))<<endl;

	//cout<<"赋值语句效率"<<endl;
	start = clock();
	Physika::SparseMatrix<float> psm3 ;
	end = clock();
	//cout<<"psm assign time consuming:"<<static_cast<double>(end - start)<<endl;
	start = clock();
	Eigen::SparseMatrix<float> esm3 ;
	end = clock();
	//cout<<"esm assign time consuming:"<<static_cast<double>(end - start)<<endl;

	cout<<"multiply effectiveness"<<endl;
	Physika::SparseMatrix<float> psm2(Max,Max);
	Eigen::SparseMatrix<float> esm2(Max, Max);
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max;
		unsigned int col = rand()%Max;
		float v =rand()%Max+1;
		psm2.setEntry(row,col,v);
		esm2.coeffRef(row,col) = v;
	}
	start = clock();
	psm3 = psm2*psm2;
	end = clock();
	cout<<"physika * consume time:"<<static_cast<double>(end - start)<<endl;
	start = clock();
	esm3 = esm2*esm2;
	end = clock();
	cout<<"eigen * consume time:"<<static_cast<double>(end - start)<<endl;

	cout<<"correctness multiply"<<endl;
	std::vector<Physika::Trituple<float>> v;
	bool correctness = true;
	for(unsigned int i = 0;i<psm3.rows();++i)
	{
		v = psm3.getRowElements(i);
		for(unsigned int j=0;j<v.size();++j)
		{
			int row = v[j].row_, col = v[j].col_;
			float value = v[j].value_;
			if(esm3.coeff(row,col) != value)
			{
				cout<<"eror:"<<row<<' '<<col<<endl;
				correctness = false;
				break;
			}
		}
	}
	if(correctness) cout<<"correctness OK!"<<endl;
	else cout<<"correctness bad!"<<endl;



	/*psm = psm.transpose();
	esm = esm.transpose();
	std::vector<Physika::Trituple<float>> v;
	bool correctness = true;
	for(unsigned int i = 0;i<psm.rows();++i)
	{
		v = psm.getRowElements(i);
		for(unsigned int j=0;j<v.size();++j)
		{
			int row = v[j].row_, col = v[j].col_;
			float value = v[j].value_;
			if(esm.coeff(row,col) != value)
			{
				cout<<"eror:"<<row<<' '<<col<<endl;
				correctness = false;
			}
		}
	}
	if(correctness) cout<<"correctness OK!"<<endl;
	else cout<<"correctness bad!"<<endl;
	*/
    return 0;
}