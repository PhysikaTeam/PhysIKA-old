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
#define Max1 10000
#define Max2 10000
#define Max  100000

using namespace std;

template <typename Scalar>
void print(vector<Scalar> &v)
{
    for(int i=0; i<v.size();++i)
        cout<<v[i]<<" ";
    cout<<endl;
}
void compare(const Physika::SparseMatrix<float> &a, const Eigen::SparseMatrix<float> &b)
{
		std::vector<Physika::Trituple<float>> v;
		bool correctness = true;
		for(unsigned int i = 0;i<a.rows();++i)
		{
			v = a.getRowElements(i);
			for(unsigned int j=0;j<v.size();++j)
			{
				int row = v[j].row_, col = v[j].col_;
				float value = v[j].value_;
				if(b.coeff(row,col) != value)
				{
					cout<<"eror:"<<row<<' '<<col<<endl;
					correctness = false;
					break;
				}
			}
		}
		if(correctness) cout<<"correctness OK!"<<endl;
		else cout<<"correctness bad!"<<endl;
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
	clock_t start,end;

	srand(time(NULL));
	cout<<"特定功能 高级测试"<<endl;

	cout<<"insert "<<endl;
	Physika::SparseMatrix<float> psm(Max1,Max2);
	Eigen::SparseMatrix<float> esm(Max1, Max2);
	start = clock();
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max1;
		unsigned int col = rand()%Max2;
		float v = rand()%Max+1;
		psm.setEntry(row,col,v);
		//esm.coeffRef(row,col) = v;
	}
	end = clock();
	cout<<"psm "<<end - start<<endl;
	//cout<<esm<<endl;
	start = clock();
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max1;
		unsigned int col = rand()%Max2;
		float v = rand()%Max+1;
		//psm.setEntry(row,col,v);
		esm.coeffRef(row,col) = v;
	}
	end = clock();
	cout<<"esm "<<end - start<<endl;
	
	/*cout<<"测量矩阵转置正确性及时间效率"<<endl;
	start = clock();
	Eigen::SparseMatrix<float> esm4 = esm.transpose();
	end = clock();
	cout<<"esm time consuming:"<<(static_cast<double>(end - start))<<endl;
	start = clock();
	Physika::SparseMatrix<float> psm4 = psm.transpose();
	end = clock();
	cout<<"psm time consuming:"<<(static_cast<double>(end - start))<<endl;
	cout<<"correctness transpose"<<endl;
	compare(psm4,esm4);
	cout<<"psm.transpose"<<endl;
	//cout<<psm4<<endl;*/


	cout<<"multiply effectiveness"<<endl;
	Physika::SparseMatrix<float> psm2(Max1,Max2),psm3(Max2, Max1);
	Eigen::SparseMatrix<float> esm2(Max1, Max2), esm3(Max2, Max1);
	for(unsigned int i=0;i<Max;++i)
	{
		unsigned int row = rand()%Max1;
		unsigned int col = rand()%Max2;
		float v =rand()%Max+1;
		psm2.setEntry(row,col,v);
		esm2.coeffRef(row,col) = v;

		row = rand()%Max2;
		col = rand()%Max1;
		v = rand()%Max +1;
		psm3.setEntry(row,col,v);
		esm3.coeffRef(row,col) = v;
	}
	/*cout<<psm2<<endl;
	cout<<endl;
	cout<<psm3<<endl;*/
	start = clock();
	
	Physika::SparseMatrix<float> psm5 = psm2*psm3;
	end = clock();
	cout<<"physika * consume time:"<<static_cast<double>(end - start)<<endl;
	start = clock();
	Eigen::SparseMatrix<float> esm5 = esm2*esm3;
	end = clock();
	cout<<"eigen * consume time:"<<static_cast<double>(end - start)<<endl;

	cout<<"correctness multiply"<<endl;
	
	//compare(psm5,esm5);



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