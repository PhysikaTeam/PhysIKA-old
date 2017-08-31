/*
* @file sparse_matrix_test.cpp
* @brief Test the SparseMatrix class. when you want to use the test file to test the SparseMatrix class,
* the best way is to use the cmd to excute the PhysikaTestDebug.exe
* @author Liyou Xu
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
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

#define Max1 5000
#define Max2 2000
#define Max3 3000
#define Max  10000
#define Maxv 10
#define mytype double

using namespace std;
using std::cout;

int main()
{
	Physika::Timer timer;
	srand(time(NULL));
	cout << "****************************   insert in order" << endl;
	Physika::SparseMatrix<mytype> psm(Max1, Max2);
	Eigen::SparseMatrix<mytype> esm(Max1, Max2);
	timer.startTimer();
	for (unsigned int i = 0; i<Max; ++i)
	{
		unsigned int row = rand() % Max1;
		unsigned int col = rand() % Max2;
		mytype v = rand() % Max + 1;
		psm.setEntry(row, col, v);
	}
	timer.stopTimer();
	cout << "psm insert time: " << timer.getElapsedTime() << endl;
	timer.startTimer();
	for (unsigned int i = 0; i<Max; ++i)
	{
		unsigned int row = rand() % Max1;
		unsigned int col = rand() % Max2;
		mytype v = rand() % Max + 1;
		esm.coeffRef(row, col) = v;
	}
	timer.stopTimer();
	cout << "esm insert time: " << timer.getElapsedTime() << endl;
	//transpose efficiency psm1 esm1
	cout << "****************************   transpose" << endl;
	Physika::SparseMatrix<mytype> psm1( Max2, Max1);
	Eigen::SparseMatrix<mytype> esm1(Max2,Max1);

	timer.startTimer();
	psm1 = psm.transpose();
	timer.stopTimer();
	cout << "psm transpose time: " << timer.getElapsedTime() << endl;
	timer.startTimer();
	esm1 = esm.transpose();
	timer.stopTimer();
	cout << "esm transpose time: " << timer.getElapsedTime() << endl;
	//multiply efficiency psm2,esm2*/
	cout << "****************************   multiply" << endl;
	Physika::SparseMatrix<mytype> psm2(Max1,Max2),psm3(Max2, Max3),psm4(Max1,Max3);
	Eigen::SparseMatrix<mytype> esm2(Max1,Max2),esm3(Max2, Max3),esm4(Max1,Max3);
	for (unsigned int i = 0; i<Max; ++i)
	{
		unsigned int row = rand() % Max1;
		unsigned int col = rand() % Max2;
		mytype v = rand() % Max + 1;
		psm2.setEntry(row, col, v);
		esm2.coeffRef(row, col) = v;
	}
	for (unsigned int i = 0; i<Max; ++i)
	{
		unsigned int row = rand() % Max2;
		unsigned int col = rand() % Max3;
		mytype v = rand() % Max + 1;
		psm3.setEntry(row, col, v);
		esm3.coeffRef(row, col) = v;
	}
	timer.startTimer();
	esm4=esm2*esm3;
	timer.stopTimer();
	cout << "esm multiply time: " << timer.getElapsedTime() << endl;
	timer.startTimer();
	psm4=psm2*psm3;
	timer.stopTimer();
	cout << "psm multiply time: " << timer.getElapsedTime() << endl;
	cout << "****************************   *vect" << endl;
	Eigen::SparseVector<mytype> ev1(Max2);
	Physika::VectorND<mytype> pv1(Max2, 0);
	for (unsigned int i = 0; i<Max2; ++i)
	{
		mytype v = rand() % Max + 1;
		pv1[i] = v;
		ev1.coeffRef(i) = v;
	}
	timer.startTimer();
	esm2*ev1;
	esm.operator*(ev1);
	timer.stopTimer();
	cout << "esm multiply vector time: " << timer.getElapsedTime() << endl;
	timer.startTimer();
	psm2*pv1;
	timer.stopTimer();
	cout << "psm multiply vector time: " << timer.getElapsedTime() << endl;
	return 0;
}