/*
 * @file heap_test.cpp 
 * @brief Test the Maxheap and MinHeap class.
 * @author Mike Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 * contact me at mikepkucs@gmail.com if any question
 */
#include<iostream>
#include "Physika_Core\Data_Structures\heap.h"
using namespace std;

int main()
{
	Physika::MinHeap<int> a(15);
	a.insert(1);
	a.insert(10);
	a.insert(9);
	a.insert(4);
	cout<<a.removeMin()<<endl;    //amazing things happenned   :    if you write this "cout<<a.removeMin()<<' '<<a.removeMin()<<' '<<a.removeMin()<<' '<<a.removeMin()<<' '"
	cout<<a.removeMin()<<endl;    //you will get a inverse result   why?
	cout<<a.removeMin()<<endl;
	cout<<a.removeMin()<<endl;
	Physika::MaxHeap<int> b(15);
	b.insert(1);
	b.insert(10);
	b.insert(9);
	b.insert(4);
	cout<<b.removeMax()<<endl;
	cout<<b.removeMax()<<endl;
	cout<<b.removeMax()<<endl;
	cout<<b.removeMax()<<endl;
	getchar();
	return 0;
}