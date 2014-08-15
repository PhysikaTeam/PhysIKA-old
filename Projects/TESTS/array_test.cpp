/*
 * @file array_test.cpp
 * @brief Test Physika array and iterators.
 * @author FeiZhu
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
#include "Physika_Core/Arrays/array.h"
#include "Physika_Core/Arrays/array_Nd.h"
using Physika::Array;
using Physika::ArrayND;
using namespace std;

int main()
{
    int p[6] = {1,1,1,1,1,1};
    Array<int> array(6, p);
    cout<<"visit array with index: \n";
    for (unsigned int i = 0; i < array.elementCount(); i++)
    {
        cout<<array[i]<<" ";
    }
    cout<<"\nvisit array with iterator:\n";
    for(Array<int>::Iterator iter = array.begin(); iter != array.end(); ++iter)
    {
        cout<<*iter<<" ";
    }
    cout<<"\noutput array with cout<<:\n";
    cout<<array<<endl;
    cout<<"modify the 1st element via iterator:\n";
    Array<int>::Iterator iter = array.begin();
    *iter = 3;
    cout<<array<<endl;
    cout<<"test iterator + operator: array.begin() + 7 == array.end() ?\n";
    iter = iter + 7;
    if(iter==array.end())
        cout<<"Yes\n";
    else
        cout<<"No\n";
    vector<unsigned int> element_counts(2,2);
    ArrayND<int,2> array_nd(element_counts,100);
    cout<<"visit 2x2 array with iterator:\n";
    for(ArrayND<int,2>::Iterator iterator = array_nd.begin(); iterator != array_nd.end(); ++iterator)
    {
        cout<<*iterator<<" ";
    }
    cout<<"test iterator + operator: array.begin() + 7 == array.end() ?\n";
    ArrayND<int,2>::Iterator iterator = array_nd.begin();
    iterator = iterator + 7;
    if(iterator==array_nd.end())
        cout<<"Yes\n";
    else
        cout<<"No\n";
    cout<<"test iterator + operator: array.begin() + 4 == array.end() ?\n";
    iterator = array_nd.begin();
    iterator = iterator + 4;
    if(iterator==array_nd.end())
        cout<<"Yes\n";
    else
        cout<<"No\n";
    // //test * operator of const iterator as left operator
    // const ArrayND<int,2>& array_nd_ref = array_nd;
    // ArrayND<int,2>::ConstIterator const_iter = array_nd_ref.begin();
    // *const_iter = 1;
    getchar();
    return 0;
}
