/*
 * @file heap.h 
 * @brief a template heap class
 * @author Mike Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 * If you have any questions about this file, you can email me at mikepkucs@gmail.com
 */

#ifndef PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_
#define PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_

#include<iostream>

namespace Physika{

template <class T>
class MinHeap
{
public:
    MinHeap(const int n);   //constructed function; n is the initial max number of elements the heap can contain
    virtual ~MinHeap(){delete[]heap_array_;}
    bool isLeaf(int pos) const;
    int leftChild(int pos) const;
    int rightChild(int pos) const;
    int parent(int pos) const;
    bool remove(int pos, T& node);  //delete elements according to position given
    bool insert(const T&newNode);
    T& removeMin();
    void shiftUp(int pos);
    void shiftDown(int pos);
private:
    void buildHeap();  //initialize the order of elements in heap
    T* heap_array_;  //array which store data
    int current_size_;  //num of elements in heap
    int max_size_;     //max num of elements the heap can contain now
};

template <class T>
class MaxHeap
{
public:
    MaxHeap(const int n);   //constructed function; n is the initial max number of elements the heap can contain
    virtual ~MaxHeap(){delete[]heap_array_;}
    bool isLeaf(int pos) const;
    int leftChild(int pos) const;
    int rightChild(int pos) const;
    int parent(int pos) const;
    bool remove(int pos, T& node);  //delete elements according to position given
    bool insert(const T&newNode);
    T& removeMax();
    void shiftUp(int pos);
    void shiftDown(int pos);
private:
    void buildHeap();  //initialize the order of elements in heap
    T* heap_array_;  //array which store data
    int current_size_;  //num of elements in heap
    int max_size_;     //max num of elements the heap can contain now
};


}  //end of namespace Physika

#include "Physika_Core/Data_Structures/heap-inl.h"

#endif //PHYSIKA_CORE_DATA_STRUCTURES_HEAP_INL_H_
