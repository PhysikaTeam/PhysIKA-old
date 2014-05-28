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


//class MinHeap is a template class. Every elements in any object of this class will be reordered as a min heap.
//but if you use other class as the template T,you must asure you define operation <,>,= between two objects of class T;
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
	
	//shiftUp will compare the pos place node with its father node and change their place if need recursively until find a suitable place for the node you input  
    void shiftUp(int pos);

	//shiftDown will compare the node you input with its sons' node and change their place if need recursively until find a suitable place for the node you input
    void shiftDown(int pos);
private:
	//initialize the order of elements in heap
	//only call once when you create a heap
    void buildHeap(); 
	//array which store data
    T* heap_array_;
	//num of elements in heap
    int current_size_;
	//max num of elements the heap can contain now
    int max_size_;     
};

//class MaxHeap is a template class. Every elements in any object of this class will be reordered as a max heap.
//but if you use other class as the template T,you must asure you define operation <,>,= between two objects of class T;
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
	//shiftDown will compare the node you input with its father's node and change their place if need recursively until find a suitable place for the node you input
    void shiftUp(int pos);
	//shiftDown will compare the node you input with its sons' node and change their place if need recursively until find a suitable place for the node you input
    void shiftDown(int pos);
private:
	//initialize the order of elements in heap
    void buildHeap();
	//array which store data
    T* heap_array_;
	//num of elements in heap
    int current_size_;
	//max num of elements the heap can contain now
    int max_size_;     
};


}  //end of namespace Physika

#include "Physika_Core\Data_Structures\heap-inl.h"

#endif //PHYSIKA_CORE_DATA_STRUCTURES_HEAP_INL_H_