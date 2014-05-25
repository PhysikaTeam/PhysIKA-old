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

template <class T>
MinHeap<T>::MinHeap(const int n)
{
	if(n <= 0)
		return ;
	current_size_ = 0;
	max_size_ = n;
	heap_array_ = new T[max_size_];
	buildHeap();
}

template <class T>
bool MinHeap<T>::isLeaf(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function isLeaf()\n";
		std::exit(EXIT_FAILURE);
	}
	return (pos >= current_size_/2)&&(pos < CurrentSize);
}

template <class T>
int MinHeap<T>::leftChild(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function leftChild()\n";
		std::exit(EXIT_FAILURE);
	}
	return 2*pos+1;
}

template <class T>
int MinHeap<T>::rightChild(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function rightChild()\n";
		std::exit(EXIT_FAILURE);
	}
	return 2*pos+2;
}

template <class T>
int MinHeap<T>::parent(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function parent()\n";
		std::exit(EXIT_FAILURE);
	}
	return (pos-1)/2;
}

template <class T>
void MinHeap<T>::shiftDown(int pos)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function shiftDown()\n";
		std::exit(EXIT_FAILURE);
	}
	int i = pos;
	int j = 2*i+1;
	T temp = heap_array_[i];
	while(j < current_size_ )
	{
		if((j < current_size_ - 1)&&(heap_array_[j] > heap_array_[j+1]))
			j++;
		if(temp > heap_array_[j])
		{
			heap_array_[i] = heap_array_[j];
			i = j;j = 2*j+1;
		}
		else break;
	}
	heap_array_[i] = temp;
}

template <class T>
void MinHeap<T>::buildHeap()
{
	for(int i = current_size_/2 - 1; i >= 0; --i)
		shiftDown(i);
}

template <class T>
bool MinHeap<T>::insert(const T& newNode)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function insert()\n";
		std::exit(EXIT_FAILURE);
	}
	if(current_size_ == max_size_)
		return false;
	heap_array_[current_size_] = newNode;
	shiftUp(current_size_);
	current_size_++;
}

template <class T>
void MinHeap<T>::shiftUp(int pos)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function shiftUp()\n";
		std::exit(EXIT_FAILURE);
	}
	int temppos = pos;
	T temp = heap_array_[temppos];
	while((temppos > 0)&&(heap_array_[parent(temppos)] > temp))
	{
		heap_array_[temppos] = heap_array_[parent(temppos)];
		temppos = parent(temppos);
	}
	heap_array_[temppos] = temp;
}

template <class T>
T & MinHeap<T>::removeMin()
{
	if(current_size_ == 0)
	{
		cout<<"can't delete";exit(1);
	}
	else{
		T temp = heap_array_[0];
		heap_array_[0] = heap_array_[--current_size_];
		if(current_size_ > 1)
			shiftDown(0);
		return temp;
	}
}

template <class T>
bool MinHeap<T>::remove(int pos, T& node)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function remove()\n";
		std::exit(EXIT_FAILURE);
	}
	if((pos < 0)||(pos >= current_size_))
	{
		return false;
	}
	T temp = heap_array_[pos];
	heap_array_[pos] = heap_array_[--current_size_];
	shiftUp(pos);
	shiftDown(pos);
	node = temp;
	return true;
}

template <class T>
MaxHeap<T>::MaxHeap(const int n)
{
	if(n <= 0)
		return ;
	current_size_ = 0;
	max_size_ = n;
	heap_array_ = new T[max_size_];
	buildHeap();
}

template <class T>
bool MaxHeap<T>::isLeaf(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function isLeaf()\n";
		std::exit(EXIT_FAILURE);
	}
	return (pos >= current_size_/2)&&(pos < CurrentSize);
}

template <class T>
int MaxHeap<T>::leftChild(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function leftChild()\n";
		std::exit(EXIT_FAILURE);
	}
	return 2*pos+1;
}

template <class T>
int MaxHeap<T>::rightChild(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function rightChild()\n";
		std::exit(EXIT_FAILURE);
	}
	return 2*pos+2;
}

template <class T>
int MaxHeap<T>::parent(int pos) const
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function parent()\n";
		std::exit(EXIT_FAILURE);
	}
	return (pos-1)/2;
}

template <class T>
void MaxHeap<T>::shiftDown(int pos)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function shiftDown()\n";
		std::exit(EXIT_FAILURE);
	}
	int i = pos;
	int j = 2*i+1;
	T temp = heap_array_[i];
	while(j < current_size_ )
	{
		if((j < current_size_ - 1)&&(heap_array_[j] < heap_array_[j+1]))
			j++;
		if(temp < heap_array_[j])
		{
			heap_array_[i] = heap_array_[j];
			i = j;j = 2*j+1;
		}
		else break;
	}
	heap_array_[i] = temp;
}

template <class T>
void MaxHeap<T>::buildHeap()
{
	for(int i = current_size_/2 - 1; i >= 0; --i)
		shiftDown(i);
}

template <class T>
bool MaxHeap<T>::insert(const T& newNode)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function insert()\n";
		std::exit(EXIT_FAILURE);
	}
	if(current_size_ == max_size_)
		return false;
	heap_array_[current_size_] = newNode;
	shiftUp(current_size_);
	current_size_++;
}

template <class T>
void MaxHeap<T>::shiftUp(int pos)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function shiftUp()\n";
		std::exit(EXIT_FAILURE);
	}
	int temppos = pos;
	T temp = heap_array_[temppos];
	while((temppos > 0)&&(heap_array_[parent(temppos)] < temp))
	{
		heap_array_[temppos] = heap_array_[parent(temppos)];
		temppos = parent(temppos);
	}
	heap_array_[temppos] = temp;
}

template <class T>
T & MaxHeap<T>::removeMax()
{
	if(current_size_ == 0)
	{
		cout<<"can't delete";exit(1);
	}
	else{
		T temp = heap_array_[0];
		heap_array_[0] = heap_array_[--current_size_];
		if(current_size_ > 1)
			shiftDown(0);
		return temp;
	}
}

template <class T>
bool MaxHeap<T>::remove(int pos, T& node)
{
	if(pos >= current_size_||pos < 0)
	{
		std::cerr<<"invalid position when call function remove()\n";
		std::exit(EXIT_FAILURE);
	}
	if((pos < 0)||(pos >= current_size_))
	{
		return false;
	}
	T temp = heap_array_[pos];
	heap_array_[pos] = heap_array_[--current_size_];
	shiftUp(pos);
	shiftDown(pos);
	node = temp;
	return true;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_