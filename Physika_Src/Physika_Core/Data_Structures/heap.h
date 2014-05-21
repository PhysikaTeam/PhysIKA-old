#ifndef PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_
#define PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_

#include<iostream>

namespace Physika{

template <class T>
class MinHeap
{
private:
	T* heapArray_;  //array which store data
	int CurrentSize_;  //num of elements in heap
	int MaxSize_;     //max num of elements the heap can contain now
	void BuildHeap();  //initialize the order of elements in heap
public:
	MinHeap(const int n);   //constructed function; n is the initial max number of elements the heap can contain
	virtual ~MinHeap(){delete[]heapArray_;}
	bool isLeaf(int pos) const;
	int leftChild(int pos) const;
	int rightChild(int pos) const;
	int parent(int pos) const;
	bool remove(int pos, T& node);  //delete elements according to position given
	bool insert(const T&newNode);
	T& removeMin();
	void shiftUp(int pos);
	void shiftDown(int pos);
};

template <class T>
class MaxHeap
{
private:
	T* heapArray_;  //array which store data
	int CurrentSize_;  //num of elements in heap
	int MaxSize_;     //max num of elements the heap can contain now
	void BuildHeap();  //initialize the order of elements in heap
public:
	MaxHeap(const int n);   //constructed function; n is the initial max number of elements the heap can contain
	virtual ~MaxHeap(){delete[]heapArray_;}
	bool isLeaf(int pos) const;
	int leftChild(int pos) const;
	int rightChild(int pos) const;
	int parent(int pos) const;
	bool remove(int pos, T& node);  //delete elements according to position given
	bool insert(const T&newNode);
	T& removeMax();
	void shiftUp(int pos);
	void shiftDown(int pos);
};

template <class T>
MinHeap<T>::MinHeap(const int n)
{
	if(n <= 0)
		return ;
	CurrentSize_ = 0;
	MaxSize_ = n;
	heapArray_ = new T[MaxSize_];
	BuildHeap();
}

template <class T>
bool MinHeap<T>::isLeaf(int pos) const
{
	return (pos >= CurrentSize_/2)&&(pos < CurrentSize);
}

template <class T>
int MinHeap<T>::leftChild(int pos) const
{
	return 2*pos+1;
}

template <class T>
int MinHeap<T>::rightChild(int pos) const
{
	return 2*pos+2;
}

template <class T>
int MinHeap<T>::parent(int pos) const
{
	return (pos-1)/2;
}

template <class T>
void MinHeap<T>::shiftDown(int pos)
{
	int i = pos;
	int j = 2*i+1;
	T temp = heapArray_[i];
	while(j < CurrentSize_ )
	{
		if((j < CurrentSize_ - 1)&&(heapArray_[j] > heapArray_[j+1]))
			j++;
		if(temp > heapArray_[j])
		{
			heapArray_[i] = heapArray_[j];
			i = j;j = 2*j+1;
		}
		else break;
	}
	heapArray_[i] = temp;
}

template <class T>
void MinHeap<T>::BuildHeap()
{
	for(int i = CurrentSize_/2 - 1; i >= 0; --i)
		shiftDown(i);
}

template <class T>
bool MinHeap<T>::insert(const T& newNode)
{
	if(CurrentSize_ == MaxSize_)
		return false;
	heapArray_[CurrentSize_] = newNode;
	shiftUp(CurrentSize_);
	CurrentSize_++;
}

template <class T>
void MinHeap<T>::shiftUp(int pos)
{
	int temppos = pos;
	T temp = heapArray_[temppos];
	while((temppos > 0)&&(heapArray_[parent(temppos)] > temp))
	{
		heapArray_[temppos] = heapArray_[parent(temppos)];
		temppos = parent(temppos);
	}
	heapArray_[temppos] = temp;
}

template <class T>
T & MinHeap<T>::removeMin()
{
	if(CurrentSize_ == 0)
	{
		cout<<"can't delete";exit(1);
	}
	else{
		T temp = heapArray_[0];
		heapArray_[0] = heapArray_[--CurrentSize_];
		if(CurrentSize_ > 1)
			shiftDown(0);
		return temp;
	}
}

template <class T>
bool MinHeap<T>::remove(int pos, T& node)
{
	if((pos < 0)||(pos >= CurrentSize_))
	{
		return false;
	}
	T temp = heapArray_[pos];
	heapArray_[pos] = heapArray_[--CurrentSize_];
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
	CurrentSize_ = 0;
	MaxSize_ = n;
	heapArray_ = new T[MaxSize_];
	BuildHeap();
}

template <class T>
bool MaxHeap<T>::isLeaf(int pos) const
{
	return (pos >= CurrentSize_/2)&&(pos < CurrentSize);
}

template <class T>
int MaxHeap<T>::leftChild(int pos) const
{
	return 2*pos+1;
}

template <class T>
int MaxHeap<T>::rightChild(int pos) const
{
	return 2*pos+2;
}

template <class T>
int MaxHeap<T>::parent(int pos) const
{
	return (pos-1)/2;
}

template <class T>
void MaxHeap<T>::shiftDown(int pos)
{
	int i = pos;
	int j = 2*i+1;
	T temp = heapArray_[i];
	while(j < CurrentSize_ )
	{
		if((j < CurrentSize_ - 1)&&(heapArray_[j] < heapArray_[j+1]))
			j++;
		if(temp < heapArray_[j])
		{
			heapArray_[i] = heapArray_[j];
			i = j;j = 2*j+1;
		}
		else break;
	}
	heapArray_[i] = temp;
}

template <class T>
void MaxHeap<T>::BuildHeap()
{
	for(int i = CurrentSize_/2 - 1; i >= 0; --i)
		shiftDown(i);
}

template <class T>
bool MaxHeap<T>::insert(const T& newNode)
{
	if(CurrentSize_ == MaxSize_)
		return false;
	heapArray_[CurrentSize_] = newNode;
	shiftUp(CurrentSize_);
	CurrentSize_++;
}

template <class T>
void MaxHeap<T>::shiftUp(int pos)
{
	int temppos = pos;
	T temp = heapArray_[temppos];
	while((temppos > 0)&&(heapArray_[parent(temppos)] < temp))
	{
		heapArray_[temppos] = heapArray_[parent(temppos)];
		temppos = parent(temppos);
	}
	heapArray_[temppos] = temp;
}

template <class T>
T & MaxHeap<T>::removeMax()
{
	if(CurrentSize_ == 0)
	{
		cout<<"can't delete";exit(1);
	}
	else{
		T temp = heapArray_[0];
		heapArray_[0] = heapArray_[--CurrentSize_];
		if(CurrentSize_ > 1)
			shiftDown(0);
		return temp;
	}
}

template <class T>
bool MaxHeap<T>::remove(int pos, T& node)
{
	if((pos < 0)||(pos >= CurrentSize_))
	{
		return false;
	}
	T temp = heapArray_[pos];
	heapArray_[pos] = heapArray_[--CurrentSize_];
	shiftUp(pos);
	shiftDown(pos);
	node = temp;
	return true;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_DATA_STRUCTURES_HEAP_H_