#ifndef _TRIPLET_
#define _TRIPLET_

#include "head.h"

template<typename T> 
class triplet
{
 public:
  size_t row,col;
  T val;
  triplet(size_t row,size_t col,T val)
    {
      this->row=row; this->col=col; this->val=val;
    }
  triplet(const triplet<T>&a)
    {
      this->row=a.row; this->col=a.col; this->val=a.val;
    }
  triplet(){}
  ~triplet(){}
};

#endif
