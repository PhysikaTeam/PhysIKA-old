#ifndef _MYVECTOR_
#define _MYVECTOR_

#include "head.h"

//myvector for completing the basic vector's computaion task. It will and can be replaced by Physika's vector or cuda vector for final version 
//need to be completed as a template
template <typename T,size_t size > //T only can be float, double 
class myvector
{
 public :
 T x[size];
 myvector()
   {
     static_assert(size>0,"size must be positive integer!");
     for(size_t i=0;i<size;i++)
       {
	 this->x[i]=0;
       }
   }
 myvector(const myvector<T,size >&a)
   {
     static_assert(size>0,"size must be positive integer!");
     for(size_t i=0;i<size;i++)
       {
	 this->x[i]=a.x[i];
       }
   }
 
 myvector(T x0,T x1,T x2)
   {
     static_assert(size==3,"size must be 3 here!");
     this->x[0]=x0; this->x[1]=x1; this->x[2]=x2;
   }
 
 myvector(T x0,T x1)
   {
     static_assert(size==2,"size must be 2 here!");
     this->x[0]=x0; this->x[1]=x1; 
   }

 void set(T x0,T x1,T x2)
 {
   static_assert(size==3,"size must be 3 here!");
   this->x[0]=x0; this->x[1]=x1; this->x[2]=x2;
   return;
 }

 void set(T x0,T x1)
 {
   static_assert(size==2,"size must be 2 here!");
   this->x[0]=x0; this->x[1]=x1; 
   return;
 }
 
 ~myvector() {}

  T& operator()(size_t key)
  {
    //    assert(key<size&&key>=0); // can not be determined at compile time
    if(key<size&&key>=0)
      {
	return this->x[key];
      }
    else
      {
	;
      }
    
  }	
  myvector<T,size >& operator=(const myvector<T,size >& a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]=a.x[i];
	}
      return *this;
    }
  
  myvector<T,size >& operator+=(T a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]+=a;
	}
      return *this;
    }
	
  myvector<T,size >& operator-=(T a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]-=a;
	}
      return *this;
    }
	
  myvector<T,size >& operator*=(T a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]*=a;
	}
      return *this;
    }
	
  myvector<T,size >& operator/=(T a)
    {
      double EPS=1e-6;
      if(fabs(a)>=EPS)
	{
	  for(size_t i=0;i<size;i++)
	    {
	      this->x[i]/=a;
	    }
	}
      return *this;
    }
	
  myvector<T,size >& operator+=(const myvector<T,size >& a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]+=a.x[i];
	}
      return *this;
    }
	
  myvector<T,size >& operator-=(const myvector<T,size >& a)
    {
      for(size_t i=0;i<size;i++)
	{
	  this->x[i]-=a.x[i];
	}
      return *this;
    }
	
  T dot(const myvector<T,size >& a) 
  {
    T dot_product=0.0;
    for(size_t i=0;i<size;i++)
      {
	dot_product+=(this->x[i]*a.x[i]);
      }
    return dot_product;
  }

  myvector<T,size > cross(const myvector<T,size >& a)
  {
    static_assert(size==3,"cross product mush be defined when size==3!");
    return myvector<T,size >(this->x[1]*a.x[2]-this->x[2]*a.x[1],this->x[2]*a.x[0]-this->x[0]*a.x[2],this->x[0]*a.x[1]-this->x[1]*a.x[0]);
  }

  T len_sq(void)
  {
    T len_sq_return=0.0;
    for(size_t i=0;i<size;i++)
      {
	len_sq_return+=(this->x[i]*this->x[i]);
      }
    return len_sq_return;
  }
	
  T len(void)
  {
    return sqrt(len_sq());
  }

  void normalize(void)
  {
    T len_sq_return=len_sq();
    double EPS=1e-6;
    if(fabs(len_sq_return>=EPS)) 
      {
	double len=sqrt(len_sq_return);
	for(size_t i=0;i<size;i++)
	  {
	    this->x[i]/=len;
	  }
      }
    return;
  }
	
};

/*
  //need to duplicate a lot of code from the template above
template <typename T>  
  class myvector<T,3>
{
 public :
  T x[3];
  myvector(T x0,T x1,T x2)
    {
      // static_assert(size==3,"size must be 3 here!");
      this->x[0]=x0; this->x[1]=x1; this->x[2]=x2;
    }
    };*/


template <typename T,size_t size >
  inline myvector<T,size > operator + (const myvector<T,size >& a, const myvector<T,size >& b)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=a.x[i]+b.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator - (const myvector<T,size >& a, const myvector<T,size >& b)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=a.x[i]-b.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator * (T s, const myvector<T,size >& a)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=s*a.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator * (const myvector<T,size >& a,T s)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=s*a.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator + (T s, const myvector<T,size >& a)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=s+a.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator + (const myvector<T,size >& a,T s)
{
  myvector<T,size > v_ret=myvector<T,size >();
  for(size_t i=0;i<size;i++)
    {
      v_ret.x[i]=s+a.x[i];
    }
  return v_ret;
}

template <typename T,size_t size >
  inline myvector<T,size > operator / (const myvector<T,size >& a,T s)
{
  myvector<T,size > v_ret=myvector<T,size >();
  double EPS=1e-6;
  if(fabs(s)>=EPS)
    {
      for(size_t i=0;i<size;i++)
	{
	  v_ret.x[i]=a.x[i]/s;
	}
    }
  return v_ret;
}
#endif
