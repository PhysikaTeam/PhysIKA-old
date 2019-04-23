/*
 <DUALSPHYSICS>  Copyright (C) 2012 by Jose M. Dominguez, Dr. Alejandro Crespo, Prof. M. Gomez Gesteira, Anxo Barreiro, Dr. Benedict Rogers.

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file TypesDef.h \brief Declares general types and functions for the entire application.

#ifndef _TypesDef_
#define _TypesDef_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#define PI 3.14159265       ///<Value of cte PI. 
#define TORAD 0.0174532925  ///<Contant for conversion to radians. rad=degrees*TORAD (TORAD=PI/180)
#define TODEG 57.295779513  ///<Contant for conversion to degrees. degrees=rad*TODEG (TODEG=180/PI)

typedef unsigned char byte;
typedef unsigned short word;


///Structure of 3 variables of type int.
typedef struct{
  int x,y,z;
}tint3;

inline tint3 TInt3(int v){ tint3 p={v,v,v}; return(p); }
inline tint3 TInt3(int x,int y,int z){ tint3 p={x,y,z}; return(p); }
inline bool operator ==(const tint3& a, const tint3& b){ return(a.x==b.x&&a.y==b.y&&a.z==b.z); }
inline bool operator !=(const tint3& a, const tint3& b){ return(a.x!=b.x||a.y!=b.y||a.z!=b.z); }
inline tint3 operator +(const tint3& a, const tint3& b){ return(TInt3(a.x+b.x,a.y+b.y,a.z+b.z)); }
inline tint3 operator -(const tint3& a, const tint3& b){ return(TInt3(a.x-b.x,a.y-b.y,a.z-b.z)); }
inline tint3 operator *(const tint3& a, const tint3& b){ return(TInt3(a.x*b.x,a.y*b.y,a.z*b.z)); }
inline tint3 operator /(const tint3& a, const tint3& b){ return(TInt3(a.x/b.x,a.y/b.y,a.z/b.z)); }


///Structure of 3 variables of type unsigned.
typedef struct{
  unsigned x,y,z;
}tuint3;

inline tuint3 TUint3(unsigned v){ tuint3 p={v,v,v}; return(p); }
inline tuint3 TUint3(unsigned x,unsigned y,unsigned z){ tuint3 p={x,y,z}; return(p); }
inline bool operator ==(const tuint3& a, const tuint3& b){ return(a.x==b.x&&a.y==b.y&&a.z==b.z); }
inline bool operator !=(const tuint3& a, const tuint3& b){ return(a.x!=b.x||a.y!=b.y||a.z!=b.z); }
inline tuint3 operator +(const tuint3& a, const tuint3& b){ return(TUint3(a.x+b.x,a.y+b.y,a.z+b.z)); }
inline tuint3 operator -(const tuint3& a, const tuint3& b){ return(TUint3(a.x-b.x,a.y-b.y,a.z-b.z)); }
inline tuint3 operator *(const tuint3& a, const tuint3& b){ return(TUint3(a.x*b.x,a.y*b.y,a.z*b.z)); }
inline tuint3 operator /(const tuint3& a, const tuint3& b){ return(TUint3(a.x/b.x,a.y/b.y,a.z/b.z)); }
inline unsigned TUint3Get(const tuint3& a,unsigned c){ return(!c? a.x: (c==1? a.y: a.z)); }
inline tuint3 TUint3Set(const tuint3& a,unsigned c,unsigned v){ return(TUint3((c? a.x: v),(c!=1? a.y: v),(c!=2? a.z: v))); }


///Structure of 3 variables of type float.
typedef struct{
  float x,y,z;
}tfloat3;

inline tfloat3 TFloat3(float v){ tfloat3 p={v,v,v}; return(p); }
inline tfloat3 TFloat3(float x,float y,float z){ tfloat3 p={x,y,z}; return(p); }
inline bool operator ==(const tfloat3& a, const tfloat3& b){ return(a.x==b.x&&a.y==b.y&&a.z==b.z); }
inline bool operator !=(const tfloat3& a, const tfloat3& b){ return(a.x!=b.x||a.y!=b.y||a.z!=b.z); }
inline tfloat3 operator +(const tfloat3& a, const tfloat3& b){ return(TFloat3(a.x+b.x,a.y+b.y,a.z+b.z)); }
inline tfloat3 operator -(const tfloat3& a, const tfloat3& b){ return(TFloat3(a.x-b.x,a.y-b.y,a.z-b.z)); }
inline tfloat3 operator *(const tfloat3& a, const tfloat3& b){ return(TFloat3(a.x*b.x,a.y*b.y,a.z*b.z)); }
inline tfloat3 operator /(const tfloat3& a, const tfloat3& b){ return(TFloat3(a.x/b.x,a.y/b.y,a.z/b.z)); }
inline float TFloat3Get(const tfloat3& a,unsigned c){ return(!c? a.x: (c==1? a.y: a.z)); }
inline tfloat3 TFloat3Set(const tfloat3& a,unsigned c,float v){ return(TFloat3((c? a.x: v),(c!=1? a.y: v),(c!=2? a.z: v))); }


///Structure of 3 variables of type double.
typedef struct{
  double x,y,z;
}tdouble3;

inline tdouble3 TDouble3(double v){ tdouble3 p={v,v,v}; return(p); }
inline tdouble3 TDouble3(double x,double y,double z){ tdouble3 p={x,y,z}; return(p); }
inline bool operator ==(const tdouble3& a, const tdouble3& b){ return(a.x==b.x&&a.y==b.y&&a.z==b.z); }
inline bool operator !=(const tdouble3& a, const tdouble3& b){ return(a.x!=b.x||a.y!=b.y||a.z!=b.z); }
inline tdouble3 operator +(const tdouble3& a, const tdouble3& b){ return(TDouble3(a.x+b.x,a.y+b.y,a.z+b.z)); }
inline tdouble3 operator -(const tdouble3& a, const tdouble3& b){ return(TDouble3(a.x-b.x,a.y-b.y,a.z-b.z)); }
inline tdouble3 operator *(const tdouble3& a, const tdouble3& b){ return(TDouble3(a.x*b.x,a.y*b.y,a.z*b.z)); }
inline tdouble3 operator /(const tdouble3& a, const tdouble3& b){ return(TDouble3(a.x/b.x,a.y/b.y,a.z/b.z)); }

///Converts \ref tuint3 to \ref tint3.
inline tint3 ToTInt3(const tuint3& v){ return(TInt3(int(v.x),int(v.y),int(v.z))); }
///Converts \ref tint3 to \ref tuint3.
inline tuint3 ToTUint3(const tint3& v){ return(TUint3(unsigned(v.x),unsigned(v.y),unsigned(v.z))); }

///Converts \ref tdouble3 to \ref tfloat3.
inline tfloat3 ToTFloat3(const tdouble3& v){ return(TFloat3(float(v.x),float(v.y),float(v.z))); }
///Converts \ref tfloat3 to \ref tdouble3.
inline tdouble3 ToTDouble3(const tfloat3& v){ return(TDouble3(v.x,v.y,v.z)); }


///Structure of 4 variables of type unsigned.
typedef struct{
  unsigned x,y,z,w;
}tuint4;

inline tuint4 TUint4(unsigned v){ tuint4 p={v,v,v,v}; return(p); }
inline tuint4 TUint4(unsigned x,unsigned y,unsigned z,unsigned w){ tuint4 p={x,y,z,w}; return(p); }
inline bool operator ==(const tuint4& a, const tuint4& b){ return(a.x==b.x&&a.y==b.y&&a.z==b.z&&a.w==b.w); }
inline bool operator !=(const tuint4& a, const tuint4& b){ return(a.x!=b.x||a.y!=b.y||a.z!=b.z||a.w!=b.w); }
inline tuint4 operator +(const tuint4& a, const tuint4& b){ return(TUint4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w)); }
inline tuint4 operator -(const tuint4& a, const tuint4& b){ return(TUint4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w)); }
inline tuint4 operator *(const tuint4& a, const tuint4& b){ return(TUint4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w)); }
inline tuint4 operator /(const tuint4& a, const tuint4& b){ return(TUint4(a.x/b.x,a.y/b.y,a.z/b.z,a.w/b.w)); }


///Matrix of 4x4 values of type float.
typedef struct{
  float a11,a12,a13,a14;
  float a21,a22,a23,a24;
  float a31,a32,a33,a34;
  float a41,a42,a43,a44;
}tmatrix4f;

///Constructor of type \ref matrix4f.
inline tmatrix4f TMatrix4f(float a11,float a12,float a13,float a14,float a21,float a22,float a23,float a24,float a31,float a32,float a33,float a34,float a41,float a42,float a43,float a44){ tmatrix4f m={a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44}; return(m); }
inline tfloat3 MatrixMulPoint(const tmatrix4f &m,const tfloat3 &p){ return(TFloat3(m.a11*p.x + m.a12*p.y + m.a13*p.z + m.a14, m.a21*p.x + m.a22*p.y + m.a23*p.z + m.a24, m.a31*p.x + m.a32*p.y + m.a33*p.z + m.a34)); }


///Matrix of 4x4 values of type double.
typedef struct{
  double a11,a12,a13,a14;
  double a21,a22,a23,a24;
  double a31,a32,a33,a34;
  double a41,a42,a43,a44;
}tmatrix4d;

///Constructor of type \ref matrix4d.
inline tmatrix4d TMatrix4d(double a11,double a12,double a13,double a14,double a21,double a22,double a23,double a24,double a31,double a32,double a33,double a34,double a41,double a42,double a43,double a44){ tmatrix4d m={a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44}; return(m); }
inline tdouble3 MatrixMulPoint(const tmatrix4d &m,const tdouble3 &p){ return(TDouble3(m.a11*p.x + m.a12*p.y + m.a13*p.z + m.a14, m.a21*p.x + m.a22*p.y + m.a23*p.z + m.a24, m.a31*p.x + m.a32*p.y + m.a33*p.z + m.a34)); }


///Symmetric matrix 3x3 of 6 values of type float.
typedef struct{
  float xx,xy,xz,yy,yz,zz;
}tsymatrix3f;

#endif




