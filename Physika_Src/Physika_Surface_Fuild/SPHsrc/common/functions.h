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

/// \file Functions.h \brief Declares basic/general functions for the entire application.

#ifndef _Functions_
#define _Functions_

#pragma warning(disable : 4996) //Cancels deprecated.

#include <ctime>
#include <string>
#include <sys/stat.h>
#include <algorithm>
#include "types_def.h"

/// \brief Implements a set of basic/general functions
namespace fun{

std::string GetDateTime();
std::string GetDateTimeAfter(int nseg);

std::string UintStr(unsigned value);
std::string IntStr(int value);
std::string Int3Str(tint3 value);
std::string Uint3Str(tuint3 value);
std::string Uint3RangeStr(const tuint3 &v,const tuint3 &v2);
std::string FloatStr(float value,const char* fmt="%f");
std::string Float3Str(tfloat3 value,const char* fmt="%f,%f,%f");
std::string Float3gStr(tfloat3 value);
std::string StrUpper(const std::string &cad);
std::string StrLower(const std::string &cad);
std::string StrTrim(const std::string &cad);
std::string StrSplit(const std::string mark,std::string &text);

std::string VarStr(const std::string &name,const char *value);
std::string VarStr(const std::string &name,const std::string &value);
std::string VarStr(const std::string &name,float value);
std::string VarStr(const std::string &name,tfloat3 value);
std::string VarStr(const std::string &name,bool value);
std::string VarStr(const std::string &name,int value);
std::string VarStr(const std::string &name,unsigned value);

std::string VarStr(const std::string &name,int n,const int* values,std::string size="?");
std::string VarStr(const std::string &name,int n,const float* values,std::string size="?",const char* fmt="%f");

void PrintVar(const std::string &name,const char *value,const std::string &post="");
void PrintVar(const std::string &name,const std::string &value,const std::string &post="");
void PrintVar(const std::string &name,float value,const std::string &post="");
void PrintVar(const std::string &name,tfloat3 value,const std::string &post="");
void PrintVar(const std::string &name,bool value,const std::string &post="");
void PrintVar(const std::string &name,int value,const std::string &post="");
void PrintVar(const std::string &name,unsigned value,const std::string &post="");

int FileType(const std::string &name);
inline bool FileExists(const std::string &name){ return(FileType(name)==2); }
inline bool DirExists(const std::string &name){ return(FileType(name)==1); }

std::string GetDirParent(const std::string &ruta);
std::string GetFile(const std::string &ruta);
std::string GetDirWithSlash(const std::string &ruta);
std::string GetDirWithoutSlash(const std::string &ruta);
std::string GetExtension(const std::string &file);
std::string GetWithoutExtension(const std::string &ruta);
void GetFileNameSplit(const std::string &file,std::string &dir,std::string &fname,std::string &fext);
std::string AddExtension(const std::string &file,const std::string &ext);
std::string FileNameSec(std::string fname,unsigned fnumber);
std::string ShortFileName(const std::string &file,unsigned maxlen,bool withpoints=true);

bool FileMask(std::string text,std::string mask);

typedef enum{ BigEndian=1,LittleEndian=0 }TpByteOrder;
TpByteOrder GetByteOrder();
void ReverseByteOrder(long long *data,int count,long long *result=NULL);
void ReverseByteOrder(int *data,int count,int *result=NULL);
void ReverseByteOrder(short *data,int count,short *result=NULL);

}

#endif




