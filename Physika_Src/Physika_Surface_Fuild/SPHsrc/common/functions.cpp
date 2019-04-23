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

/// \file Functions.cpp \brief Implements basic/general functions for the entire application.

#include "functions.h"
#include <cstdio>

namespace fun{

//==============================================================================
/// Returns date and time of the system in text format (dd-mm-yyyy hh:mm:ss)
//==============================================================================
std::string GetDateTime(){
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo=localtime(&rawtime);
  char bufftime[64];
  strftime(bufftime,64,"%d-%m-%Y %H:%M:%S",timeinfo);
  return(bufftime);
}

//==============================================================================
/// Returns date and time of the system + nseg (dd-mm-yy hh:mm)
//==============================================================================
std::string GetDateTimeAfter(int nseg){
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  rawtime+=nseg;
  timeinfo=localtime(&rawtime);
  char bufftime[64];
  strftime(bufftime,64,"%d-%m-%y %H:%M",timeinfo);
  return(bufftime);
}

//==============================================================================
/// Converts unsigned value to string.
//==============================================================================
std::string UintStr(unsigned value){
  char cad[128];
  sprintf(cad,"%u",value);
  return(std::string(cad));
}

//==============================================================================
/// Converts int value to string.
//==============================================================================
std::string IntStr(int value){
  char cad[128];
  sprintf(cad,"%d",value);
  return(std::string(cad));
}

//==============================================================================
/// Converts tint3 value to string.
//==============================================================================
std::string Int3Str(tint3 value){
  char cad[128];
  sprintf(cad,"%d,%d,%d",value.x,value.y,value.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts tuint3 value to string.
//==============================================================================
std::string Uint3Str(tuint3 value){
  char cad[128];
  sprintf(cad,"%u,%u,%u",value.x,value.y,value.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts range of tuint3 values to string.  
//==============================================================================
std::string Uint3RangeStr(const tuint3 &v,const tuint3 &v2){
  char cad[256];
  sprintf(cad,"(%u,%u,%u)-(%u,%u,%u)",v.x,v.y,v.z,v2.x,v2.y,v2.z);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string FloatStr(float value,const char* fmt){
  char cad[128];
  sprintf(cad,fmt,value);
  return(std::string(cad));
}

//==============================================================================
/// Converts real value to string.
//==============================================================================
std::string Float3Str(tfloat3 value,const char* fmt){
  char cad[1024];
  sprintf(cad,fmt,value.x,value.y,value.z);
  return(std::string(cad));
}
//==============================================================================
std::string Float3gStr(tfloat3 value){ return(Float3Str(value,"%g,%g,%g")); }

//==============================================================================
/// Gets string in uppercase.
//==============================================================================
std::string StrUpper(const std::string &cad){
  std::string ret;
  for(unsigned c=0;c<cad.length();c++)ret=ret+char(toupper(cad[c]));
  return(ret);
}

//==============================================================================
/// Gets string in lowercase.
//==============================================================================
std::string StrLower(const std::string &cad){
  std::string ret;
  for(unsigned c=0;c<cad.length();c++)ret=ret+char(tolower(cad[c]));
  return(ret);
}

//==============================================================================
/// Gets string without spaces.
//==============================================================================
std::string StrTrim(const std::string &cad){
  std::string ret;
  int lsp=0,rsp=0;
  for(int c=0;c<int(cad.length())&&cad[c]==' ';c++)lsp++;
  for(int c=int(cad.length())-1;c<int(cad.length())&&cad[c]==' ';c--)rsp++;
  int size=int(cad.length())-(lsp+rsp);
  return(size>0? cad.substr(lsp,size): "");
}

//==============================================================================
/// Returns the text till the indicated mark and save the rest in text format.
//==============================================================================
std::string StrSplit(const std::string mark,std::string &text){
  int tpos=int(text.find(mark));
  std::string ret=(tpos>0? text.substr(0,tpos): text);
  text=(tpos>0? text.substr(tpos+1): "");
  return(ret);
}

//==============================================================================
/// Returns variable and its value in text format.
//==============================================================================
std::string VarStr(const std::string &name,const char *value){ return(name+"=\""+value+"\""); }
std::string VarStr(const std::string &name,const std::string &value){ return(name+"=\""+value+"\""); }
std::string VarStr(const std::string &name,float value){ return(name+"="+FloatStr(value)); }
std::string VarStr(const std::string &name,tfloat3 value){ return(name+"=("+FloatStr(value.x)+","+FloatStr(value.y)+","+FloatStr(value.z)+")"); }
std::string VarStr(const std::string &name,bool value){ return(name+"="+(value? "True": "False")+""); }
std::string VarStr(const std::string &name,int value){
  char cad[30];
  sprintf(cad,"=%d",value);
  return(name+cad);
}
std::string VarStr(const std::string &name,unsigned value){
  char cad[30];
  sprintf(cad,"=%u",value);
  return(name+cad);
}
std::string VarStr(const std::string &name,int n,const int* values,std::string size){
  std::string tex=name+"["+(size=="?"? IntStr(n): size)+"]=[";
  for(int c=0;c<n;c++)tex=tex+(c? ",": "")+fun::IntStr(values[c]);
  return(tex+"]");
}
std::string VarStr(const std::string &name,int n,const float* values,std::string size,const char* fmt){
  std::string tex=name+"["+(size=="?"? IntStr(n): size)+"]=[";
  for(int c=0;c<n;c++)tex=tex+(c? ",": "")+fun::FloatStr(values[c],fmt);
  return(tex+"]");
}

//==============================================================================
/// Prints on the screen a variable with its value.
//==============================================================================
void PrintVar(const std::string &name,const char *value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,const std::string &value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,float value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,tfloat3 value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,bool value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,int value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
void PrintVar(const std::string &name,unsigned value,const std::string &post){ printf("%s%s",VarStr(name,value).c_str(),post.c_str()); }
    

//##############################################################################
//##############################################################################
//##############################################################################
//==============================================================================
/// Returns information about a file, indicates whether file or directory.
/// 0:No exists, 1:Directory, 2:File
//==============================================================================
int FileType(const std::string &name){
  int ret=0;
  struct stat stfileinfo;
  int intstat=stat(name.c_str(),&stfileinfo);
  if(intstat==0){
    if(stfileinfo.st_mode&S_IFDIR)ret=1;
    if(stfileinfo.st_mode&S_IFREG)ret=2;
  }
  return(ret);
}

//==============================================================================
/// Returns the parent directory with the path (ruta).
//==============================================================================
std::string GetDirParent(const std::string &ruta){
  std::string dir;
  int pos=int(ruta.find_last_of("/"));
  if(pos<=0)pos=int(ruta.find_last_of("\\"));
  if(pos>0)dir=ruta.substr(0,pos);
  return(dir);
}

//==============================================================================
/// Returns the filename or directory of a path.
//==============================================================================
std::string GetFile(const std::string &ruta){
  std::string file;
  int c;
  for(c=int(ruta.size())-1;c>=0&&ruta[c]!='\\'&&ruta[c]!='/';c--);
  file=(c<0? ruta: ruta.substr(c+1));
  return(file);
}

//==============================================================================
/// Returns the path with slash.
//==============================================================================
std::string GetDirWithSlash(const std::string &ruta){
  std::string rut=ruta;
  if(ruta!=""){
    char last=ruta[ruta.length()-1];
    if(last!='\\'&&last!='/')rut=ruta+"/";
  }
  return(rut);
}

//==============================================================================
/// Returns the path without slash.
//==============================================================================
std::string GetDirWithoutSlash(const std::string &ruta){
  char last=ruta[ruta.length()-1];
  if(last=='\\'||last=='/')return(ruta.substr(0,ruta.length()-1));
  return(ruta);
}

//==============================================================================
/// Returns the extension of a file.
//==============================================================================
std::string GetExtension(const std::string &file){
  std::string ext;
  int pos=(int)file.find_last_of(".");
  int posmin=std::max((int)file.find_last_of("/"),(int)file.find_last_of("\\"));
  if(pos>=0&&pos>posmin)ext=file.substr(pos+1);
  //printf("[%s].[%s]\n",file.c_str(),ext.c_str());
  return(ext);
}

//==============================================================================
/// Returns the path of a file without the extension (and without the point).
//==============================================================================
std::string GetWithoutExtension(const std::string &ruta){
  int pos=(int)ruta.find_last_of(".");
  int posmin=std::max((int)ruta.find_last_of("/"),(int)ruta.find_last_of("\\"));
  return(pos>=0&&pos>posmin? ruta.substr(0,pos): ruta);
}

//==============================================================================
/// Returns the parent directory, name and extension of a file.
//==============================================================================
void GetFileNameSplit(const std::string &file,std::string &dir,std::string &fname,std::string &fext){
  dir=GetDirParent(file);
  fname=GetFile(file);
  fext=GetExtension(fname);
  if(!fext.empty())fname=fname.substr(0,fname.size()-fext.size()-1);
}

//==============================================================================
/// Adds extension (without point) to the path of a file.
//==============================================================================
std::string AddExtension(const std::string &file,const std::string &ext){
  std::string file2=file;
  if(file2.empty()||file2[file2.length()-1]!='.')file2+='.';
  file2+=ext;
  return(file2);
}

//==============================================================================
/// Returns the filename with number.
//==============================================================================
std::string FileNameSec(std::string fname,unsigned fnumber){
  std::string fext=GetExtension(fname);
  if(!fext.empty())fname=fname.substr(0,fname.size()-fext.size()-1);
  char cad[64];
  sprintf(cad,"_%04d.",fnumber);
  return(fname+cad+fext);
}

//==============================================================================
/// Returns the filename with a requested size of characteres.
//==============================================================================
std::string ShortFileName(const std::string &file,unsigned maxlen,bool withpoints){
  std::string file2;
  if(file.length()<=maxlen)file2=file;
  else{
    file2=file.substr(file.length()-maxlen);
    int pos1=(int)file2.find("\\");
    int pos2=(int)file2.find("/");
    if(pos1<0||(pos2>=0&&pos2<pos1))pos1=pos2;
    if(pos1>=0)file2=file2.substr(pos1);
    if(withpoints){
      if(file2.length()+3>maxlen)file2=ShortFileName(file2,maxlen-3,false);
      file2=std::string("...")+file2;
    }
  }
  return(file2);
}

//==============================================================================
/// Indicates whether there is concordance between filename and mask.
//  Following special characters can be used:
//   - '?': Replaces for any character.
//   - '*': Replaces one or several characters to zero.
//   - '|': Allows to indicate several masks. Example: *.vtk|*.ply
//==============================================================================
bool FileMask(std::string text,std::string mask){
  /*/-Removes '*' consecutives.
  int pos=(int)mask.find("**");
  while(pos>=0){
    mask=mask.substr(0,pos)+mask.substr(pos+1);
    pos=(int)mask.find("**");
  }*/
  //-Checks multiple masks.
  int pos=(int)mask.find("|");
  if(pos>=0)return(FileMask(text,mask.substr(0,pos))||FileMask(text,mask.substr(pos+1)));
  else{
  //-Checks corrleation of text with mask.
    int stext=(int)text.length();
    int smask=(int)mask.length();
    if(!stext&&!smask)return(true);
    else if(smask==1&&mask[0]=='*')return(true);
    else if((smask&&!stext)||(!smask&&stext))return(false);
    else if(mask[0]!='*'){
      if(mask[0]=='?'||mask[0]==text[0])return(FileMask(text.substr(1),mask.substr(1)));
      else return(false);
    }
    else{
      bool res=false;
      for(int c=0;c<stext;c++)res|=FileMask(text.substr(c),mask.substr(1));
      return(res);
    }
  }
} 


//##############################################################################
//##############################################################################
//##############################################################################
//==============================================================================
/// Returns the type of codification using BigEndian or LittleEndian.
//==============================================================================
TpByteOrder GetByteOrder(){
  int i=1;
  return(*((char*)&i)==1? LittleEndian: BigEndian);
}

//==============================================================================
/// Reverses the order of the bytes to exchange BigEndian and LittleEndian.
//==============================================================================
void ReverseByteOrder(long long *data,int count,long long *result){
  for(int c=0;c<count;c++){
    unsigned int v=((unsigned int*)data)[c*2+1];
    unsigned int v2=((unsigned int*)data)[c*2];
    ((unsigned int*)result)[c*2]=((v<<24)&0xFF000000)|((v<<8)&0x00FF0000)|((v>>8)&0x0000FF00)|((v>>24)&0x000000FF);
    ((unsigned int*)result)[c*2+1]=((v2<<24)&0xFF000000)|((v2<<8)&0x00FF0000)|((v2>>8)&0x0000FF00)|((v2>>24)&0x000000FF);
  }
}
//==============================================================================
void ReverseByteOrder(int *data,int count,int *result){
  for(int c=0;c<count;c++){
    unsigned int v=((unsigned int*)data)[c];
    result[c]=((v<<24)&0xFF000000)|((v<<8)&0x00FF0000)|((v>>8)&0x0000FF00)|((v>>24)&0x000000FF);
  }
}
//==============================================================================
void ReverseByteOrder(short *data,int count,short *result){
  for(int c=0;c<count;c++){
    unsigned short v=((unsigned short*)data)[c];
    result[c]=((v<<8)&0xFF00)|((v>>8)&0x00FF);
  }
}

}




