#include "para.h"
#include "input_output.h"
#include "object_creator.h"
using namespace std;
using namespace Eigen;

template<typename T>
object_creator<T>::object_creator()
{
  out_dir=para::out_dir_object_creator;
  object_name=para::object_name;
  dmetric=para::dmetric;
  printf("dmetric: %lf\n",dmetric);
  lc=para::lc; wc=para::wc; hc=para::hc;
  dim_simplex=para::dim_simplex;

  switch(dim_simplex)
    {
    case 1:
      wc=hc=0;
      break;
    case 2:
      hc=0;
      break;
    case 3:
      break;
    }
  create_object();
}

template<typename T>
object_creator<T>::~object_creator()
{
   if(index_for_vertex!=NULL)
    {
      size_t i,j;
      for(i=0;i<lc;++i)
	{
	  for(j=0;j<wc;++j)
	    {
	      delete[] index_for_vertex[i][j];
	    }
	  delete[] index_for_vertex[i];
	}
      delete[] index_for_vertex;
    }
}

template<typename T>
int object_creator<T>::create_object()
{
  mass_spring_obj<T,3>* my_mass_spring_obj=new mass_spring_obj<T,3>();
  my_mass_spring_obj->dim_simplex=dim_simplex;
  length=dmetric*lc; width=dmetric*wc; height=dmetric*hc;
  switch(dim_simplex)
    {
    case 1:
      lc++; wc+=2; hc+=2;
      break;
    case 2:
      lc++; wc++; hc+=2;
      break;
    case 3:
      break;   
    }
  
  this->index_for_vertex=NULL;
  T length_now,width_now,height_now;
  size_t i,j,k;
  size_t a,b,c;

  // 分配内存空间 保存顶点索引
  index_for_vertex=(int***)new int**[lc];
  for(i=0;i<lc;++i)
    {
      index_for_vertex[i]=(int**)new int*[wc];
      for(j=0;j<wc;++j)
	{
	  index_for_vertex[i][j]=new int[hc];
	}
    }


  for(i=0;i<lc;++i)
    {
      for(j=0;j<wc;++j)
	{
	  for(k=0;k<hc;++k)
	    {
	      index_for_vertex[i][j][k]=-1;
	    }
	}
    }
  
  size_t index_now_vertex=0;
  //set index for all vertexs and EPS is to make double/float`error correction and set it as local constant
  T EPS=1e-6;
  for(i=0,length_now=-length*0.5;length_now<length*0.5+EPS;length_now+=dmetric,i++)
    {
      for(j=0,width_now=-width*0.5;width_now<width*0.5+EPS;width_now+=dmetric,j++)
	{
	  for(k=0,height_now=-height*0.5;height_now<height*0.5+EPS;height_now+=dmetric,k++)
	    {
	      index_for_vertex[i][j][k]=index_now_vertex;
	      my_mass_spring_obj->myvertexs.push_back(vertex<T,3>(myvector<T,3>(length_now,width_now,height_now+0.12)));
	      index_now_vertex++;
	    }
	}
    }
  
  my_mass_spring_obj->num_vertex=my_mass_spring_obj->myvertexs.size();
  
  size_t index_vertex_now[2][2][2];
  //traverse all the grid

  size_t index_vertex_now_simplex[4];
  for(i=0;i<lc-1;++i)
    {   
      for(j=0;j<wc-1;++j)
	{
	  for(k=0;k<hc-1;++k)
	    {
	      for(a=0;a<2;++a)
		{
		  for(b=0;b<2;++b)
		    {
		      for(c=0;c<2;++c)
			{
			  index_vertex_now[a][b][c]=index_for_vertex[i+a][j+b][k+c];
			}
		    }
		}
	      if(dim_simplex==1)
		{
		  T vol=dmetric;
		  index_vertex_now_simplex[0]=index_vertex_now[0][0][0];
		  index_vertex_now_simplex[1]=index_vertex_now[1][0][0];
		  my_mass_spring_obj->mysimplexs.push_back(simplex<T>(vol,dim_simplex,index_vertex_now_simplex));
		}
	      else if(dim_simplex==2)
		{
		  T vol=dmetric*dmetric*0.5;
		  index_vertex_now_simplex[0]=index_vertex_now[0][1][0];
		  index_vertex_now_simplex[1]=index_vertex_now[0][0][0];
		  index_vertex_now_simplex[2]=index_vertex_now[1][1][0];
		  my_mass_spring_obj->mysimplexs.push_back(simplex<T>(vol,dim_simplex,index_vertex_now_simplex));

		  index_vertex_now_simplex[0]=index_vertex_now[1][0][0];
		  index_vertex_now_simplex[1]=index_vertex_now[1][1][0];
		  index_vertex_now_simplex[2]=index_vertex_now[0][0][0];
		  my_mass_spring_obj->mysimplexs.push_back(simplex<T>(vol,dim_simplex,index_vertex_now_simplex));
		}
	      else if(dim_simplex==3)
		{
		  ;
		}
	      
	    }	
	}
    }
  my_mass_spring_obj->num_simplexs=my_mass_spring_obj->mysimplexs.size();

  string path_name=out_dir+"/"+this->object_name+".vtk";
  io<T> myio = io<T>();
  myio.saveAsVTK(my_mass_spring_obj,path_name);
  delete my_mass_spring_obj;
  my_mass_spring_obj=NULL;
  return 0;
}
