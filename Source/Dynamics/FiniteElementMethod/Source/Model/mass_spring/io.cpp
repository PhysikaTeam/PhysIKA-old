#include "input_output.h"
using namespace std;

template<typename T>
int io<T>::saveAsVTK(mass_spring_obj<T,3>* my_mass_spring_obj,const string name)
{
  FILE *fp;
  size_t i,j;
  fp=fopen(name.c_str(),"w");
  fprintf(fp,"# vtk DataFile Version 2.0\n");
  fprintf(fp,"tet\n");
  fprintf(fp,"ASCII\n\n");
  fprintf(fp,"DATASET UNSTRUCTURED_GRID\n");
  fprintf(fp,"POINTS %d double\n",my_mass_spring_obj->num_vertex);
  for(i=0;i<my_mass_spring_obj->num_vertex;++i)
  {
    fprintf(fp,"%lf %lf %lf\n",(double)my_mass_spring_obj->myvertexs[i].location.x[0],(double)my_mass_spring_obj->myvertexs[i].location.x[1],(double)my_mass_spring_obj->myvertexs[i].location.x[2]);
  }
  fprintf(fp,"CELLS %d %d\n",my_mass_spring_obj->num_simplexs,my_mass_spring_obj->num_simplexs*(my_mass_spring_obj->dim_simplex+2));
  for(i=0;i<my_mass_spring_obj->num_simplexs;++i)
  {
    fprintf(fp,"%d",my_mass_spring_obj->dim_simplex+1);
    for(j=0;j<my_mass_spring_obj->dim_simplex+1;++j)
    {
      fprintf(fp," %d",my_mass_spring_obj->mysimplexs[i].index_vertex[j]);
    }
    fprintf(fp,"\n");

  }
  fprintf(fp,"CELL_TYPES %d\n",my_mass_spring_obj->num_simplexs);
  for(i=0;i<my_mass_spring_obj->num_simplexs;++i)
  {
    switch(my_mass_spring_obj->dim_simplex)
    {
    case 1:
      fprintf(fp,"3\n");
      break;
    case 2:
      fprintf(fp,"5\n");
      break;
    case 3:
      fprintf(fp,"10\n");
      break;
    }
  }
  
  fclose(fp);
  return 0;
}

template<typename T>
int io<T>::getConstraintFromCsv(mass_spring_obj<T,3>* my_mass_spring_obj,const std::string name)
{
  FILE *fp;
  printf("%s \n",name.c_str());
  fp=fopen(name.c_str(),"r");
  if (!fp)
    return -1;
  char filter[100];
  fscanf(fp,"%s",filter);
  double x[3];
  int fixed_now;
  my_mass_spring_obj->num_fixed=0;
  fgetc(fp);fgetc(fp);

  int eof=feof(fp);
  if(eof)
  {
    printf("eof %d\n",eof);
  }
  else if(!eof)
  {
    printf("eof %d\n",eof);      
    rewind(fp);
    fscanf(fp,"%s",filter);
    fgetc(fp);
    while(!feof(fp))
    {
      fscanf(fp,"%d,%lf,%lf,%lf\n",&fixed_now,&x[0],&x[1],&x[2]);
      my_mass_spring_obj->myvertexs[fixed_now].isFixed=1;
      my_mass_spring_obj->num_fixed++;
    }
  }    
  fclose(fp);
  return 0;
}


template<typename T>
int io<T>::getVertexAndSimplex(std::vector<vertex<T,3> > &myvertexs,std::vector<simplex<T> > &mysimplexs,int &dim_simplex,const std::string name)
{
  printf("%s \n",name.c_str());
  FILE* fp;
  fp=fopen(name.c_str(),"r");
  char filter[100];
  do
  {
    fscanf(fp,"%s",filter);
  }while(!(filter[0]=='P'&&filter[1]=='O'&&filter[2]=='I'&&filter[3]=='N'&&filter[4]=='T'));

  int num_vertex_file;
  int index_vertex_file[4]; size_t index_vertex[4];
  int num_simplexs_file,num_simplexs_dim_2_file;
  int filter_dim_1;
  // 从文件中读取整数用int接受%d ,即使使用时是用size_t ,如果直接用size_t接受%u使用是会出错，此处可能是本身的bug
  double x,y,z;

  fscanf(fp,"%d%s",&num_vertex_file,filter);
  printf("num_vertex_file: %d\n",num_vertex_file);
  size_t i,j;
  for(i=0;i<num_vertex_file;++i)
  {
    fscanf(fp,"%lf%lf%lf",&x,&y,&z);
    // printf("%lf %lf %lf\n",x,y,z);
    myvertexs.push_back(vertex<T,3>(myvector<T,3>(x,y,z)));
  }
  fscanf(fp,"%s%d%d",filter,&num_simplexs_file,&num_simplexs_dim_2_file);

  dim_simplex=int(num_simplexs_dim_2_file/num_simplexs_file)-2;

  T vol;
  int mark=-1;
  for(i=0;i<num_simplexs_file;++i)
  {
    fscanf(fp,"%d",&filter_dim_1);
    for(j=0;j<filter_dim_1;++j)
    {
      fscanf(fp,"%d",&index_vertex_file[j]);
      index_vertex[j]=index_vertex_file[j];
    }

    if(mark==-1)
    {
      myvector<T,3> loc[2];
      loc[0]=myvertexs[index_vertex[0]].location_original;
      loc[1]=myvertexs[index_vertex[1]].location_original;
      T dmetric=(loc[0]-loc[1]).len();
      mark=1;
      switch(dim_simplex)
	    {
	    case 1:
	      vol=dmetric;
	      break;
	    case 2:
	      vol=dmetric*dmetric*0.5;
	      break;
	    case 3:
	      vol=dmetric*dmetric*dmetric / 6.0;
	      break;
	    }
    }
    mysimplexs.push_back(simplex<T>(vol,dim_simplex,index_vertex));
  }
  fclose(fp);
  return 0;
}

template<typename T>
int io<T>::saveTimeNormPair(mass_spring_obj<T,3>* my_mass_spring_obj,const std::string name)
{
  FILE *fp;
  printf("%s \n",name.c_str());
  fp=fopen(name.c_str(),"w");
  // fprintf(fp,"time,norm_Jacobian_cal\n");
  // vector<pair<T,T > >::iterator it;
  for(auto it=my_mass_spring_obj->time_norm_pair.begin();it!=my_mass_spring_obj->time_norm_pair.end();it++)
  {
    fprintf(fp,"%lf %lf\n",(double)it->first,(double)it->second);
  }
  fclose(fp);
  return 0;
}



