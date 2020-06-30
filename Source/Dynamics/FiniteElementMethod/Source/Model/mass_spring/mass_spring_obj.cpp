#include "myvector.h"
#include "mass_spring_obj.h"
#include "input_output.h"
#include "para.h"
#include "solve_ms.h"


using namespace std;
using namespace Eigen;
using namespace wtyatzoo;

//using namespace __gnc_cxx;
template<typename T,size_t dim>
mass_spring_obj<T,dim >::mass_spring_obj()
{
  myvertexs.clear();
  mysimplexs.clear();
  myedges.clear();
  time_norm_pair.clear();
  
  time_all=norm_Jacobian_cal=0;
  this->mapIndexToLocInMartix=NULL;
  this->mapLocInMatrixToIndex=NULL;
  this->my_matrix_form_cuda_jacobi_solver = NULL;
}


template<typename T,size_t dim>
mass_spring_obj<T,dim>::~mass_spring_obj()
{
  // because the harmonic deformation do not need these two array, they can be null
  if(mapIndexToLocInMartix!=NULL)
  {
    delete[] mapIndexToLocInMartix;
  }
  if(mapLocInMatrixToIndex!=NULL)
  {
    delete[] mapLocInMatrixToIndex;
  }
  if (my_matrix_form_cuda_jacobi_solver != NULL)
    delete my_matrix_form_cuda_jacobi_solver;
}


template<typename T,size_t dim >
mass_spring_obj<T,dim>::mass_spring_obj(std::string input_dir,T dt,T density,int line_search,T weight_line_search,T stiffness,std::string newton_fastMS)
{
  while(!myvertexs.empty())
  {
    myvertexs.pop_back();
  }
  while(!mysimplexs.empty())
  {
    mysimplexs.pop_back();
  }
  while(!myedges.empty())
  {
    myedges.pop_back();
  }
  while(!time_norm_pair.empty())
  {
    time_norm_pair.pop_back();
  }
  time_all=norm_Jacobian_cal=0;

  converge=0;

  pre_succeed=0;
  this->dt=dt;
  this->density=density;
  this->line_search=line_search;
  this->weight_line_search=weight_line_search;
  this->stiffness=stiffness;
  this->newton_fastMS=newton_fastMS;
  
  this->mapIndexToLocInMartix=NULL;
  this->mapLocInMatrixToIndex=NULL;
  this->my_matrix_form_cuda_jacobi_solver = NULL;
  size_t i,j,k;
  size_t x,y,z;

  io<T> myio=io<T>();
  cout << "here" << input_dir << endl;
  myio.getVertexAndSimplex(myvertexs, mysimplexs, dim_simplex, input_dir);
  num_vertex=myvertexs.size();
  printf("num_vertex ::%u \n",num_vertex);
  num_simplexs=mysimplexs.size();
  printf("num_simplexs::%u\n",num_simplexs);
  
  num_all_dof=dim*num_vertex;

  prepare();
  
}

template<typename T,size_t dim >
int mass_spring_obj<T,dim>::prepare()
{
  getEdges();
  init_Energy_now_ForEdge();
  calMassForVertex();
  if(newton_fastMS=="newton")
  {
      
  }
  else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
  {
    max_iteration_num=50; // fast mass spring's max iteration number
    // when we fix the max iteration number to compare the results of orginal fast mass spring method and the accelerated one by Chebyshev semi-iterative method, we let the both method to iterate to the same max_iteration_num times and compare the results' error to the newton method's result which is regarded as the groudtruth.

    if(newton_fastMS=="fastMS_ChebyshevSIM")
    {
      rho=0.8; //need to be learned, here we hardcode it as a  constant for cloth simulation.
      gamma=0.9;
      start_iteration_num=5;
      omega=1.0;
    }
  }

  // trick for no penetration of a sphere
  intensity=para::intensity;
  return 0;
}


template<typename T,size_t dim >
int mass_spring_obj<T,dim >::getEdges()
{
  std::map< std::pair<int,int > ,int> mpFromIndexVertexToIndexEdge;
  mpFromIndexVertexToIndexEdge.clear();

  size_t i;
  size_t index_vertex_for_edge[2];
  size_t index_edge;
  for(i=0;i<num_simplexs;i++)
  {
    if(dim_simplex==1)
    {
      index_vertex_for_edge[0]=mysimplexs[i].index_vertex[0];
      index_vertex_for_edge[1]=mysimplexs[i].index_vertex[1];
      sort(index_vertex_for_edge,index_vertex_for_edge+2);

      if(mpFromIndexVertexToIndexEdge.find(make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1]))==mpFromIndexVertexToIndexEdge.end())
	    {
	      index_edge=myedges.size();
	      mpFromIndexVertexToIndexEdge[make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1])]=index_edge;

	      myvector<T,dim> loc[2];
	      loc[0]=myvertexs[index_vertex_for_edge[0]].location_original;
	      loc[1]=myvertexs[index_vertex_for_edge[1]].location_original;
	      T rest_length=(loc[0]-loc[1]).len();
	      myedges.push_back(edge<T,dim>(rest_length,index_vertex_for_edge));
	    }
    }
    else if(dim_simplex==2)
    {
      // edge 0
      index_vertex_for_edge[0]=mysimplexs[i].index_vertex[0];
      index_vertex_for_edge[1]=mysimplexs[i].index_vertex[1];
      sort(index_vertex_for_edge,index_vertex_for_edge+2);

      if(mpFromIndexVertexToIndexEdge.find(make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1]))==mpFromIndexVertexToIndexEdge.end())
	    {
	      index_edge=myedges.size();
	      mpFromIndexVertexToIndexEdge[make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1])]=index_edge;

	      myvector<T,dim> loc[2];
	      loc[0]=myvertexs[index_vertex_for_edge[0]].location_original;
	      loc[1]=myvertexs[index_vertex_for_edge[1]].location_original;
	      T rest_length=(loc[0]-loc[1]).len();
	      myedges.push_back(edge<T,dim>(rest_length,index_vertex_for_edge));
	    }

      //edge 1
      index_vertex_for_edge[0]=mysimplexs[i].index_vertex[0];
      index_vertex_for_edge[1]=mysimplexs[i].index_vertex[2];
      sort(index_vertex_for_edge,index_vertex_for_edge+2);

      if(mpFromIndexVertexToIndexEdge.find(make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1]))==mpFromIndexVertexToIndexEdge.end())
	    {
	      index_edge=myedges.size();
	      mpFromIndexVertexToIndexEdge[make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1])]=index_edge;

	      myvector<T,dim> loc[2];
	      loc[0]=myvertexs[index_vertex_for_edge[0]].location_original;
	      loc[1]=myvertexs[index_vertex_for_edge[1]].location_original;
	      T rest_length=(loc[0]-loc[1]).len();
	      myedges.push_back(edge<T,dim>(rest_length,index_vertex_for_edge));
	    }

	  
      // here  we use a triangle shape spring system for testing the chebyshev semi-iterative method to accelerate the convergence of the orginal fast mass spring method.
      //do not use the long edge's spring
      //edge 2
	  
      index_vertex_for_edge[0]=mysimplexs[i].index_vertex[1];
      index_vertex_for_edge[1]=mysimplexs[i].index_vertex[2];
      sort(index_vertex_for_edge,index_vertex_for_edge+2);

      if(mpFromIndexVertexToIndexEdge.find(make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1]))==mpFromIndexVertexToIndexEdge.end())
	    {
	      index_edge=myedges.size();
	      mpFromIndexVertexToIndexEdge[make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1])]=index_edge;
	      
	      myvector<T,dim> loc[2];
	      loc[0]=myvertexs[index_vertex_for_edge[0]].location_original;
	      loc[1]=myvertexs[index_vertex_for_edge[1]].location_original;
	      T rest_length=(loc[0]-loc[1]).len();
	      myedges.push_back(edge<T,dim>(rest_length,index_vertex_for_edge));
	    }
    }
    else if(dim_simplex==3)
    {
      vector<array<int, 2>> edge_vert_idx = {{0, 1}, {1, 2}, {2, 0},
                                             {3, 0}, {3, 1}, {3, 2}};
      for (const auto &ei : edge_vert_idx)
      {
        index_vertex_for_edge[0]=mysimplexs[i].index_vertex[ei[0]];
        index_vertex_for_edge[1]=mysimplexs[i].index_vertex[ei[1]];
        sort(index_vertex_for_edge,index_vertex_for_edge+2);
        if(mpFromIndexVertexToIndexEdge.find(make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1]))==mpFromIndexVertexToIndexEdge.end())
        {
          index_edge=myedges.size();
          mpFromIndexVertexToIndexEdge[make_pair(index_vertex_for_edge[0],index_vertex_for_edge[1])]=index_edge;

          myvector<T,dim> loc[2];
          loc[0]=myvertexs[index_vertex_for_edge[0]].location_original;
          loc[1]=myvertexs[index_vertex_for_edge[1]].location_original;
          T rest_length=(loc[0]-loc[1]).len();
          myedges.push_back(edge<T,dim>(rest_length,index_vertex_for_edge));
        }
      }
    }
  }
  num_edges=myedges.size();
  printf("num_edges ::%u \n",num_edges);
  printf("dim_simplex ::%d\n",dim_simplex);
  return 0;
}

template<typename T,size_t dim >
int mass_spring_obj<T,dim>::checkFixedOrFree()
{
  int i,j,k,a,b,c;  
  num_cal_dof=0;
  mapIndexToLocInMartix=new size_t[num_all_dof];
  mapLocInMatrixToIndex=new size_t[num_all_dof];
  for(i=0;i<num_vertex;++i)
  {
    if(myvertexs[i].isFixed==0)
    {
      for(j=0;j<dim;++j)
	    {
	      //   printf("dim %u\n",dim);
	      mapIndexToLocInMartix[i*dim+j]=num_cal_dof;
	      mapLocInMatrixToIndex[num_cal_dof]=i*dim+j;
	      ++num_cal_dof;
	    }
    }
  }
  return 0;
}


template<typename T,size_t dim>
int mass_spring_obj<T,dim>::init_Energy_now_ForEdge()
{
  size_t i;
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<num_edges;++i)
  {
    myedges[i].energy_now=0;
  }
  return 0;
}


template<typename T,size_t dim>
int mass_spring_obj<T,dim>::calMassForVertex()
{
  size_t i,j;
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<num_vertex;++i)
  {
    myvertexs[i].mass=0;
  }

  printf("dim_simplex : %u\n",dim_simplex);
  T help=1.0/(dim_simplex+1);

  for(i=0;i<num_simplexs;++i)
  {
    for(j=0;j<dim_simplex+1;j++)
    {
      myvertexs[mysimplexs[i].index_vertex[j]].mass+=(help*mysimplexs[i].vol*density);
    }
  }
  return 0;
}


template<typename T,size_t dim>
T mass_spring_obj<T,dim>::calElasticEnergy()
{
  size_t i;
  elasticE=0;
  for(i=0;i<num_edges;++i)
  {
    elasticE+=myedges[i].energy_now;
  }
  return elasticE;
}


template<typename T,size_t dim>
int mass_spring_obj<T,dim>::dynamicSimulator()
{
  size_t i;
  //printf("num_fixed: %u\n",num_fixed);
  

  //  MatrixXd Hessian=MatrixXd::Random(num_all_dof,num_all_dof);
  // VectorXd Jacobian(num_all_dof);
  iteration_num=0;
  converge=0;

  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<num_vertex;++i)
  {
    myvertexs[i].location_lastFrame=myvertexs[i].location;
    myvertexs[i].velocity_lastFrame=myvertexs[i].velocity;
  }
  
  clock_t start,finish;
  T totaltime;
  while(!converge)
  { 
    vector<Triplet<double> > tripletsForHessian;
    VectorXd Jacobian(num_all_dof);
    while(!tripletsForHessian.empty())
    {
      tripletsForHessian.pop_back();
    }
    Jacobian.fill(0);
      
    printf("--[Inf]:calJacobianAndHessianForEdge\n");
    start=clock();
    calJacobianAndHessianForEdge(tripletsForHessian,Jacobian);
    finish=clock();
    totaltime=(T)(finish-start)/CLOCKS_PER_SEC;
    time_all+=totaltime; 
    printf("Assemble Time Cost: %lf\n",totaltime);

      
    printf("--[Inf]:solve Matrix\n");
    start=clock();
    //  printf("here converge before solve: %d\n",converge);


    solve(tripletsForHessian, Jacobian, *this);
    finish=clock();
    totaltime=(T)(finish-start)/CLOCKS_PER_SEC;
    time_all+=totaltime;
    printf("Solve Time Cost: %lf\n",totaltime);
    //  printf("here converge after solve: %d\n",converge);
  }
  return 0;
}



template<typename T,size_t dim>
int mass_spring_obj<T,dim>::calJacobianAndHessianForEdge(std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian)
{
  //size_t i,j;

  if(pre_succeed==0)
  {
    for(size_t i=0;i<num_edges;++i)
    {
      myedges[i].calJacobianAndHessian(myvertexs,tripletsForHessian,Jacobian,stiffness,newton_fastMS);
    }
  }
  else if(pre_succeed==1)
  {
    for(size_t i=0;i<num_edges;++i)
    {
      myedges[i].calJacobian(myvertexs,Jacobian,stiffness,newton_fastMS);
    }
  }

  return 0;
  //auto diff
  Jacobian*=-1;
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for
  for(size_t i=0;i<num_vertex;++i)
  {
    for(size_t j=0;j<dim;++j)
    {
      Jacobian(i*dim+j)+=myvertexs[i].force_external(j);
    }
  }
  
  T EPS=1e-5; // local constant to judge zero for dt-100
  T d1dt;
  if(fabs(dt-100)<EPS) 
  {
    printf("--[INF]:: No mass matrix\n");
    d1dt=0;
  }
  else
  {
    d1dt=1.0/dt;
  }
  T mdt,mdtdt;
  myvector<T,dim> vmdt;
  for(size_t i=0;i<num_vertex;++i)
  {
    mdt=myvertexs[i].mass*d1dt;
    mdtdt=mdt*d1dt;
    vmdt=(myvertexs[i].velocity_lastFrame-(myvertexs[i].location-myvertexs[i].location_lastFrame)*d1dt)*mdt;
    for(size_t j=0;j<dim;++j)
    {
      if(pre_succeed==0)
	    {
	      tripletsForHessian.emplace_back(i*dim+j,i*dim+j,mdtdt);
	    }
      Jacobian(i*dim+j)+=vmdt(j);
    }
  }

  return 0;
}
 

template<typename T,size_t dim>
bool mass_spring_obj<T,dim>::checkInversion(Eigen::VectorXd &dx,Eigen::VectorXd &Jacobian)
{
  size_t i;
  size_t x,y;
  Jacobian.fill(0);
  for(i=0;i<num_cal_dof;++i)
  {
    y=mapLocInMatrixToIndex[i]%dim;
    x=mapLocInMatrixToIndex[i]/dim;
    myvertexs[x].location_maybe(y)=dx(i)+myvertexs[x].location(y);
  }
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<num_edges;++i)
  {         
    myedges[i].checkInversion(myvertexs,Jacobian,stiffness,newton_fastMS);
  } 
  return false;
}


template<typename T,size_t dim>
bool mass_spring_obj<T,dim>::checkInversion(Eigen::VectorXd &dx)
{
  size_t i;
  size_t x,y;
  for(i=0;i<num_cal_dof;++i)
  {
    y=mapLocInMatrixToIndex[i]%dim;
    x=mapLocInMatrixToIndex[i]/dim;
    myvertexs[x].location_maybe(y)=dx(i)+myvertexs[x].location(y);
  }
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<num_edges;++i)
  {         
    myedges[i].checkInversion(myvertexs,stiffness,newton_fastMS);
  }  
  return false;
}


template<typename T,size_t dim>
T mass_spring_obj<T,dim>::calEnergyDif()
{
  size_t i;
  T energy_old,energy_new;
  energy_new=energy_old=0;
  for(i=0;i<num_edges;++i)
  {
    energy_new+=myedges[i].energy_maybe;
    energy_old+=myedges[i].energy_now;
  }

  // printf("the first part old energy:%lf new energy:%lf  \n",energy_old,energy_new);
  T EPS=1e-5; // local constant to judge zero for dt-100
  T d1dt;
  if(fabs(dt-100)<EPS) 
  {  
    d1dt=0;
  }
  else
  {
    d1dt=1.0/dt;
  }
  
  myvector<T,dim> help1,help2;
  for(i=0;i<num_vertex;++i)
  {
      
    help1=(myvertexs[i].location_maybe-myvertexs[i].location_lastFrame)*d1dt-myvertexs[i].velocity_lastFrame;
    help2=(myvertexs[i].location-myvertexs[i].location_lastFrame)*d1dt-myvertexs[i].velocity_lastFrame;
      
    energy_new+=help1.len_sq()*myvertexs[i].mass*0.5;
    energy_old+=help2.len_sq()*myvertexs[i].mass*0.5;
      
    energy_new+=-1*myvertexs[i].location_maybe.dot(myvertexs[i].force_external);
    energy_old+=-1*myvertexs[i].location.dot(myvertexs[i].force_external);
  }
  // printf("old energy:%lf new energy:%lf \n",energy_old,energy_new);
  return energy_new-energy_old;
}


template<typename T,size_t dim>
int mass_spring_obj<T, dim>::Val(const T *x, std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const
{
  save_x_regardless_of_const(x);
  T total_energy = 0;
  for (size_t i=0; i<num_edges; ++i)
  {
    T edge_energy = 0;
    myedges[i].calValue(*(const_cast<std::vector<vertex<T,dim> >*>(&myvertexs)), stiffness, newton_fastMS, edge_energy);
    total_energy += edge_energy;
  }

  data->save_val(total_energy);
  return 0;
}

template<typename T,size_t dim>
int mass_spring_obj<T, dim>::Gra(const T *x, std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const
{
  save_x_regardless_of_const(x);

  VectorXd Jacobian = VectorXd::Zero(num_all_dof);
  for (size_t i = 0; i < num_edges; ++i)
  {
    myedges[i].calJacobian(*(const_cast<std::vector<vertex<T,dim> >*>(&myvertexs)),
                           Jacobian, stiffness, newton_fastMS);
  }

  data->save_gra(Jacobian.template cast<T>());
  return 0;
}

template<typename T,size_t dim>
int mass_spring_obj<T, dim>::Hes(const T *x, std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const
{
  save_x_regardless_of_const(x);
  
  vector<Triplet<double>> triplet;
  for (size_t e = 0; e < num_edges; ++e)
  {
    myedges[e].calHessian(*(const_cast<std::vector<vertex<T,dim> >*>(&myvertexs)),
                          triplet, stiffness, newton_fastMS);
  }

  for (auto &t : triplet)
    data->save_hes(t.row(), t.col(), t.value());
  return 0;
}

template<typename T, size_t dim>
int mass_spring_obj<T, dim>::save_x_regardless_of_const(const T* x) const
{
  for (size_t v = 0; v < num_vertex; ++v)
  {
    for (size_t a = 0; a < 3; ++a)
      const_cast<mass_spring_obj<T, dim>*>(this)->myvertexs[v].location.x[a] = x[3*v+a];
  }

  return 0;
}

template<typename T, size_t dim>
size_t mass_spring_obj<T, dim>::Nx() const
{
  return num_all_dof;
}
