#include "solve_ms.h"
#include "mass_spring_obj.h"
#include <iostream>
#include "para.h"

using namespace std;
using namespace Eigen;
using namespace wtyatzoo;



template<typename T,size_t dim>
int solve(std::vector<Eigen::Triplet<double> > &tripletsForHessian,
          Eigen::VectorXd &Jacobian,
          mass_spring_obj<T, dim> &ms)
{
  size_t i,j;
  size_t ii,jj;
  size_t row=ms.num_cal_dof;
  size_t col=row;
  ms.Jacobian_cal = VectorXd(row);
  VectorXd dx(row),dx_now(row);
  SparseMatrix<double,Eigen::RowMajor > Hessianspa(row,col);
  ms.tripletsForHessianspa.clear();
  T EPS=1e-10; // local constant to judge zero for K(i,j)
  if(ms.newton_fastMS=="newton"||((ms.newton_fastMS=="fastMS_ChebyshevSIM"||ms.newton_fastMS=="fastMS_original")&&ms.pre_succeed==0))
  {
    //  SimplicialLLT<SparseMatrix<double>> linearSolver;

    size_t size_Hessian=tripletsForHessian.size();
      
    for(i=0;i<size_Hessian;++i)
    {
      int row=tripletsForHessian[i].row();
      int col=tripletsForHessian[i].col();
      double val=tripletsForHessian[i].value();
      int map_row=ms.mapIndexToLocInMartix[row];
      int map_col=ms.mapIndexToLocInMartix[col];
      if(map_row>=0&&map_row<ms.num_cal_dof&&map_col>=0&&map_col<ms.num_cal_dof&&fabs(val)>=EPS)
	    {
	      //   printf("row:%d col:%d val:%lf\n",row,col,val);
	      ms.tripletsForHessianspa.emplace_back(map_row,map_col,val);
	    }
    }
  }
  
  omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
  for(i=0;i<row;++i)
  {
    ms.Jacobian_cal(i)=Jacobian(ms.mapLocInMatrixToIndex[i]);
  }

  double norm_Jacobian_cal=0;
  for(i=0;i<row;++i)
  {
    norm_Jacobian_cal+=(ms.Jacobian_cal(i)*ms.Jacobian_cal(i));
  }
  norm_Jacobian_cal=sqrt(norm_Jacobian_cal);
  printf("norm_Jacobian_cal: %lf\n",norm_Jacobian_cal);  
  ms.time_norm_pair.push_back(make_pair(ms.time_all,ms.norm_Jacobian_cal));

  SparseMatrix<double,Eigen::RowMajor > diagMatrix_spa(row,col);
  if(ms.newton_fastMS=="newton"||((ms.newton_fastMS=="fastMS_ChebyshevSIM"||ms.newton_fastMS=="fastMS_original")&&ms.pre_succeed==0))
  {      
    vector< Triplet<double > > tripletsFordiagMatrix_spa;

    for(i=0;i<row;i++)
    {
      tripletsFordiagMatrix_spa.emplace_back(i,i,1e-8);
    }
    diagMatrix_spa.setFromTriplets(tripletsFordiagMatrix_spa.begin(),tripletsFordiagMatrix_spa.end());
    diagMatrix_spa.makeCompressed();
    Hessianspa.setFromTriplets(ms.tripletsForHessianspa.begin(),ms.tripletsForHessianspa.end());
    Hessianspa.makeCompressed();
  }
  
  while(1)
  {
    if(ms.newton_fastMS=="newton")
    {
	  
      ms.linearSolver.compute(Hessianspa);
      int info=(int)ms.linearSolver.info();
      //  cout<<info<<"info"<<endl;
      if(info==0)
	    {
	      break;
	    }
      else if(info==1)
	    {
	      Hessianspa=Hessianspa+diagMatrix_spa;
	      diagMatrix_spa=diagMatrix_spa*2;
	    }
    }
    else if(ms.newton_fastMS=="fastMS_ChebyshevSIM"||ms.newton_fastMS=="fastMS_original")
    {
      //  printf("[info]:: here is fast mass spring method\n");
      if(ms.pre_succeed==0)
	    {
	      // printf("[info]:: the first time to decompose the matrix\n");
	      //	      linearSolver.compute(Hessianspa);
	      ms.my_matrix_form_cuda_jacobi_solver=new matrix_form_cuda_jacobi_solver<double>(Hessianspa.valuePtr(),Hessianspa.innerIndexPtr(),Hessianspa.outerIndexPtr(),Hessianspa.nonZeros(),ms.num_cal_dof);
	      ms.pre_succeed=1;
	      break;
	    }	  
      else if(ms.pre_succeed==1)
	    {
	      //printf("[info]:: no decomposition here!!!\n");
	      break;
	    }
    }
      
  }

  if(ms.newton_fastMS=="newton")
    dx=ms.linearSolver.solve(ms.Jacobian_cal);
  else if(ms.newton_fastMS=="fastMS_ChebyshevSIM"||ms.newton_fastMS=="fastMS_original")
    ms.my_matrix_form_cuda_jacobi_solver->
      apply(ms.Jacobian_cal.data(), dx.data(), 1e-10, 400, 0);

  T h=2;
  size_t max_lineSearch=20;
  T energy_dif,threshold;
  bool find=0;

  VectorXd Jacobian_loc_maybe(ms.num_all_dof);
  VectorXd Jacobian_cal_loc_maybe(row);
  
  if(ms.line_search==1)
  {
    printf("[control info]:: line_search open\n");
    for(i=0;i<max_lineSearch;++i)
    {
      h*=0.5;
      dx_now=dx*h;
      if(ms.checkInversion(dx_now,Jacobian_loc_maybe)==1)
	    {
	      continue;
	    }
      else
	    {
	      energy_dif=ms.calEnergyDif();
	      threshold=ms.weight_line_search*dx_now.dot(ms.Jacobian_cal)*-1;
	      if((energy_dif<=threshold)/*&&(fabs(Jdx)<=fabs(Jdx_now))*/)
        {
          find=1;
          break;
        }
	    }
    }
  }
  else if(ms.line_search==0)
  {
    printf("[control info]:: line_search closed\n");
    h=1;
    dx_now=dx*h;
    if(ms.checkInversion(dx_now)==1)
    {
      for(i=0;i<max_lineSearch;++i)
	    {
	      h*=0.5;
	      dx_now=dx*h;
	      if(ms.checkInversion(dx_now)==1)
        {
          continue;
        }
	      else
        {
          energy_dif=ms.calEnergyDif();
          printf("energy_dif %lf\n",energy_dif);
          threshold=ms.weight_line_search*dx_now.dot(ms.Jacobian_cal)*-1;
          if(energy_dif<=threshold)
          {
            find=1;
            break;
          }
        }
	    }
    }
    find=1;  
    energy_dif=ms.calEnergyDif();
  }  
      
  EPS=1e-4; // local constant to judge zero for energy_dif
  if(find==1)
  {
    if(fabs(norm_Jacobian_cal)<=EPS&&ms.newton_fastMS=="newton")
    {
      printf("bingo\n");
      ms.converge=1;
    }

    omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
    for(i=0;i<ms.num_vertex;++i)
    {
      ms.myvertexs[i].location=ms.myvertexs[i].location_maybe;
    }
    omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
    for(i=0;i<ms.num_edges;++i)
    {
      ms.myedges[i].energy_now=ms.myedges[i].energy_maybe;
    }
  }
  else if(find==0)
  {
    ms.converge=1;
  }

  ++ms.iteration_num;
  /*
    if(ms.iteration_num==max_ms.iteration_num&&(ms.newton_fastMS=="fastMS_ChebyshevSIM"||ms.newton_fastMS=="fastMS_original"))
    {
    ms.converge=1;
    }
  */
 
  if(ms.newton_fastMS=="fastMS_original")
  {
    if(ms.iteration_num==ms.max_iteration_num)
    {
      ms.converge=1;
    }
  }
  else if(ms.newton_fastMS=="fastMS_ChebyshevSIM")
  {
    if(ms.iteration_num==1)
    {
      ;
    }
    else if(ms.iteration_num>=2)
    {
      if(ms.iteration_num<ms.start_iteration_num)
	    {
	      ms.omega=1;
	    }
      else if(ms.iteration_num==ms.start_iteration_num)
	    {
	      ms.omega=2.0/(2.0-ms.rho*ms.rho);
	    }
      else if(ms.iteration_num>ms.start_iteration_num)
	    {
	      ms.omega=4.0/(4.0-ms.rho*ms.rho*ms.omega);
	    }
      if(ms.iteration_num>=3)
	    {
	      for(i=0;i<ms.num_vertex;i++)
        {
          ms.myvertexs[i].location=ms.omega*(ms.gamma*(ms.myvertexs[i].location-ms.myvertexs[i].location_lastIteration)+ms.myvertexs[i].location_lastIteration-ms.myvertexs[i].location_lastlastIteration)+ms.myvertexs[i].location_lastlastIteration;
        }
	    }	  	  
    }
    omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
    for(i=0;i<ms.num_vertex;i++)
    {
      ms.myvertexs[i].location_lastlastIteration=ms.myvertexs[i].location_lastIteration;
    }
    omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
    for(i=0;i<ms.num_vertex;i++)
    {
      ms.myvertexs[i].location_lastIteration=ms.myvertexs[i].location;
    }

    if(ms.iteration_num==ms.max_iteration_num)
    {
      ms.converge=1;
    }
  }
  
  if(ms.converge==1)
  {
    T EPS=1e-5; // local constant to judge zero for dt-100
    T d1dt;
    if(fabs(ms.dt-100)<EPS) 
    {
      d1dt=0;
    }
    else
    {
      d1dt=1.0/ms.dt;
    }
    omp_set_num_threads(para::num_threads);  
#pragma omp parallel for 
    for(i=0;i<ms.num_vertex;++i)
    {      
      ms.myvertexs[i].velocity=(ms.myvertexs[i].location-ms.myvertexs[i].location_lastFrame)*d1dt;
    }
  }

  return 0;
}

template int solve(std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian, mass_spring_obj<double, 3> &ms);
template int solve(std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian, mass_spring_obj<float, 3> &ms);

