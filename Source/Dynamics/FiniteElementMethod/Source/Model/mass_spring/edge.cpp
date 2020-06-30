#include "edge.h"
#include "autodiff.h"
using namespace std;
using namespace Eigen;
//using namespace __gnc_cxx;
DECLARE_DIFFSCALAR_BASE();

template<typename T,size_t dim>
edge<T,dim>::edge(T rest_length,const size_t (&index_vertex)[2])
{
  this->rest_length=rest_length;
  // printf("rest_length ::%lf\n",rest_length);
  size_t i;
  for(i=0;i<2;++i)
  {
    this->index_vertex[i]=index_vertex[i];
  }
}

template<typename T,size_t dim>
bool edge<T,dim>::checkInversion(std::vector< vertex<T,dim > > &myvertexs,Eigen::VectorXd &Jacobian,T stiffness,const std::string &newton_fastMS)
{
  // printf("dim edge: %u\n",dim);
  {
    typedef DScalar1<T,VectorXd > DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

    size_t i,j,ii,jj,row,col;
    size_t ct,ct2;
    VectorXd x(dim*2); 
    for(i=0;i<2;++i)
    {
      for(j=0;j<dim;++j)
      {
        x(i*dim+j)=myvertexs[index_vertex[i]].location_maybe(j);
      }
    }

    DiffScalarBase::setVariableCount(dim*2);
    DScalar x_d[dim*2];
    DScalar energy_d=DScalar(0);
    for(i=0;i<dim*2;i++)
    {
      x_d[i]=DScalar(i,x(i));
    }

    if(newton_fastMS=="newton")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      DScalar len_sq_d(0);
      for(i=0;i<dim;i++)
      {
        len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
      }
      DScalar len_d=sqrt(len_sq_d);
      energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
    }
    else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
      d_aux.normalize(); d_aux*=rest_length;
      DScalar vec_x_sub_daux_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
        energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
      }
      energy_d*=(0.5*stiffness);
	
    }
    this->energy_maybe=energy_d.getValue();
    MatrixXd grad(dim*2,1);
    grad=energy_d.getGradient();
    ct=0; 
    for(i=0;i<2;++i)
    {
      for(row=0;row<dim;++row)
      {
        Jacobian(index_vertex[i]*dim+row)+=grad(ct*dim+row);
      }
      ct++;
    }
  }  
  return false;
}

template<typename T,size_t dim>
bool edge<T,dim>::checkInversion(std::vector< vertex<T,dim > > &myvertexs,T stiffness,const std::string &newton_fastMS)
{
  {
    typedef DScalar2<T,VectorXd, MatrixXd> DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

    size_t i,j,ii,jj,row,col;
    size_t ct,ct2;
    VectorXd x(dim*2); 
    for(i=0;i<2;++i)
    {
      for(j=0;j<dim;++j)
      {
        x(i*dim+j)=myvertexs[index_vertex[i]].location_maybe(j);
      }
    }

    DiffScalarBase::setVariableCount(dim*2);
    DScalar x_d[dim*2];
    DScalar energy_d=DScalar(0);
    for(i=0;i<dim*2;i++)
    {
      x_d[i]=DScalar(i,x(i));
    }

    if(newton_fastMS=="newton")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      DScalar len_sq_d(0);
      for(i=0;i<dim;i++)
      {
        len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
      }
      DScalar len_d=sqrt(len_sq_d);
      energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
    }
    else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
      d_aux.normalize(); d_aux*=rest_length;
      DScalar vec_x_sub_daux_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
        energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
      }
      energy_d*=(0.5*stiffness);
	
    }
    this->energy_maybe=energy_d.getValue();
  }  
  return false; 
}

template<typename T,size_t dim>
int edge<T,dim>::calJacobian(vector< vertex<T,dim> > &myvertexs,Eigen::VectorXd &Jacobian,T stiffness,const std::string &newton_fastMS) const
{ 
  {
    typedef DScalar1<T,VectorXd > DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

    size_t i,j,ii,jj,row,col;
    size_t ct,ct2;
    VectorXd x(dim*2); 
    for(i=0;i<2;++i)
    {
      for(j=0;j<dim;++j)
      {
        x(i*dim+j)=myvertexs[index_vertex[i]].location(j);
      }
    }

    DiffScalarBase::setVariableCount(dim*2);
    DScalar x_d[dim*2];
    DScalar energy_d=DScalar(0);
    for(i=0;i<dim*2;i++)
    {
      x_d[i]=DScalar(i,x(i));
    }

    if(newton_fastMS=="newton")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      DScalar len_sq_d(0);
      for(i=0;i<dim;i++)
      {
        len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
      }
      DScalar len_d=sqrt(len_sq_d);
      energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
    }
    else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
      d_aux.normalize(); d_aux*=rest_length;
      DScalar vec_x_sub_daux_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
        energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
      }
      energy_d*=(0.5*stiffness);
	
    }    
    MatrixXd grad(dim*2,1);
    grad=energy_d.getGradient();
    ct=0; 
    for(i=0;i<2;++i)
    {
      for(row=0;row<dim;++row)
      {
        Jacobian(index_vertex[i]*dim+row)+=grad(ct*dim+row);
      }
      ct++;
    }
  }  
  return 0;
}


template<typename T,size_t dim>
int edge<T,dim>::calJacobianAndHessian(vector< vertex<T,dim> > &myvertexs,std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian,T stiffness,const std::string &newton_fastMS) const
{ 
  {
    typedef DScalar2<T,VectorXd, MatrixXd> DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

    size_t i,j,ii,jj,row,col;
    size_t ct,ct2;
    VectorXd x(dim*2); 
    for(i=0;i<2;++i)
    {
      for(j=0;j<dim;++j)
      {
        x(i*dim+j)=myvertexs[index_vertex[i]].location(j);
      }
    }

    DiffScalarBase::setVariableCount(dim*2);
    DScalar x_d[dim*2];
    DScalar energy_d=DScalar(0);
    for(i=0;i<dim*2;i++)
    {
      x_d[i]=DScalar(i,x(i));
    }

    if(newton_fastMS=="newton")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      DScalar len_sq_d(0);
      for(i=0;i<dim;i++)
      {
        len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
      }
      DScalar len_d=sqrt(len_sq_d);
      energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
    }
    else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
    {
      DScalar vec_x_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_d[i]=x_d[i]-x_d[i+dim];
      }
      myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
      d_aux.normalize(); d_aux*=rest_length;
      DScalar vec_x_sub_daux_d[dim];
      for(i=0;i<dim;i++)
      {
        vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
        energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
      }
      energy_d*=(0.5*stiffness);
	
    }    
    MatrixXd grad(dim*2,1);
    MatrixXd hes(dim*2,dim*2);
    grad=energy_d.getGradient();
    hes=energy_d.getHessian();
    ct=0; 
    for(i=0;i<2;++i)
    {
      for(row=0;row<dim;++row)
      {
        Jacobian(index_vertex[i]*dim+row)+=grad(ct*dim+row);
        ct2=0;
        for(ii=0;ii<2;++ii)
	      {			
          for(col=0;col<dim;col++)
          {
            tripletsForHessian.emplace_back(index_vertex[i]*dim+row,index_vertex[ii]*dim+col,hes(ct*dim+row,ct2*dim+col));
          }
          ct2++;			
	      }
      }
      ct++;
    }
  }  
  return 0;
}

template<typename T,size_t dim>
int edge<T, dim>::calHessian(std::vector< vertex<T,dim > > &myvertexs,std::vector<Eigen::Triplet<double > > &tripletsForHessian, T stiffness,const std::string &newton_fastMS) const
{
  typedef DScalar2<T,VectorXd, MatrixXd> DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

  size_t i,j,ii,jj,row,col;
  size_t ct,ct2;
  VectorXd x(dim*2); 
  for(i=0;i<2;++i)
  {
    for(j=0;j<dim;++j)
    {
      x(i*dim+j)=myvertexs[index_vertex[i]].location(j);
    }
  }

  DiffScalarBase::setVariableCount(dim*2);
  DScalar x_d[dim*2];
  DScalar energy_d=DScalar(0);
  for(i=0;i<dim*2;i++)
  {
    x_d[i]=DScalar(i,x(i));
  }

  if(newton_fastMS=="newton")
  {
    DScalar vec_x_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_d[i]=x_d[i]-x_d[i+dim];
    }
    DScalar len_sq_d(0);
    for(i=0;i<dim;i++)
    {
      len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
    }
    DScalar len_d=sqrt(len_sq_d);
    energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
  }
  else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
  {
    DScalar vec_x_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_d[i]=x_d[i]-x_d[i+dim];
    }
    myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
    d_aux.normalize(); d_aux*=rest_length;
    DScalar vec_x_sub_daux_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
      energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
    }
    energy_d*=(0.5*stiffness);
	
  }    
  MatrixXd grad(dim*2,1);
  MatrixXd hes(dim*2,dim*2);
  grad=energy_d.getGradient();
  hes=energy_d.getHessian();
  ct=0; 
  for(i=0;i<2;++i)
  {
    for(row=0;row<dim;++row)
    {
      ct2=0;
      for(ii=0;ii<2;++ii)
      {			
        for(col=0;col<dim;col++)
        {
          tripletsForHessian.emplace_back(index_vertex[i]*dim+row,index_vertex[ii]*dim+col,hes(ct*dim+row,ct2*dim+col));
        }
        ct2++;			
      }
    }
    ct++;
  }
  return 0;
}


template<typename T, size_t dim>
int edge<T, dim>::calValue(std::vector< vertex<T,dim > > &myvertexs, const T stiffness,const std::string &newton_fastMS, T& val) const
{
  typedef DScalar1<T,VectorXd > DScalar; // use DScalar2 for calculating gradient and hessian and use DScalar1 for calculating gradient

  size_t i,j,ii,jj,row,col;
  size_t ct,ct2;
  VectorXd x(dim*2); 
  for(i=0;i<2;++i)
  {
    for(j=0;j<dim;++j)
    {
      x(i*dim+j)=myvertexs[index_vertex[i]].location_maybe(j);
    }
  }

  DiffScalarBase::setVariableCount(dim*2);
  DScalar x_d[dim*2];
  DScalar energy_d=DScalar(0);
  for(i=0;i<dim*2;i++)
  {
    x_d[i]=DScalar(i,x(i));
  }

  if(newton_fastMS=="newton")
  {
    DScalar vec_x_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_d[i]=x_d[i]-x_d[i+dim];
    }
    DScalar len_sq_d(0);
    for(i=0;i<dim;i++)
    {
      len_sq_d+=(vec_x_d[i]*vec_x_d[i]);
    }
    DScalar len_d=sqrt(len_sq_d);
    energy_d=(len_d-rest_length)*(len_d-rest_length)*stiffness*0.5;
  }
  else if(newton_fastMS=="fastMS_original"||newton_fastMS=="fastMS_ChebyshevSIM")
  {
    DScalar vec_x_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_d[i]=x_d[i]-x_d[i+dim];
    }
    myvector<T,dim> d_aux=myvertexs[index_vertex[0]].location-myvertexs[index_vertex[1]].location;
    d_aux.normalize(); d_aux*=rest_length;
    DScalar vec_x_sub_daux_d[dim];
    for(i=0;i<dim;i++)
    {
      vec_x_sub_daux_d[i]=vec_x_d[i]-d_aux(i);
      energy_d+=(vec_x_sub_daux_d[i]*vec_x_sub_daux_d[i]);
    }
    energy_d*=(0.5*stiffness);
	
  }
  val = energy_d.getValue();

  return 0;
}
