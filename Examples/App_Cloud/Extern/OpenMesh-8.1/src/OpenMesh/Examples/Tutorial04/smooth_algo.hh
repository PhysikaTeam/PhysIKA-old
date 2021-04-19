#include <algorithm>
#include <OpenMesh/Core/Utils/Property.hh>

#ifndef DOXY_IGNORE_THIS

template <class Mesh> class SmootherT
{
public:

  typedef typename Mesh::Point            cog_t;
  typedef OpenMesh::VPropHandleT< cog_t > Property_cog;

public:

  // construct with a given mesh
  explicit SmootherT(Mesh& _mesh) 
    : mesh_(_mesh)
  { 
    mesh_.add_property( cog_ );
  }

  ~SmootherT()
  {
    mesh_.remove_property( cog_ );
  }

  // smooth mesh _iterations times
  void smooth(unsigned int _iterations)
  {
    for (unsigned int i=0; i < _iterations; ++i)
    {
      std::for_each(mesh_.vertices_begin(), 
		    mesh_.vertices_end(), 
		    ComputeCOG(mesh_, cog_));

      std::for_each(mesh_.vertices_begin(), 
		    mesh_.vertices_end(), 
		    SetCOG(mesh_, cog_));
    }
  }


private:


  //--- private classes ---

  class ComputeCOG
  {
  public:
    ComputeCOG(Mesh& _mesh, Property_cog& _cog) 
      : mesh_(_mesh), cog_(_cog)
    {}

    void operator()(typename Mesh::Vertex& _v)
    {
      typename Mesh::VertexHandle      vh( mesh_.handle(_v) );
      typename Mesh::VertexVertexIter  vv_it;
      typename Mesh::Scalar            valence(0.0);
    
      mesh_.property(cog_, vh) = typename Mesh::Point(0.0, 0.0, 0.0);

      for (vv_it=mesh_.vv_iter(vh); vv_it; ++vv_it)
      {
	mesh_.property(cog_, vh) += mesh_.point( vv_it );
	++valence;
      }

      mesh_.property(cog_, mesh_.handle(_v) ) /= valence;
    }

  private:
    Mesh&         mesh_;
    Property_cog& cog_;
  };


  class SetCOG
  {
  public:
    SetCOG(Mesh& _mesh, Property_cog& _cog) 
      : mesh_(_mesh), cog_(_cog)
    {}

    void operator()(typename Mesh::Vertex& _v)
    {
      typename Mesh::VertexHandle vh(mesh_.handle(_v));

      if (!mesh_.is_boundary(vh))
	mesh_.set_point( vh, mesh_.property(cog_, vh) );
    }

  private:

    Mesh&         mesh_;
    Property_cog& cog_;
  };


  //--- private elements ---

  Mesh&        mesh_;
  Property_cog cog_;
};

#endif
