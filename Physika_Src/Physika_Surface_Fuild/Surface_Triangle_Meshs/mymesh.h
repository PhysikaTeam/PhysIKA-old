#ifndef MYMESH_H
#define MYMESH_H

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Utils/GenProg.hh>
#include <set>
#include <map>
#include <vector_types.h>

class IndexOnVertex;

struct MyTraits: public OpenMesh::DefaultTraits {
	VertexTraits {
		IndexOnVertex *index;
	};
	FaceTraits { };
	EdgeTraits { };
	VertexAttributes(OpenMesh::Attributes::Status);
	FaceAttributes(OpenMesh::Attributes::Status | OpenMesh::Attributes::Normal);
	EdgeAttributes(OpenMesh::Attributes::Status);
	typedef OpenMesh::Vec3f Point;
	typedef OpenMesh::Vec3f Normal;
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> MyMesh;
typedef OpenMesh::VectorT<MyMesh::Point, 3> CoordSystem;
typedef OpenMesh::VectorT<float, 4> Tensor22;

CoordSystem gen_coord_system_by_z_axis(MyMesh::Point const &ez);
float calc_average_edge_length(MyMesh const &mesh);
void insert_vertex_nring(MyMesh const &mesh, MyMesh::VertexHandle vh, int n, std::set<MyMesh::FaceHandle> &set_fh);
void calc_vertex_nring(MyMesh const &mesh, MyMesh::VertexHandle vh, int n, std::map<MyMesh::VertexHandle, int> &map_vh);

#define MAX_VERTEX	10
#define MAX_GRID	250
#define MAX_FACE	18
#define MAX_NEAR_V	80

#define MAX_FACES   125

struct MyVertex {
	float x0, y0;
	float dx, dy;
	int nx, ny;
};

struct VertexOppositeHalfedge {
	bool is_valid;
	bool is_boundary;
	bool opph_is_boundary;
	int from_v;
	int to_v;
	int opph_oppv;
};

#endif
