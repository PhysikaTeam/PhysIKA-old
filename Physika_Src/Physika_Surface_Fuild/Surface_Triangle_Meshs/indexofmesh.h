#ifndef INDEXOFMESH_H
#define INDEXOFMESH_H

#include "mymesh.h"
#include <vector>
#include <map>

class IndexOfMesh {
public:
	IndexOfMesh(MyMesh &mesh);
	~IndexOfMesh();
	void nearest_intersection(MyMesh::Point base, MyMesh::Point normal, MyMesh::FaceHandle &fh, float &dist);

private:
	IndexOfMesh(IndexOfMesh const &);
	IndexOfMesh &operator=(IndexOfMesh const &);
	void create_index(MyMesh &mesh);
	void release_index();
	int toMapIndex(int i, int j, int k) const;
	void calc_intersection(MyMesh::Point base, MyMesh::Point direct, MyMesh::FaceHandle fh, bool &have_intersection, float &dist);

private:
	MyMesh *m_mesh;
	float avg_edge_len;
	MyMesh::Point base;
	float delta;
	int nx, ny, nz;
	std::map<int, std::vector<MyMesh::FaceHandle> > div;
};

#endif // INDEXOFMESH_H
