#pragma once
#include <stdio.h>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <vector>

struct MyTraits : public OpenMesh::DefaultTraits
{
	VertexAttributes(OpenMesh::Attributes::Status);
	FaceAttributes(OpenMesh::Attributes::Status);
	EdgeAttributes(OpenMesh::Attributes::Status);
};
typedef OpenMesh::PolyMesh_ArrayKernelT<MyTraits>  openMesh;


#define  F_ZERO    0.0000001
#define  MAXVAL    999999999
#define  PARTICLE_RES   80
