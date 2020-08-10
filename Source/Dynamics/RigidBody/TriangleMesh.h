#pragma once

#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleTopology.h"
#include "Dynamics/RigidBody/HostNeighborList.h"
#include <vector>
#include <string>

namespace PhysIKA
{
	
	template<typename TDataType>
	class TriangleMesh
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		typedef typename TopologyModule::PointType PointType;
		typedef typename TopologyModule::Triangle Triangle;

		TriangleMesh();

		HostArray<Triangle>* getTriangles();
		PointType getTriangle(unsigned int i);
		void setTriangles(std::vector<Triangle>& triangles);
		HostNeighborList<int>* getTriangleNeighbors();

		

		void loadObjFile(std::string filename);

	};

#ifdef PRECISION_FLOAT
	template class TriangleMesh<DataType3f>;
#else
	template class TriangleMesh<DataType3d>;
#endif
}