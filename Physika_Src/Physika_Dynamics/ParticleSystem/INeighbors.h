#pragma once
#include <vector_types.h>
#include "Platform.h"

namespace Physika
{
	#define NEIGHBOR_SIZE 30

	class NeighborList
	{
	public:
		HYBRID_FUNC NeighborList() { size = 0; }
		HYBRID_FUNC ~NeighborList() {};

		HYBRID_FUNC int& operator[] (int id) { return ids[id]; }
		HYBRID_FUNC int operator[] (int id) const { return ids[id]; }
	public:
		int size;
		int ids[NEIGHBOR_SIZE];
	};


	class RestShape
	{
	public:
		HYBRID_FUNC RestShape() { size = 0; }
		HYBRID_FUNC ~RestShape() {};
	public:
		int size;
		int idx;
		int ids[NEIGHBOR_SIZE];
		float distance[NEIGHBOR_SIZE];
		float3 pos[NEIGHBOR_SIZE];
	};

}

