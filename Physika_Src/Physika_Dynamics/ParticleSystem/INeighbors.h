#pragma once
#include <vector_types.h>
#include "Platform.h"

namespace Physika
{
	#define NEIGHBOR_SIZE 30

	class SPHNeighborList
	{
	public:
		COMM_FUNC SPHNeighborList() { size = 0; }
		COMM_FUNC ~SPHNeighborList() {};

		COMM_FUNC int& operator[] (int id) { return ids[id]; }
		COMM_FUNC int operator[] (int id) const { return ids[id]; }
	public:
		int size;
		int ids[NEIGHBOR_SIZE];
	};


	class RestShape
	{
	public:
		COMM_FUNC RestShape() { size = 0; }
		COMM_FUNC ~RestShape() {};
	public:
		int size;
		int idx;
		int ids[NEIGHBOR_SIZE];
		float distance[NEIGHBOR_SIZE];
		float3 pos[NEIGHBOR_SIZE];
	};

}

