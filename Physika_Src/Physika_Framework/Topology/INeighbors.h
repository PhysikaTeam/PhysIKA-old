#pragma once
#include <vector_types.h>
#include "Platform.h"

namespace Physika
{
	#define UNDEFINED -1
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

	template<typename TDataType>
	class TRestShape
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		COMM_FUNC TRestShape() { size = 0; idx = UNDEFINED; }
		COMM_FUNC ~TRestShape() {};
	public:
		int size;
		int idx;
		int ids[NEIGHBOR_SIZE];
		Real distance[NEIGHBOR_SIZE];
		Coord pos[NEIGHBOR_SIZE];
	};

}

