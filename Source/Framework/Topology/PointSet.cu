#include "PointSet.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Core/Utility.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(PointSet, TDataType)

	template<typename TDataType>
	PointSet<TDataType>::PointSet()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	PointSet<TDataType>::~PointSet()
	{
	}

	template<typename TDataType>
	bool PointSet<TDataType>::initializeImpl()
	{
		return true;
	}


	template<typename TDataType>
	void PointSet<TDataType>::loadObjFile(std::string filename)
	{
		if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj")) {
			std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
			exit(-1);
		}

		std::ifstream infile(filename);
		if (!infile) {
			std::cerr << "Failed to open. Terminating.\n";
			exit(-1);
		}

		int ignored_lines = 0;
		std::string line;
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;
		
		int maxNum = 20;
		std::vector<int>  index;
		std::vector<int>  neighborCount;
		std::vector<int>  elements;
		int count = 0;

		while (!infile.eof()) 
		{
			std::getline(infile, line);

			//.obj files sometimes contain vertex normals indicated by "vn"
			if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn")) 
			{
				std::stringstream data(line);
				char c;
				Coord point;
				data >> c >> point[0] >> point[1] >> point[2];
				vertList.push_back(point);
				index.push_back(count++);
				neighborCount.push_back(0);
			}
			else if (line.substr(0, 2) == std::string("vn")) 
			{
				std::stringstream data(line);
				char c;
				Coord normal;
				data >> c >> normal[0] >> normal[1] >> normal[2];
				normalList.push_back(normal);
			}
			else 
			{
				//vertex read over,init elements capacity
				if (count != 0)
				{
					elements.resize(maxNum*index.size());
					count = 0;
				}
				if(line.substr(0, 1) == std::string("f"))
				{
					//f v1 v2 v3 (v4)
					std::vector<std::string> verStr;
					std::vector<int> verIndex;
					line = line.substr(2, line.size());
					while(line.find_first_of(' ') != std::string::npos)
					{
						verStr.push_back(line.substr(0, line.find_first_of(' ')));
						line = line.substr(line.find_first_of(' '), line.size());
					}
					verStr.push_back(line);
					for (int i = 0; i < verStr.size();++i)
					{
						if (verStr[i].find_first_of('/') == std::string::npos)
							verIndex.push_back(std::stoi(verStr[i]));
						else
							verIndex.push_back(std::stoi(verStr[i].substr(0, verStr[i].find_first_of('/'))));
					}
					//push vertex j into i's neighborList
					for (int i = 0; i < verIndex.size();++i)
					{
						for(int j = 0; j < verIndex.size();++j)
						{
							if (i == j)
								continue;
							bool record = false;
							int aindex = verIndex[i], jndex = verIndex[j];
							for(int t = 0; t < neighborCount[aindex];++t)
								if (elements[aindex*maxNum + t] == jndex)
								{
									record = true;
									break;
								}
							if (!record)
							{
								elements[aindex*maxNum + neighborCount[aindex]] = jndex;
								neighborCount[aindex]++;
								elements[aindex*maxNum + neighborCount[aindex]] = -1;
							}
						}
					}
				}
				else
					++ignored_lines;
			}
		}
		infile.close();

		if (normalList.size() < vertList.size())
		{
			Log::sendMessage(Log::Warning, "The normal size is not equal to the vertex size!");

			int more = vertList.size() - normalList.size();
			for (int i = 0; i < more; i++)
			{
				normalList.push_back(Coord(0));
			}
		}

		assert(normalList.size() == vertList.size());

		std::cout << "Total number of particles: " << vertList.size() << std::endl;

		setPoints(vertList);
		setNormals(normalList);
		setNeighbors(maxNum, elements, index);
		
		vertList.clear();
		normalList.clear();
	}

	template<typename TDataType>
	void PointSet<TDataType>::copyFrom(PointSet<TDataType>& pointSet)
	{
		if (m_coords.size() != pointSet.getPointSize())
		{
			m_coords.resize(pointSet.getPointSize());
			m_normals.resize(pointSet.getPointSize());
		}
		Function1Pt::copy(m_coords, pointSet.getPoints());
		Function1Pt::copy(m_normals, pointSet.getNormals());
	}

	template<typename TDataType>
	void PointSet<TDataType>::setNeighbors(int maxNum, std::vector<int>& elements, std::vector<int>& index)
	{
		m_pointNeighbors.copyFrom(maxNum, elements, index);
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(std::vector<Coord>& pos)
	{
		//printf("%d\n", pos.size());
		m_coords.resize(pos.size());
		Function1Pt::copy(m_coords, pos);

		tagAsChanged();
	}
	template<typename TDataType>
	void PointSet<TDataType>::setSize(int size)
	{
		m_coords.resize(size);
		m_coords.reset();
	}

	template<typename TDataType>
	void PointSet<TDataType>::setNormals(std::vector<Coord>& normals)
	{
		m_normals.resize(normals.size());

		Function1Pt::copy(m_normals, normals);
	}

	template<typename TDataType>
	NeighborList<int>* PointSet<TDataType>::getPointNeighbors()
	{
		if (isTopologyChanged())
		{
			updatePointNeighbors();
		}

		return &m_pointNeighbors;
	}

	template<typename TDataType>
	void PointSet<TDataType>::updatePointNeighbors()
	{
		if (m_coords.isEmpty())
			return;
	}

	template <typename Real, typename Coord>
	__global__ void PS_Scale(
		DeviceArray<Coord> vertex,
		Real s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;
		//return;
		vertex[pId] = vertex[pId] * s;
	}

	template<typename TDataType>
	void PointSet<TDataType>::scale(Real s)
	{
		cuExecute(m_coords.size(), PS_Scale, m_coords, s);
	}

	template <typename Coord>
	__global__ void PS_Scale(
		DeviceArray<Coord> vertex,
		Coord s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		Coord pos_i = vertex[pId];
		vertex[pId] = Coord(pos_i[0] * s[0], pos_i[1] * s[1], pos_i[2] * s[2]);
	}

	template<typename TDataType>
	void PhysIKA::PointSet<TDataType>::scale(Coord s)
	{
		cuExecute(m_coords.size(), PS_Scale, m_coords, s);
	}

	template <typename Coord>
	__global__ void PS_Translate(
		DeviceArray<Coord> vertex,
		Coord t)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		vertex[pId] = vertex[pId] + t;
	}


	template<typename TDataType>
	void PhysIKA::PointSet<TDataType>::translate(Coord t)
	{
		cuExecute(m_coords.size(), PS_Translate, m_coords, t);

// 		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);
// 
// 		PS_Translate << <pDims, BLOCK_SIZE >> > (
// 			m_coords,
// 			t);
// 		cuSynchronize();
	}
}