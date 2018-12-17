#include "TriangleSet.h"
#include "Physika_Core/Utilities/Function1Pt.h"

namespace Physika
{
	template<typename TDataType>
	TriangleSet<TDataType>::TriangleSet()
	{
	}

	template<typename TDataType>
	TriangleSet<TDataType>::~TriangleSet()
	{
	}

	template<typename TDataType>
	void TriangleSet<TDataType>::updatePointNeighbors()
	{

	}


	template<typename TDataType>
	bool TriangleSet<TDataType>::initializeImpl()
	{
		if (getPoints()->size() <= 0)
		{
			std::vector<Coord> positions;
			std::vector<Triangle> triangles;
			float dx = 0.005f;
			int Nx = 40;
			int Nz = 40;
			
			for (int k = 0; k < Nz; k++) {
				for (int i = 0; i < Nx; i++) {
					positions.push_back(Coord(Real(i*dx+0.4f), Real(0.5), Real(k*dx + 0.4f)));
					if (k < Nz-1 && i < Nx-1)
					{
						Triangle tri1(i + k*Nx, i + 1 + k*Nx, i + 1 + (k + 1)*Nx);
						Triangle tri2(i + k*Nx, i + 1 + (k + 1)*Nx, i + (k + 1)*Nx);
						triangles.push_back(tri1);
						triangles.push_back(tri2);
					}
				}
			}
			this->setPoints(positions);
			this->setTriangles(triangles);
		}
		
		return true;
	}


	template<typename TDataType>
	void TriangleSet<TDataType>::setTriangles(std::vector<Triangle>& triangles)
	{
		m_triangls.resize(triangles.size());
		Function1Pt::copy(m_triangls, triangles);
	}

}