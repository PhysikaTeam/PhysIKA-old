#pragma once
#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Topology/NeighborList.h"
#include "Core/Vector.h"


namespace PhysIKA
{
	template<typename TDataType>
	class PointSet : public TopologyModule
	{
		DECLARE_CLASS_1(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSet();
		~PointSet() override;

		void copyFrom(PointSet<TDataType>& pointSet);

		void setPoints(std::vector<Coord>& pos);
		void setNormals(std::vector<Coord>& normals);
		void setSize(int size);

		DeviceArray<Coord>& getPoints() { return m_coords; }
		DeviceArray<Coord>& getNormals() { return m_normals; }

		int getPointSize() { return m_coords.size(); };

		NeighborList<int>* getPointNeighbors();
		virtual void updatePointNeighbors();

		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

		void loadObjFile(std::string filename);

	protected:
		bool initializeImpl() override;

		Real m_samplingDistance;

		DeviceArray<Coord> m_coords;
		DeviceArray<Coord> m_normals;
		NeighborList<int> m_pointNeighbors;
	};


#ifdef PRECISION_FLOAT
	template class PointSet<DataType3f>;
#else
	template class PointSet<DataType3d>;
#endif
}

