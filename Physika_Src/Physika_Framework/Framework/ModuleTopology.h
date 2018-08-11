#pragma once
#include "Vectors/vector_fixed.h"
#include "Framework/Module.h"

namespace Physika
{

class TopologyModule : public Module
{
	DECLARE_CLASS(TopologyModule)

public:
	typedef unsigned int PointType;
	typedef PointType					Point;
	typedef FixedVector<PointType, 2>	Edge;
	typedef FixedVector<PointType, 3>	Triangle;
	typedef FixedVector<PointType, 4>	Quad;
	typedef FixedVector<PointType, 4>	Tetrahedron;
	typedef FixedVector<PointType, 5>	Pyramid;
	typedef FixedVector<PointType, 6>	Pentahedron;
	typedef FixedVector<PointType, 8>	Hexahedron;
public:
	TopologyModule();
	virtual ~TopologyModule();

	virtual int getDOF() { return 0; }

	virtual bool updateTopology() { return true; }

	inline void tagAsChanged() { m_topologyChanged = true; }
	inline void tagAsUnchanged() { m_topologyChanged = false; }
	inline bool isTopologyChanged() { return m_topologyChanged; }

private:
	bool m_topologyChanged;
};
}
