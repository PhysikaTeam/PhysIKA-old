#pragma once
#include "Core/Vector.h"
#include "Framework/Framework/Module.h"

namespace PhysIKA
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
	~TopologyModule() override;

	virtual int getDOF() { return 0; }

	virtual bool updateTopology() { return true; }

	inline void tagAsChanged() { m_topologyChanged = true; }
	inline void tagAsUnchanged() { m_topologyChanged = false; }
	inline bool isTopologyChanged() { return m_topologyChanged; }

	std::string getModuleType() override { return "TopologyModule"; }
private:
	bool m_topologyChanged;
};
}
