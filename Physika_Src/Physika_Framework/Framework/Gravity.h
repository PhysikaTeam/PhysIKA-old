#pragma once
#include "Physika_Framework/Framework/ModuleForce.h"

namespace Physika{

template<typename TDataType>
class Gravity : public ForceModule
{
	DECLARE_CLASS_1(Gravity, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	Gravity();
	virtual ~Gravity();

	void applyForce() override;

	void setGravity(Coord g) { m_gravity = g; }
private:
	Coord m_gravity;
};

#ifdef PRECISION_FLOAT
template class Gravity<DataType3f>;
#else
template class Gravity<DataType3d>;
#endif

}

