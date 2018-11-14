#pragma once
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "Physika_Core/Platform.h"
#include "Physika_Core/Typedef.h"
#include "Framework/Module.h"

namespace Physika
{
class MechanicalState : public Module
{
public:
	enum MaterialType {
		RIGIDBODY = 0,
		FLUID,
		ELASTIC,
		PLASTIC,
		GRNULAR,
		UNDFINED
	};

public:
	MechanicalState();
	virtual ~MechanicalState(void);

	static std::string position() { return "position"; }
	static std::string pre_position() { return "pre_position"; }
	static std::string d_position() { return "d_position"; }
	static std::string init_position() { return "init_position"; }

	static std::string velocity() { return "velocity"; }
	static std::string angularVelocity() { return "angular_velocity"; }
	static std::string pre_velocity() { return "pre_velocity"; }

	static std::string acceleration() { return "acceleration"; }

	static std::string force() { return "force"; }
	static std::string forceMoment() { return "force_moment"; }
	static std::string d_force() { return "d_force"; }
	
	static std::string mass() { return "mass"; }
	static std::string angularMass() { return "angular_mass"; }
	static std::string rotation() { return "rotation"; }

	std::string getModuleType() override { return "MechanicalState"; }

	MaterialType getMaterialType() { return m_type; }
	void setMaterialType(MaterialType type) { m_type = type; }

	void resetForce();
	void resetField(std::string name);
private:
	MaterialType m_type;
};
}