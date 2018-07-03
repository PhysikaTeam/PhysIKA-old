#pragma once
#include "Platform.h"
namespace Physika 
{
	/*!
	*	\class	Attribute
	*	\brief	particle attribute 0x00000000: [31-30]material; [29]motion; [28]Dynamic; [27-8]undefined yet, for future use; [7-0]correspondding to the id of a fluid phase in multiphase fluid or an object in a multibody system
	*/
	class Attribute
	{
	public:
		HYBRID_FUNC Attribute() { m_tag = 0; }
		HYBRID_FUNC ~Attribute() {};

		enum MaterialType
		{
			MATERIAL_MASK = 0xC0000000,
			MATERIAL_FLUID = 0x00000000,
			MATERIAL_RIGID = 0xA0000000,
			MATERIAL_ELASTIC = 0xB0000000,
			MATERIAL_PLASTIC = 0xC0000000
		};

		enum KinematicType
		{
			KINEMATIC_MASK = 0x30000000,
			KINEMATIC_FIXED = 0x00000000,
			KINEMATIC_PASSIVE = 0x10000000,
			KINEMATIC_POSITIVE = 0x20000000
		};

		enum ObjectID
		{
			OBJECTID_MASK = 0x000000FF
		};

		HYBRID_FUNC inline void SetMaterialType(MaterialType type) { m_tag = ((~MATERIAL_MASK) & m_tag) | type; }
		HYBRID_FUNC inline void SetKinematicType(KinematicType type) { m_tag = ((~KINEMATIC_MASK) & m_tag) | type; }
		HYBRID_FUNC inline void SetObjectId(unsigned id) { m_tag = ((~OBJECTID_MASK) & m_tag) | id; }

		HYBRID_FUNC inline MaterialType GetMaterialType() { return (MaterialType)(m_tag&MATERIAL_MASK); }
		HYBRID_FUNC inline KinematicType GetKinematicType() { return (KinematicType)(m_tag&KINEMATIC_MASK); }
		HYBRID_FUNC inline unsigned GetObjectId() { (unsigned)(m_tag&OBJECTID_MASK); }

		HYBRID_FUNC inline bool IsFluid() { return MaterialType::MATERIAL_FLUID == GetMaterialType(); }
		HYBRID_FUNC inline bool IsRigid() { return MaterialType::MATERIAL_RIGID == GetMaterialType(); }
		HYBRID_FUNC inline bool IsElastic() { return MaterialType::MATERIAL_ELASTIC == GetMaterialType(); }
		HYBRID_FUNC inline bool IsPlastic() { return MaterialType::MATERIAL_PLASTIC == GetMaterialType(); }

		HYBRID_FUNC inline void SetFluid() { SetMaterialType(MaterialType::MATERIAL_FLUID); }
		HYBRID_FUNC inline void SetRigid() { SetMaterialType(MaterialType::MATERIAL_RIGID); }
		HYBRID_FUNC inline void SetElastic() { SetMaterialType(MaterialType::MATERIAL_ELASTIC); }
		HYBRID_FUNC inline void SetPlastic() { SetMaterialType(MaterialType::MATERIAL_PLASTIC); }

		HYBRID_FUNC inline bool IsFixed() { return KinematicType::KINEMATIC_FIXED == GetKinematicType(); }
		HYBRID_FUNC inline bool IsPassive() { return KinematicType::KINEMATIC_PASSIVE == GetKinematicType(); }
		HYBRID_FUNC inline bool IsDynamic() { return KinematicType::KINEMATIC_POSITIVE == GetKinematicType(); }

		HYBRID_FUNC inline void SetFixed() { SetKinematicType(KinematicType::KINEMATIC_FIXED); }
		HYBRID_FUNC inline void SetPassive() { SetKinematicType(KinematicType::KINEMATIC_PASSIVE); }
		HYBRID_FUNC inline void SetDynamic() { SetKinematicType(KinematicType::KINEMATIC_POSITIVE); }

	private:
		unsigned m_tag;
	};
}

