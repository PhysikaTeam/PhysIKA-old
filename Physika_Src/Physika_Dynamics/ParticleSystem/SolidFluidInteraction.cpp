#include "SolidFluidInteraction.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utilities/Function1Pt.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(SolidFluidInteraction, TDataType)

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::SolidFluidInteraction()
		: Node()
	{
		setName("default");
		construct();
	}

	template<typename TDataType>
	Physika::SolidFluidInteraction<TDataType>::SolidFluidInteraction(std::string name)
	{
		setName(name);
		construct();
	}

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::~SolidFluidInteraction()
	{
		
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::initialize()
	{
		
		return true;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		return false;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		return false;
	}

	template<typename TDataType>
	void SolidFluidInteraction<TDataType>::construct()
	{

	}
}