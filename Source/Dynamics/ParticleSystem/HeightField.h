#pragma once
#include "ParticleSystem.h"
#include "ShallowWaterEquationModel.h"
namespace PhysIKA
{
	/*!
	*	\class	HeightField
	*	\brief	A height field node
	*/
	template<typename TDataType>
	class HeightField : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(HeightField, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightField(std::string name = "default");
		virtual ~HeightField();

		void loadParticles(Coord lo, Coord hi, Real distance, Real slope, Real relax);

		DeviceArrayField<Coord>* getPosition()
		{
			return &m_position;
		}

		DeviceArrayField<Coord>* getVelocity()
		{
			return &m_velocity;
		}

		void loadObjFile(std::string filename);

		bool initialize() override;
		void advance(Real dt) override;
		void SWEconnect();
	private:
		Real distance;
		Real relax;
		DeviceArrayField<Coord> solid;
		DeviceArrayField<Coord> normal;
		DeviceArrayField<int>  isBound;
		DeviceArrayField<Real> h;//water surface height
		NeighborField<int> neighbors;
		//ShallowWaterEquationModel<TDataType>* swe;
		//DeviceArrayField<Real> b;//bottom height position

	};

#ifdef PRECISION_FLOAT
	template class HeightField<DataType3f>;
#else
	template class HeightField<DataType3d>;
#endif
}