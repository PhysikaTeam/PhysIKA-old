#pragma once
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"

namespace PhysIKA
{
	template<typename TDataType>
	class ShallowWaterEquationModel : public NumericalModel
	{
		DECLARE_CLASS_1(ShallowWaterEquationModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ShallowWaterEquationModel();
		virtual ~ShallowWaterEquationModel();

		void step(Real dt) override;

		void setDistance(Real distance) { this->distance = distance; }
		void setRelax(Real relax) { this->relax = relax; }
		void setZcount(int zcount) { this->zcount = zcount; }
	public:
		
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_accel;
		
		DeviceArrayField<Coord> solid;
		DeviceArrayField<Coord> normal;
		DeviceArrayField<int>  isBound;
		DeviceArrayField<int>  xindex;
		DeviceArrayField<int>  zindex;
		DeviceArrayField<Real> h;//solid_pos + h*Normal = m_position
		DeviceArrayField<Real> h_buffer;
		
	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;
		Real distance;
		Real relax;
		int zcount;

		//µ÷ÊÔÊ±¼ä
		float sumtimes = 0;
		int sumnum = 0;
	};

#ifdef PRECISION_FLOAT
	template class ShallowWaterEquationModel<DataType3f>;
#else
	template class ShallowWaterEquationModel<DataType3d>;
#endif
}