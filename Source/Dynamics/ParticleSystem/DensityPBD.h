#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Kernel.h"

namespace PhysIKA {

	template<typename TDataType> class DensitySummation;

	/*!
	*	\class	DensityPBD
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class DensityPBD : public ConstraintModule
	{
		DECLARE_CLASS_1(DensityPBD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DensityPBD();
		~DensityPBD() override;

		bool constrain() override;

		void takeOneIteration();

		void updateVelocity();

		void setIterationNumber(int n) { m_maxIteration = n; }

		DeviceArray<Real>& getDensity() { return m_density.getValue(); }

	protected:
		bool initializeImpl() override;

	public:
		VarField<Real> m_restDensity;
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Real> m_massInv; // mass^-1 as described in unified particle physics

		NeighborField<int> m_neighborhood;

		DeviceArrayField<Real> m_density;
	private:
		int m_maxIteration;

		SpikyKernel<Real> m_kernel;

		DeviceArray<Real> m_lamda;
		DeviceArray<Coord> m_deltaPos;
		DeviceArray<Coord> m_position_old;

		std::shared_ptr<DensitySummation<TDataType>> m_densitySum;
	};



}