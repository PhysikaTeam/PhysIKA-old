/**
 * @file MultipleFluidModel.h
 * @author Chen Xiaosong
 * @brief An implementation of "Fast Multiple-fluid Simulation Using Helmholtz Free Energy"
 * @version 0.1
 * @date 2019-06-13
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "Framework/Framework/NumericalModel.h"
#include "CahnHilliard.h"

namespace Physika
{	
	template<typename TDataType> class PointSetToPointSet;
	template<typename TDataType> class ParticleIntegrator;
	template<typename TDataType> class NeighborQuery;
	template<typename TDataType> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
	class ForceModule;
	class ConstraintModule;
	/*!
	*	\class	MultiFluidModel
	*   \brief  Multifluid with Mixture Model
	*
	*/
	template<typename TDataType>
	class MultipleFluidModel : public NumericalModel
	{
		DECLARE_CLASS_1(MultipleFluidModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		using PhaseVector = typename CahnHilliard<TDataType>::PhaseVector;


		MultipleFluidModel();
		~MultipleFluidModel() override;

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothingLength.setValue(len); }
		void setRestDensity(PhaseVector rho) { m_restDensity = rho; }
	public:
		VarField<Real> m_smoothingLength;
		VarField<PhaseVector> m_restDensity;
		// m_helmholtzFunction;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Vector3f> m_color;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Real> m_massInv; // for pbd constraints
        DeviceArrayField<PhaseVector> m_concentration;

		DeviceArrayField<Coord> m_forceDensity;

	protected:
		bool initializeImpl() override;

	private:
		std::shared_ptr<ForceModule> m_surfaceTensionSolver;
		std::shared_ptr<ConstraintModule> m_viscositySolver;
		std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

		std::shared_ptr<CahnHilliard<TDataType>> m_phaseSolver;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;

		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQuery;
	};

#ifdef PRECISION_FLOAT
	template class MultipleFluidModel<DataType3f>;
#else
	template class MultipleFluidModel<DataType3d>;
#endif
}
