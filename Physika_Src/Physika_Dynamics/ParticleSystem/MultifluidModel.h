#pragma once
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Framework/FieldVar.h"
#include "Physika_Framework/Framework/FieldArray.h"
#include "Physika_Framework/Topology/FieldNeighbor.h"

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
	class MultifluidModel : public NumericalModel
	{
		DECLARE_CLASS_1(MultifluidModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
        enum { PhaseCount = 2 };
        using phaseVector = Vector<Real, PhaseCount>;


		MultifluidModel();
		virtual ~MultifluidModel();

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothingLength.setValue(len); }
		void setRestDensity(phaseVector rho) { m_restDensity = rho; }

		void setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver);
		void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
		void setSurfaceTensionSolver(std::shared_ptr<ForceModule> solver);

	public:
		VarField<Real> m_smoothingLength;
		VarField<phaseVector> m_restDensity;
		VarField<Real> m_degenerateMobilityM;
		VarField<Real> m_interfaceDiffusionEpsilon;
		// m_helmholtzFunction;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Vector3f> m_color;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Real> m_massInv; // for pbd constraints
        DeviceArrayField<phaseVector> m_concentration;

		DeviceArrayField<Coord> m_forceDensity;

	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;

		std::shared_ptr<ForceModule> m_surfaceTensionSolver;
		std::shared_ptr<ConstraintModule> m_viscositySolver;
		std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;

		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQuery;
	};

#ifdef PRECISION_FLOAT
	template class MultifluidModel<DataType3f>;
#else
	template class MultifluidModel<DataType3d>;
#endif
}
