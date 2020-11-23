#pragma once
#include <boost/property_tree/ptree.hpp>

#include "Framework/Framework/NumericalIntegrator.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Common/framework.h"
#include "Problem/integrated_problem/embedded_elas_fem_problem.h"
#include "Problem/integrated_problem/embedded_mass_spring_problem.h"
#include "Solver/solver_lists.h"


namespace PhysIKA {
	template<typename TDataType>
	class EmbeddedIntegrator : public NumericalIntegrator
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EmbeddedIntegrator();
		~EmbeddedIntegrator() override {};

		void begin() override;
		void end() override;

		bool integrate() override;

		bool updateVelocity();
		bool updatePosition();


    virtual void bind_problem(const std::shared_ptr<embedded_problem_builder<Real, 3>>& epb_fac, const boost::property_tree::ptree& pt);


	protected:
		bool initializeImpl() override;

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_forceDensity;

	private:
		DeviceArray<Coord> m_prePosition;
		DeviceArray<Coord> m_preVelocity;
    std::vector<Real> pos_;
    std::vector<Real> vel_;          
    HostArray<Coord> m_position_host;
    HostArray<Coord> m_velocity_host;          


    std::shared_ptr<embedded_problem_builder<Real, 3>> epb_fac_;
    std::shared_ptr<newton_base<Real, 3>> solver_;
    std::shared_ptr<dat_str_core<Real, 3>> dat_str_;
    boost::property_tree::ptree pt_;
    std::shared_ptr<semi_implicit<Real>> semi_implicit_;
    std::shared_ptr<fast_ms_info<Real>> fast_ms_solver_info_;
    std::shared_ptr<embedded_interpolate<Real>> embedded_interp_;
    std::string solver_type_;
	};

#ifdef PRECISION_FLOAT
	template class EmbeddedIntegrator<DataType3f>;
#else
 	template class EmbeddedIntegrator<DataType3d>;
#endif
}
