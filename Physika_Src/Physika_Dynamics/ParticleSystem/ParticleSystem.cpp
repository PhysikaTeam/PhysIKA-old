#include "ParticleSystem.h"
#include "DensityPBD.h"
#include "ParticlePrediction.h"
#include "ViscosityBase.h"
#include "NeighborQuery.h"
#include "SummationDensity.h"
#include "SurfaceTension.h"
#include "DensitySimple.h"
#include "BoundaryManager.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Physika_Core/Utilities/Reduction.h"


namespace Physika
{
	template<typename TDataType>
	ParticleSystem<TDataType>::ParticleSystem(String name)
		: Node(name)
	{
		m_num = this->allocHostVariable<size_t>("num", "Particle number");
		m_mass = this->allocHostVariable<Real>("mass", "Particle mass", Real(1));
		m_smoothingLength = this->allocHostVariable<Real>("smoothingLength", "Smoothing length", Real(0.0125));
		m_samplingDistance = this->allocHostVariable<Real>("samplingDistance", "Sampling distance", Real(0.005));
		m_restDensity = this->allocHostVariable<Real>("restDensity", "Rest density", Real(1000));
		m_viscosity = this->allocHostVariable<Real>("viscosity", "Viscosity", Real(0.05));

		m_lowerBound = this->allocHostVariable<Coord>("lowerBound", "Lower bound", make_float3(0));
		m_upperBound = this->allocHostVariable<Coord>("upperBound", "Upper bound", make_float3(1));

		m_gravity = this->allocHostVariable<Coord>("gravity", "gravity", make_float3(0.0f, -9.8f, 0.0f));
	}

	template<typename TDataType>
	bool ParticleSystem<TDataType>::initialize()
	{
		setAsCurrentContext();

// 		for (int m = 0; m < models.size(); m++)
// 		{
// 			int num = models[m]->positions.size();
// 			for (int i = 0; i < num; i++)
// 			{
// 				poss.push_back(models[m]->positions[i]);
// 				vels.push_back(models[m]->velocities[i]);
// 				atts.push_back(models[m]->attributes[i]);
// 			}
// 		}


		std::vector<Coord> positions;
		std::vector<Coord> velocities;
		std::vector<Attribute> attributes;
		for (float x = 0.4; x < 0.6; x += 0.005f) {
			for (float y = 0.1; y < 0.2; y += 0.005f) {
				for (float z = 0.4; z < 0.6; z += 0.005f) {
					float3 pos = make_float3(float(x), float(y), float(z));
					positions.push_back(pos);
					velocities.push_back(make_float3(0));
					Attribute attri;
					attri.SetFluid();
					attri.SetDynamic();
					attributes.push_back(attri);
				}
			}
		}


		SetParticleNumber(positions.size());

		std::shared_ptr<DeviceContext> dc = getContext();

		dc->allocDeviceBuffer<Coord>("POSITION", "Particle positions", m_num->getValue());
		dc->allocDeviceBuffer<Coord>("VELOCITY", "Particle velocities", m_num->getValue());
		dc->allocDeviceBuffer<Coord>("OLD_POSITION", "Old particle positions", m_num->getValue());
		dc->allocDeviceBuffer<Coord>("OLD_VELOCITY", "Particle positions", m_num->getValue());
		dc->allocDeviceBuffer<Real>("DENSITY", "Particle densities", m_num->getValue());
		dc->allocDeviceBuffer<NeighborList>("NEIGHBORHOOD", "Particle neighbor ids", m_num->getValue());
		dc->allocDeviceBuffer<Attribute>("ATTRIBUTE", "Particle attributes", m_num->getValue());

		Function1Pt::Copy(*(this->GetNewPositionBuffer()->getDataPtr()), positions);
		Function1Pt::Copy(*(this->GetNewVelocityBuffer()->getDataPtr()), velocities);
		Function1Pt::Copy(*(this->GetAttributeBuffer()->getDataPtr()), attributes);

		int nn = this->GetParticleNumber();
		this->addModule("CONSTRAIN_DENSITY", new DensityPBD<TDataType>(this));
		this->addModule("COMPUTE_VISCOSITY", new ViscosityBase<TDataType>(this));
		this->addModule("COMPUTE_NEIGHBORS", new NeighborQuery<TDataType>(this));
		this->addModule("COMPUTE_DENSITY", new SummationDensity<TDataType>(this));
		this->addModule("COMPUTE_SURFACE_TENSION", new SurfaceTension<TDataType>(this));
 		this->addModule("PREDICT_PARTICLES", new ParticlePrediction<TDataType>(this));

		BoundaryManager<TDataType>* bmgr = new BoundaryManager<TDataType>(this);

		DistanceField3D * box = new DistanceField3D();
		box->SetSpace(this->GetLowerBound() - this->GetSamplingDistance() * 5, this->GetUpperBound() + this->GetSamplingDistance() * 5, 105, 105, 105);
		box->DistanceFieldToBox(this->GetLowerBound(), this->GetUpperBound(), true);
		//		box->DistanceFieldToSphere(make_float3(0.5f), 0.2f, true);
		bmgr->InsertBarrier(new BarrierDistanceField3D(box));
		this->addModule("BOUNDARY_HANDLING", bmgr);


		if (1)
		{
			this->execute("COMPUTE_NEIGHBORS");
			this->execute("COMPUTE_DENSITY");

			DeviceArray<Real>* gpgRho = this->GetDensityBuffer()->getDataPtr();

			Reduction<Real>* pReduce = Reduction<Real>::Create(gpgRho->Size());

			Real maxRho = pReduce->Maximum(gpgRho->getDataPtr(), gpgRho->Size());

			SummationDensity<TDataType>* sd = this->getModule<SummationDensity<TDataType>>("COMPUTE_DENSITY");// TypeInfo::CastPointerDown<SummationDensity<TDataType>>(this->GetModule("COMPUTE_DENSITY"));
			sd->SetCorrectFactor(this->GetRestDensity() / maxRho);

// 			this->Execute("COMPUTE_DENSITY");
// 			maxRho = pReduce->Maximum(gpgRho->GetDataPtr(), gpgRho->Size());

			std::cout << "Maximum Density: " << maxRho << std::endl;

			delete pReduce;
		}

		this->updateModules();

		return true;
	}

	template<typename TDataType>
	void ParticleSystem<TDataType>::SetParticleNumber(size_t num)
	{
		m_num->setValue(num);
	}


	template<typename TDataType>
	void ParticleSystem<TDataType>::advance(float dt)
	{
		Function1Pt::Copy(this->GetOldPositionBuffer()->getValue(), this->GetNewPositionBuffer()->getValue());
		Function1Pt::Copy(this->GetOldVelocityBuffer()->getValue(), this->GetNewVelocityBuffer()->getValue());

		this->execute("PREDICT_PARTICLES");
		this->execute("COMPUTE_NEIGHBORS");
//		this->Execute("COMPUTE_SURFACE_TENSION");
		this->execute("CONSTRAIN_DENSITY");
		this->execute("COMPUTE_VISCOSITY");
		this->execute("BOUNDARY_HANDLING");

// 		this->GetParticlePrediction()->PredictPosition(dt);
// 		this->GetParticlePrediction()->PredictVelocity(dt);
// 		this->GetNeighborQuery()->Execute();
// 		this->GetSurfaceTension()->Execute(dt);
// 		this->GetDensityConstraint()->Execute(dt);
// 		this->GetSummationDensity()->Execute();
// 		this->GetVelocityConstraint()->Execute(dt);
// 		this->GetViscosityModuel()->Execute(dt);
// 		//		this->GetParticlePrediction()->CorrectPosition(dt);
// 		this->GetBoundaryManager()->Execute(dt);
	}

	template<typename TDataType>
	DeviceBuffer<Attribute>* ParticleSystem<TDataType>::GetAttributeBuffer() {
		SPtr< DeviceBuffer<Attribute> > buf = getContext()->getDeviceBuffer<Attribute>("ATTRIBUTE");
		return buf.get();
	}

	template<typename TDataType>
	DeviceBuffer<NeighborList>* ParticleSystem<TDataType>::GetNeighborBuffer() {
		SPtr< DeviceBuffer<NeighborList> > buf = getContext()->getDeviceBuffer<NeighborList>("NEIGHBORHOOD");
		return buf.get();
	}
}