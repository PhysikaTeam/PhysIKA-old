#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "Core/Common.h"
#include "Core/SharedData.h"
#include "Material/ISimulation.h"
#include "Core/Array.h"
#include "Core/GridHash.h"
#include "Material/BoundaryManager.h"
#include "Core/Timer.h"
#include "Scene/Model.h"

using namespace std;
namespace CUDA{

	class ParticleSystem : public ISimulation {
	public:

		class MaterialSettings
		{
		public:
			MaterialSettings() {};
		
			float smoothingLength;
			float samplingDistance;

			float restDensity;

			float viscosity;
		};


		struct Settings {
		public:
			int pNum;
			int nbrMaxSize;

			float mass;
			
			float3 lowBound;
			float3 upBound;

			float3 gravity;

			int iterNum;

			float smoothingLength;
			float samplingDistance;

			float restDensity;
			float maxAii;
		};


		ParticleSystem(Settings settings);
		~ParticleSystem();

		virtual float GetTimeStep();
		virtual void Advance(float dt);
		virtual void TakeOneFrame();

		void AllocateMemory();

		void ComputeNeighbors();

		void Step(float dt);

		void Predict(float dt);

		void ComputeDensity();

		void ComputeSurfaceTension(float dt);

		void CorrectWithPBD(float dt);

		void CorrectWithEnergy(float dt);

		void IterationEnergy(float dt);

		void IterationPBD(float dt);

		void ApplyViscosity(float dt);

		void BoundaryHandling(float dt);

	public:
		void Projection(float dt);

	public:
		void AddModel(Model* model) { models.push_back(model); }
		virtual bool Initialize(std::string in_filename);

		void InitialSceneBoundary();

	public:
		int simItor;

		CUDA::Array<float3> posArr;
		CUDA::Array<float3> velArr;

		CUDA::Array<float> rhoArr;

		CUDA::Array<float> lambdaArr;

		CUDA::Array<float3> buffer;

		CUDA::Array<float> divArr;
		CUDA::Array<float> preArr;
		CUDA::Array<float> pBufArr;
		CUDA::Array<float> aiiArr;
		CUDA::Array<bool> bSurface;

		CUDA::Array<float> aiiSymArr;

		CUDA::Array<NeighborList> neighborsArr;

		CUDA::GridHash hash;

		Settings params;

		CUDA::BoundaryManager m_boundary;

		vector<Model*> models;
	};

}

#endif
