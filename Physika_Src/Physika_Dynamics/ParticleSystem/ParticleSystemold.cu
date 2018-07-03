#include "ParticleSystem.h"
#include "Kernel.h"
#include "Geometry/DistanceField3D.h"

using namespace std;

static float t = 0.0f;
static int flag = 1;
static const float deltaT = 0.0083f;

namespace CUDA {

	__constant__ ParticleSystem::Settings PARAMS;
	__constant__ SpikyKernel kernSpiky;
	__constant__ CubicKernel kernCubic;
	__constant__ SmoothKernel kernSmooth;

	ParticleSystem::ParticleSystem(Settings settings) : simItor(0)
	{
		params = settings;
	}

	ParticleSystem::~ParticleSystem()
	{
		hash.Release();
	}


	float ParticleSystem::GetTimeStep()
	{
		return 0.001f;
	}

	__global__ void K_SetupRendering(float3* dst, float4* color, CUDA::Array<float3> posArr, CUDA::Array<float> colorIndex)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		dst[pId] = posArr[pId];

		float a = 0.0f;
		float b = 0.0f;
		if (colorIndex[pId] > 0.0f)
			a = colorIndex[pId] / 10000.0f;
		if (a > 1.0f)
			a = 1.0f;

		if (colorIndex[pId] < 0.0f)
			b = - colorIndex[pId] / 10000.0f;
		if (b > 1.0f)
			b = 1.0f;
		color[pId] = colorIndex[pId] > 0.0f ? make_float4(1.0-a, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 1.0-b, 0.0f, 1.0f);

//		color[pId] = colorIndex[pId] > 0.0f ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f,1.0f);

//		printf("%f \n", colorIndex[pId]);
	}

	void ParticleSystem::Step(float dt)
	{
		cudaCheck(cudaMemcpyToSymbol(PARAMS, &params, sizeof(Settings)));

		CTimer t_predict;
		t_predict.Start();

		ComputeNeighbors();

//		ComputeSurfaceTension(dt);

		Predict(dt);

		t_predict.Stop();
		t_predict.OutputString("Predict: ");


		CTimer t_pressure;
		t_pressure.Start();
	
		CorrectWithPBD(dt);

//		Projection(dt);
//		CorrectWithEnergy(dt);

		//if (simItor % 30 == 0)
// 		{
// 			float3* dst = renderer->GetBuffer();
// 			float4* clr = renderer->GetColorBuffer();
// 			dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
// 			K_SetupRendering << <pDims, BLOCK_SIZE >> > (dst, clr, posArr, lambdaArr);
// 			renderer->UnMapCudaBuffer();
// 			renderer->UnmapCudaColorBuffer();
// 		}
		
		t_pressure.Stop();
		t_pressure.OutputString("Incompressibility: ");

		//Output total kinetic energy
/*		float3* bufArr = new float3[params.pNum];
		cudaCheck(cudaMemcpy(bufArr, velArr.data, params.pNum * sizeof(float3), cudaMemcpyDeviceToHost));

		double E = 0.0;
		for (int i = 0; i < params.pNum; i++)
		{
			E += bufArr[i].x * bufArr[i].x + bufArr[i].y * bufArr[i].y + bufArr[i].z * bufArr[i].z;
		}

		std::cout << "Total Kinetic Energy: " << E / params.pNum << std::endl;
		delete [] bufArr;*/

//
		CTimer t_viscosity;
		t_viscosity.Start();

		ApplyViscosity(dt);

		BoundaryHandling(dt);

		t_viscosity.Stop();
		t_viscosity.OutputString("Predict: ");
		
//		cudaMemcpy(data, posArr.data, params.pNum * sizeof(float3), cudaMemcpyDeviceToHost);
	}


	void ParticleSystem::TakeOneFrame()
	{
		float dt = 0.001f;
		PreProcessing();

		cout << "---------------------------------Frame " << simItor << " Begin!--------------------------" << endl;
//		m_start = clock();

		Advance(dt);

//		m_end = clock();
		cout << "------------------------Costs totally " << GetTimeCostPerFrame() << " million seconds!----------------" << endl;

		PostProcessing();

		cout << endl << endl << endl;

		simItor++;
	}

	void ParticleSystem::Advance(float dt)
	{
		Step(dt);
	}

	void ParticleSystem::AllocateMemory()
	{
		std::cout << "Total particle number: " << params.pNum << std::endl;

		posArr.Resize(params.pNum);
		velArr.Resize(params.pNum);
		rhoArr.Resize(params.pNum);

		lambdaArr.Resize(params.pNum);

		buffer.Resize(params.pNum);
		divArr.Resize(params.pNum);
		preArr.Resize(params.pNum);
		aiiArr.Resize(params.pNum);
		pBufArr.Resize(params.pNum);
		bSurface.Resize(params.pNum);
		aiiSymArr.Resize(params.pNum);

		neighborsArr.Resize(params.pNum);
	}

	void ParticleSystem::ComputeNeighbors()
	{
		//hash.QueryNeighbors(posArr, neighborsArr, params.smoothingLength, params.samplingDistance, NEIGHBOR_SIZE);
		hash.QueryNeighborSlow(posArr, neighborsArr, params.smoothingLength, NEIGHBOR_SIZE);
	}

	__global__ void K_Predict(CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];
		float3 vel_i = velArr[pId];

		//update velocity vi = vi + dt * fExt
// 		float3 center = make_float3(0.2f, 0.2f, 0.2f);
// 		float3 dir = normalize(center - pos_i);
// 		vel_i += 10.0f*dir * dt;

		vel_i += make_float3(0.0f, -9.8f, 0.0f)*dt;

		//predict position x* = xi + dt * vi
		pos_i += vel_i*dt;

		posArr[pId] = pos_i;
		velArr[pId] = vel_i;
	}

	void ParticleSystem::Predict(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		K_Predict << <pDims, BLOCK_SIZE >> > (posArr, velArr, dt);
	}

	__global__ void K_ConstrainPosition(CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, float3 lo, float3 hi)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos = posArr[pId];
		float3 vel = velArr[pId];

		if (pos.x < lo.x) {
			vel.x = 0;
			pos.x = lo.x;
		}
		else if (pos.x > hi.x) {
			vel.x = 0;
			pos.x = hi.x;
		}

		if (pos.y < lo.y) {
			vel.y = 0;
			pos.y = lo.y;
		}
		else if (pos.y > hi.y) {
			vel.y = 0;
			pos.y = hi.y;
		}

		if (pos.z < lo.z) {
			vel.z = 0;
			pos.z = lo.z;
		}
		else if (pos.z > hi.z) {
			vel.z = 0;
			pos.z = hi.z;
		}

		posArr[pId] = pos;
		velArr[pId] = vel;
	}

	void ParticleSystem::BoundaryHandling(float dt)
	{
		m_boundary.Constrain(posArr, velArr, dt);

		// 	uint pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		// 	K_ConstrainPosition << < pDims, BLOCK_SIZE >> >(posArr, velArr, hash.lo+0.02f, hash.hi-0.02f);
	}

	__global__ void K_ComputeDivergence(CUDA::Array<float> divArr, CUDA::Array<float> aiiArr, CUDA::Array<float> rhoArr, CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];
		float3 vel_i = velArr[pId];

		float div_vi = 0.0f;

		float invAii = 1.0f / aiiArr[pId];

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 g = invAii*kernCubic.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				float div_ij = dot(vel_i - velArr[j], g)*PARAMS.restDensity / dt;
				atomicAdd(&divArr[pId], div_ij);
				atomicAdd(&divArr[j], div_ij);
			}
		}

//		divArr[pId] = div_vi*PARAMS.restDensity/dt;

//		printf("%f \n", divArr[pId]);
// 		if (rhoArr[pId] > PARAMS.restDensity)
// 		{
// 			atomicAdd(&divArr[pId], 0.1f*(rhoArr[pId] - PARAMS.restDensity) / dt);
// 		}
	}

	__global__ void K_ComputeCoefficient(CUDA::Array<float> coefArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float A_i = 0.0f;
		float con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float a_ij = -kernCubic.Gradient(r, PARAMS.smoothingLength);
				A_i += a_ij;
			}
		}

		A_i *= con1;
		A_i = A_i < EPSILON ? 1.0f : A_i;

		coefArr[pId] = A_i;
	}

	__global__ void K_ComputeSymCoefficient(CUDA::Array<float> aiiSymArr, CUDA::Array<float> aiiArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		atomicAdd(&aiiSymArr[pId], 1.0f);

		float invAii = 1.0f/aiiArr[pId];
		float3 pos_i = posArr[pId];

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float a_ij = -invAii * kernCubic.Gradient(r, PARAMS.smoothingLength);
				atomicAdd(&aiiSymArr[j], a_ij);
			}
		}
	}

	__global__ void K_OneJacobiStep(CUDA::Array<float> resArr, CUDA::Array<float> preArr, CUDA::Array<float> aiiArr, CUDA::Array<float> divArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float invAii = 1.0f / aiiArr[pId];

//		float residual = divArr[pId];
		atomicAdd(&resArr[pId], divArr[pId]);

		float con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float a_ij = -invAii*kernCubic.Gradient(r, PARAMS.smoothingLength);
//				residual += con1*a_ij*preArr[j];
				atomicAdd(&resArr[pId], con1*a_ij*preArr[j]);
				atomicAdd(&resArr[j], con1*a_ij*preArr[pId]);
			}
		}
//		resArr[pId] = residual/* / coefArr[pId]*/;
	}

	__global__ void K_ComputeSymPressure(CUDA::Array<float> preArr, CUDA::Array<float> aiiSymArr, CUDA::Array<float> resArr)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= resArr.Size()) return;

		preArr[pId] = resArr[pId] / aiiSymArr[pId];

//		printf("%f \n", aiiSymArr[pId]);
	}

	__global__ void K_DetectSurface(CUDA::Array<bool> bSurface, CUDA::Array<float> coefArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float total_weight = 0.0f;
		float3 div_i = make_float3(0.0f);

		float3 pos_i = posArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float weight = -kernSmooth.Gradient(r, PARAMS.smoothingLength);
				total_weight += weight;
				div_i += (posArr[j] - pos_i)*(weight / r);
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		float absDiv = length(div_i) / total_weight;

		if ((absDiv > 0.25f && coefArr[pId] < PARAMS.maxAii) || nbSize < 2)
		{
			bSurface[pId] = true;
			coefArr[pId] = PARAMS.maxAii;
		}
		else
			bSurface[pId] = false;
	}

	__global__ void K_UpdateVelocity2(CUDA::Array<float> preArr, CUDA::Array<float> aiiArr, CUDA::Array<bool> bSurface, CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float invAii = 1.0f / aiiArr[pId];

		float scale = 0.5f*dt / PARAMS.restDensity;// *PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		float3 dv_i = make_float3(0.0f);
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float weight = invAii*kernCubic.Gradient(r, PARAMS.smoothingLength);
				float3 dnij = - scale*(pos_i - posArr[j])*(1.0f / r)*weight;

				float3 dvij = (preArr[j] - preArr[pId])*dnij;
				float3 dvjj = preArr[j] * dnij;

				if (bSurface[pId])
				{
					dv_i += dvjj;
				}
				else
				{
					dv_i += dvij;
				}

// 				atomicAdd(&velArr[j].x, 0.5f*dvij.x);
// 				atomicAdd(&velArr[j].y, 0.5f*dvij.y);
// 				atomicAdd(&velArr[j].z, 0.5f*dvij.z);
			}
		}

		velArr[pId] += 0.5f*dv_i;
// 		atomicAdd(&velArr[pId].x, 0.5f*dv_i.x);
// 		atomicAdd(&velArr[pId].y, 0.5f*dv_i.y);
// 		atomicAdd(&velArr[pId].z, 0.5f*dv_i.z);
	}

	void ParticleSystem::Projection(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		
		ComputeDensity();

		K_ComputeCoefficient << <pDims, BLOCK_SIZE >> > (aiiArr, posArr, neighborsArr);
		K_DetectSurface << <pDims, BLOCK_SIZE >> > (bSurface, aiiArr, posArr, neighborsArr, dt);

		aiiSymArr.Reset();
		K_ComputeSymCoefficient << <pDims, BLOCK_SIZE >> > (aiiSymArr, aiiArr, posArr, neighborsArr);

		divArr.Reset();
		K_ComputeDivergence << <pDims, BLOCK_SIZE >> > (divArr, aiiArr, rhoArr, posArr, velArr, neighborsArr, dt);
		
		lambdaArr.Reset();
		pBufArr.Reset();
		for (int i = 0; i < 100; i++)
		{
			lambdaArr.Swap(pBufArr);
			lambdaArr.Reset();
			K_OneJacobiStep << <pDims, BLOCK_SIZE >> > (lambdaArr, pBufArr, aiiArr, divArr, posArr, neighborsArr, dt);
			lambdaArr.Swap(pBufArr);
			K_ComputeSymPressure << <pDims, BLOCK_SIZE >> > (lambdaArr, aiiSymArr, pBufArr);
		}

		K_UpdateVelocity2 << <pDims, BLOCK_SIZE >> > (lambdaArr, aiiArr, bSurface, posArr, velArr, neighborsArr, dt);

/*		K_ComputeCoefficient << <pDims, BLOCK_SIZE >> > (aiiArr, posArr, neighborsArr);
		K_DetectSurface << <pDims, BLOCK_SIZE >> > (bSurface, aiiArr, posArr, neighborsArr, dt);
		lambdaArr.Clear();
		pBufArr.Clear();
		for (int i = 0; i < 50; i++)
		{
			K_ComputeDivergence << <pDims, BLOCK_SIZE >> > (divArr, posArr, velArr, neighborsArr, dt);
			lambdaArr.Swap(pBufArr);
			pBufArr.Clear();
			K_OneJacobiStep << <pDims, BLOCK_SIZE >> > (lambdaArr, pBufArr, aiiArr, divArr, posArr, neighborsArr, dt);
			K_UpdateVelocity2 << <pDims, BLOCK_SIZE >> > (lambdaArr, bSurface, posArr, velArr, neighborsArr, dt);
		}*/
	}

	__global__ void K_ComputeAii(CUDA::Array<float> aiiArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float VOL = PARAMS.mass / PARAMS.restDensity;

		float coef_i = 0.0f;
		float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);
		float div_vi = 0.0f;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 g = VOL*kernCubic.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				float normG = dot(g, g);

				grad_ci += g;
				coef_i += normG;
			}
		}

		coef_i += dot(grad_ci, grad_ci);
		aiiArr[pId] = coef_i;
	}

	bool ParticleSystem::Initialize(std::string in_filename)
	{
		AllocateMemory();

		hash.SetSpace(2*params.samplingDistance, params.lowBound, params.upBound);

		std::vector<float3> poss;
		std::vector<float3> vels;
		for (int m = 0; m < models.size(); m++)
		{
			int num = models[m]->positions.size();
			for (int i = 0; i < num; i++)
			{
				poss.push_back(models[m]->positions[i]);
				vels.push_back(models[m]->velocities[i]);
			}
		}
		
		posArr.CopyFrom(poss);
		velArr.CopyFrom(vels);

		poss.clear();
		vels.clear();

		cudaCheck(cudaMemcpyToSymbol(PARAMS, &params, sizeof(Settings)));

		ComputeNeighbors();
		ComputeDensity();

		float maxRho = 0.0f;

		float* bufArr = new float[params.pNum];
		cudaCheck(cudaMemcpy(bufArr, rhoArr.GetDataPtr(), params.pNum * sizeof(float), cudaMemcpyDeviceToHost));


		for (int i = 0; i < params.pNum; i++)
		{
			if (bufArr[i] > maxRho)
			{
				maxRho = bufArr[i];
			}
		}

		params.mass *= params.restDensity / maxRho;
		std::cout << "Mass per particle: " << params.mass << std::endl;

		cudaCheck(cudaMemcpyToSymbol(PARAMS, &params, sizeof(Settings)));

		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		K_ComputeCoefficient << <pDims, BLOCK_SIZE >> > (aiiArr, posArr, neighborsArr);
//		K_ComputeAii<< <pDims, BLOCK_SIZE >> > (aiiArr, posArr, neighborsArr);
		cudaCheck(cudaMemcpy(bufArr, aiiArr.GetDataPtr(), params.pNum * sizeof(float), cudaMemcpyDeviceToHost));

		float maxAii = 0.0f;
		for (int i = 0; i < params.pNum; i++)
		{
			if (bufArr[i] > maxAii)
			{
				maxAii = bufArr[i];
			}
		}
		params.maxAii = maxAii;
		std::cout << "Maximum Aii: " << maxAii << std::endl;

		cudaCheck(cudaMemcpyToSymbol(PARAMS, &params, sizeof(Settings)));
		// 	cudaCheck(cudaMemcpyToSymbol(PARAMS, &params, sizeof(SolverParams)));
		// 
		// 	ComputeDensity();
		// 
		// 	cudaCheck(cudaMemcpy(rho, rhoArr.data, params.pNum * sizeof(float), cudaMemcpyDeviceToHost));
		// 	for (int i = 0; i < params.pNum; i++)
		// 	{
		// 		std::cout << "Density: " << rho[i] << std::endl;
		// 	}
		// 
		// 	delete[]rho;
		// 
		// 	exit(0);

		InitialSceneBoundary();

		return true;
	}


	void ParticleSystem::InitialSceneBoundary()
	{
		CUDA::DistanceField3D * box = new CUDA::DistanceField3D();
		box->SetSpace(params.lowBound - params.samplingDistance * 5, params.upBound + params.samplingDistance * 5, 100, 100, 100);
//		box->DistanceFieldToBox(params.lowBound, params.upBound, true);
		box->DistanceFieldToSphere(make_float3(0.5f, 0.5f, 0.5f), 0.2f, true);
		m_boundary.InsertBarrier(new CUDA::BarrierDistanceField3D(box));


// 		CUDA::DistanceField3D* bunny = new CUDA::DistanceField3D("data/cow.sdf");
// 		bunny->Scale(1.0f);
// 		bunny->Translate(make_float3(0.9f, 0.0f, 0.0f));
// 		m_boundary.InsertBarrier(new CUDA::BarrierDistanceField3D(bunny));

	}

	void ParticleSystem::CorrectWithPBD(float dt)
	{
		int total_itoration = 3;

		int itor = 0;
		do {
			ComputeDensity();

			IterationPBD(dt);

			itor++;

		} while (itor <= total_itoration);
	}


	void ParticleSystem::CorrectWithEnergy(float dt)
	{
		int total_itoration = 50;
		float avgDiv = 0.0f;
		float firstAvgDiv = 0.0f;
		float preDiv = 0.0f;

		lambdaArr.Reset();
		int itor = 0;
		do {
//			ComputeDensity();

			IterationEnergy(dt);

/*			float* bufArr = new float[params.pNum];
			cudaCheck(cudaMemcpy(bufArr, divArr.data, params.pNum * sizeof(float), cudaMemcpyDeviceToHost));


			for (int i = 0; i < params.pNum; i++)
			{
				avgDiv += abs(bufArr[i]);
			}
			avgDiv /= params.pNum;

			if (itor == 0)
			{
				firstAvgDiv = avgDiv;
			}
			else
			{
				cout << "Relative Error: " << abs(avgDiv - preDiv) / firstAvgDiv << endl;
			}

			preDiv = avgDiv;*/

			itor++;

		} while (itor <= total_itoration);

/*		float* bufArr = new float[params.pNum];
		cudaCheck(cudaMemcpy(bufArr, divArr.data, params.pNum * sizeof(float), cudaMemcpyDeviceToHost));

		for (int i = 0; i < params.pNum; i++)
		{
			avgDiv += abs(bufArr[i]);
		}
		avgDiv /= params.pNum;
		delete [] bufArr;
		cout << "Divergence: " << abs(avgDiv) << endl;*/
	}

	__global__ void K_ComputePressures(CUDA::Array<bool> bSurface, CUDA::Array<float> tatalP, CUDA::Array<float> deltaP, CUDA::Array<float> divArr, CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];
		float3 vel_i = velArr[pId];

		float VOL = PARAMS.mass/PARAMS.restDensity;

		float coef_i = 0.0f;
		float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);
		float div_vi = 0.0f;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 g = VOL*kernCubic.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				//float3 g2 = VOL*kernSmooth.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				float normG = dot(g, g);
				float div_ij = dot(vel_i - velArr[j], g);

				grad_ci += g;
				coef_i += normG;
				div_vi += div_ij;
			}
		}

		coef_i += dot(grad_ci, grad_ci);
		div_vi *= (PARAMS.restDensity / dt);
		div_vi -= 0.5f;

//		printf("%f \n", div_vi);

// 		if (coef_i < PARAMS.maxAii)
// 		{
// 			coef_i = PARAMS.maxAii;
// 			bSurf[pId] = true;
// 		}
// 		else
// 		{
// 			bSurf[pId] = false;
// 		}
//		if (coef_i < PARAMS.maxAii) coef_i = PARAMS.maxAii;
		if (bSurface[pId]) coef_i = PARAMS.maxAii;

		divArr[pId] = div_vi;
		deltaP[pId] = - div_vi / coef_i;
		tatalP[pId] += deltaP[pId];
	}

	__global__ void K_ComputeDisplacement2(CUDA::Array<bool> bSurface, CUDA::Array<float3> dPos, CUDA::Array<float> lambdas, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		//float3 dpos_i = make_float3(0.0f, 0.0f, 0.0f);
		float lamda_i = lambdas[pId];
		float weight = PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 dp_ij = 0.25f*weight*(lambdas[j] + lamda_i)*kernCubic.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				float3 dp_j = 0.25f*weight*(lamda_i)*kernCubic.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				//dpos_i += dp_ij;

// 				if (bSurface[pId])
// 				{
// 					atomicAdd(&dPos[pId].x, dp_j.x);
// 					atomicAdd(&dPos[pId].y, dp_j.y);
// 					atomicAdd(&dPos[pId].z, dp_j.z);
// 				}
// 				else
// 				{
// 					atomicAdd(&dPos[pId].x, dp_ij.x);
// 					atomicAdd(&dPos[pId].y, dp_ij.y);
// 					atomicAdd(&dPos[pId].z, dp_ij.z);
// 				}
				
				atomicAdd(&dPos[pId].x, dp_ij.x);
 				atomicAdd(&dPos[pId].y, dp_ij.y);
 				atomicAdd(&dPos[pId].z, dp_ij.z);

				atomicAdd(&dPos[j].x, -dp_ij.x);
				atomicAdd(&dPos[j].y, -dp_ij.y);
				atomicAdd(&dPos[j].z, -dp_ij.z);
			}
		}
		//	dPos[pId] = dpos_i;
	}

	__global__ void K_UpdatePosition2(CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<float3> accArr, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

//		posArr[pId] += 0.5f*dt*dt*accArr[pId];
		velArr[pId] += dt*accArr[pId];
	}

	void ParticleSystem::IterationEnergy(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));

		K_ComputeAii << <pDims, BLOCK_SIZE >> > (aiiArr, posArr, neighborsArr);
		K_DetectSurface << <pDims, BLOCK_SIZE >> > (bSurface, aiiArr, posArr, neighborsArr, dt);

		K_ComputePressures<< <pDims, BLOCK_SIZE >> > (bSurface, lambdaArr, preArr, divArr, posArr, velArr, neighborsArr, dt);
		buffer.Reset();
		K_ComputeDisplacement2 << <pDims, BLOCK_SIZE >> > (bSurface, buffer, preArr, posArr, neighborsArr, dt);
		K_UpdatePosition2 << <pDims, BLOCK_SIZE >> > (posArr, velArr, buffer, dt);
	}

	__global__ void K_ComputeLambdas(CUDA::Array<float> lambdaArr, CUDA::Array<float> rhoArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float lamda_i = 0.0f;
		float3 grad_ci = make_float3(0.0f, 0.0f, 0.0f);

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 g = PARAMS.mass*kernSpiky.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += dot(g, g);
			}
		}

		lamda_i += dot(grad_ci, grad_ci);

		float rho_i = rhoArr[pId];
		// 	if (rho_i > 1010)
		// 	{
		// 		rho_i = 1010.0f;
		// 	}

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	__global__ void K_ComputeDisplacement(CUDA::Array<float3> dPos, CUDA::Array<float> lambdas, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float3 pos_i = posArr[pId];

		float3 dpos_i = make_float3(0.0f, 0.0f, 0.0f);
		float lamda_i = lambdas[pId];

		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 dp_ij = 0.01f*0.5f*(lamda_i + lambdas[j])*PARAMS.mass*kernSpiky.Gradient(r, PARAMS.smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
				dpos_i += dp_ij;
				atomicAdd(&dPos[pId].x, dp_ij.x);
				atomicAdd(&dPos[pId].y, dp_ij.y);
				atomicAdd(&dPos[pId].z, dp_ij.z);

				atomicAdd(&dPos[j].x, -dp_ij.x);
				atomicAdd(&dPos[j].y, -dp_ij.y);
				atomicAdd(&dPos[j].z, -dp_ij.z);
			}
		}
		//	dPos[pId] = dpos_i;
	}

	__global__ void K_UpdatePosition(CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<float3> dPos, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		posArr[pId] += dPos[pId];
		velArr[pId] += dPos[pId] / dt;
	}

	void ParticleSystem::IterationPBD(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));

		K_ComputeLambdas << <pDims, BLOCK_SIZE >> > (lambdaArr, rhoArr, posArr, neighborsArr);
		buffer.Reset();
		K_ComputeDisplacement << <pDims, BLOCK_SIZE >> > (buffer, lambdaArr, posArr, neighborsArr, dt);
		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (posArr, velArr, buffer, dt);
	}

	__global__ void K_ComputeDensity(CUDA::Array<float> rhoArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float r;
		float rho_i = 0.0f;
		float3 pos_i = posArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			r = length(pos_i - posArr[j]);

			rho_i += PARAMS.mass*kernSpiky.Weight(r, PARAMS.smoothingLength);
		}
		rhoArr[pId] = rho_i;
	}

	void ParticleSystem::ComputeDensity()
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		K_ComputeDensity << < pDims, BLOCK_SIZE >> > (rhoArr, posArr, neighborsArr);
	}

	__global__ void K_ComputeSurfaceEnergy(CUDA::Array<float> energyArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float total_weight = 0.0f;
		float3 dir_i = make_float3(0.0f);

		float3 pos_i = posArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float weight = -kernSmooth.Gradient(r, PARAMS.smoothingLength);
				total_weight += weight;
				dir_i += (posArr[j] - pos_i)*(weight / r);
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		float absDir = length(dir_i) / total_weight;

		energyArr[pId] = absDir*absDir;
	}

	__global__ void K_ComputeSurfaceTension(CUDA::Array<float3> velArr, CUDA::Array<float> energyArr, CUDA::Array<float3> posArr, CUDA::Array<NeighborList> neighbors, float dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float Vref = PARAMS.mass / PARAMS.restDensity;


		float alpha = (float) 945.0f / (32.0f * (float)M_PI * PARAMS.smoothingLength * PARAMS.smoothingLength * PARAMS.smoothingLength);
		float ceof = 80000.0f * alpha;

		float3 F_i = make_float3(0.0f);

		float3 pos_i = posArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			float r = length(pos_i - posArr[j]);

			if (r > EPSILON)
			{
				float3 temp = Vref*Vref*kernSmooth.Gradient(r, PARAMS.smoothingLength)*(posArr[j] - pos_i) * (1.0f / r);
				F_i += ceof*1.0f*(energyArr[pId])*temp;
			}
		}

		velArr[pId] -= dt*F_i / PARAMS.mass;
	}

	void ParticleSystem::ComputeSurfaceTension(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		K_ComputeSurfaceEnergy << < pDims, BLOCK_SIZE >> > (lambdaArr, posArr, neighborsArr);
		K_ComputeSurfaceTension << < pDims, BLOCK_SIZE >> > (velArr, lambdaArr, posArr, neighborsArr, dt);
	}

	__device__ float VisWeight(const float r, const float h)
	{
		float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const float d = 1.0f - q;
			const float RR = h*h;
			return 45.0f / (13.0f * (float)M_PI * RR *h) *d;
		}
	}

	__global__ void K_ApplyViscosity(CUDA::Array<float3> dVel, CUDA::Array<float3> posArr, CUDA::Array<float3> velArr, CUDA::Array<NeighborList> neighbors)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.Size()) return;

		float r;
		float3 dv_i = make_float3(0.0f, 0.0f, 0.0f);
		float3 pos_i = posArr[pId];
		float3 vel_i = velArr[pId];
		int nbSize = neighbors[pId].size;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors[pId][ne];
			r = length(pos_i - posArr[j]);

			dv_i += 0.00005f*PARAMS.mass*(velArr[j] - vel_i)*VisWeight(r, PARAMS.smoothingLength);
		}
		dVel[pId] = dv_i;
	}

	__global__ void K_UpdateVelocity(CUDA::Array<float3> velArr, CUDA::Array<float3> dVel)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.Size()) return;

		velArr[pId] += dVel[pId];
	}

	void ParticleSystem::ApplyViscosity(float dt)
	{
		dim3 pDims = int(ceil(posArr.Size() / BLOCK_SIZE + 0.5f));
		K_ApplyViscosity << < pDims, BLOCK_SIZE >> > (buffer, posArr, velArr, neighborsArr);
		K_UpdateVelocity << < pDims, BLOCK_SIZE >> > (velArr, buffer);
	}

}