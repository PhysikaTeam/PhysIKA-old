#pragma once
#ifndef _SSESANDSOLVER_H
#define _SSESANDSOLVER_H

#include "SandSolverInterface.h"
#include "SandGrid.h"
#include "Core/Utility/Reduction.h"
#include "Core/Array/DynamicArray.h"

#include "Dynamics/Sand/swe/SandGrid.h"

#include "Dynamics/HeightField/HeightFieldGrid.h"

namespace PhysIKA
{
	class SSESandSolver :public SandSolverInterface
	{
	public:

		SSESandSolver();

		virtual ~SSESandSolver();

		virtual bool initialize()override;

		virtual bool stepSimulation(float deltime) override;

		void advection(float deltime);

		void updateVeclocity(float deltime);


		void updateSandStaticHeight(float dt);


		void updateSandGridHeight() { m_sandData.updateSandGridHeight(); }

		void setSandGridHeight(DeviceHeightField1d& sandheight) { m_sandData.setSandGridHeight(sandheight); }
		void setSandGridHeight(HostHeightField1d& sandheight) { m_sandData.setSandGridHeight(sandheight); }
		void setSandGridHeight(double* sandheight) { m_sandData.setSandGridHeight(sandheight); }
		void getSandGridHeight(double* sandheight) { m_sandData.getSandGridHeight(sandheight); }

		virtual void setSandGridInfo(const SandGridInfo& sandinfo) override;

		SandGridInfo* getSandGridInfo() { return &m_SandInfo; }

		SandGrid& getSandGrid() { return m_sandData; }
		const SandGrid& getSandGrid()const { return m_sandData; }

		/**
		*@brief Calculate max time step according to CFL condition.
		*/
		virtual float getMaxTimeStep() override;

		void applyVelocityChange(float dt, int minGi, int minGj, int sizeGi, int sizeGj);

		void setCFLNumber(float cfl) { m_CFLNumber = cfl; }
		float getCFLNumber()const { return m_CFLNumber; }

	public:
		DeviceDArray<Vector3d>* m_gridVel = 0;

		DeviceHeightField1d m_sandStaticHeight;
		DeviceHeightField1d m_macStaticHeightx;
		DeviceHeightField1d m_macStaticHeightz;



	private:


	private:
		SandGrid m_sandData;
		//SandGridInfo m_sandinfo;


		int m_threadBlockx = 16;
		int m_threadBlocky = 16;

		float m_CFLNumber = 0.1f;// 0.1f;
		float m_maxTimeStep = 0.04f;

		Reduction<float>* m_CFLReduction = 0;
		DeviceArray<float> m_velocityNorm;


	private:
	};
}
#endif