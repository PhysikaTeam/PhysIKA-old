#ifndef __TimeStepFluidModel_h__
#define __TimeStepFluidModel_h__

#include "FluidModel.h"

// zzl---2018-6-24----start
#include <Eigen/dense>
#include "Demos/Simulation/CubicSDFCollisionDetection.h"
#include "Discregrid/all"


//auto sdf = std::unique_ptr<Discregrid::DiscreteGrid>{};
// zzl---2018-6-24----end

//-------------zzl-----2018-9-10----start-----GPU-------
#include "Demos/FluidDemo/pbf_solver_gpu.h" 
//--------------zzl----------2018-9-10------end---------

namespace PBD
{
	class SDFGradientField; //先声明类
	class TimeStepFluidModel 
	{
	protected:
		unsigned int m_velocityUpdateMethod;

		void clearAccelerations(FluidModel &model);
		void computeXSPHViscosity(FluidModel &model);
		void computeDensities(FluidModel &model);
		void updateTimeStepSizeCFL(FluidModel &model, const Real minTimeStepSize, const Real maxTimeStepSize);
		void constraintProjection(FluidModel &model);
		
		// for cloud
		void comoutePressureForce(FluidModel & model); //zzl--for cloud motion
		void computeBuoyantByGradientOfTemperature(FluidModel & model);
		void phaseChange(FluidModel & model);
		void updateAccelerations(FluidModel & model);
		void computeBuoyantByTemperature(FluidModel & model);
		void computePressure(FluidModel &model);
		void computePotentialForce(FluidModel &model,SDFGradientField &potentialField); //计算势场力
		//void computeSignDistanceAndGradient(FluidModel &model, SDFGradientField &potentialField);
		void updateParticleNum(FluidModel &model);
		
	private:
		//创建GpuParticleSolver对象，执行GPU计算
		//pbf::GpuParticleSolver m_gpuSolver;

	public:
		TimeStepFluidModel();
		virtual ~TimeStepFluidModel(void);

		//void step(FluidModel &model); //原有的函数，只需一个参数
		void step(std::string& rootPath, FluidModel &model, SDFGradientField &sdfGraField, Discregrid::CubicLagrangeDiscreteGrid &m_sdf, std::string &iniFileName, std::string &tarFileName); //zzl--8-24--为了增加几何势场的计算新增了第二/三个参数
		void reset();

		unsigned int getVelocityUpdateMethod() const { return m_velocityUpdateMethod; }
		void setVelocityUpdateMethod(unsigned int val) { m_velocityUpdateMethod = val; }
		// for SDF
		Discregrid::CubicLagrangeDiscreteGrid computeSDF(std::string fileName); //从文件中加载SDF
		//void test(std::string name) 
		//{ 
		//	std::cout << "begin" << std::endl;
		//}
	};


	//根据SDF计算得到的梯度场类
	//zzl------2018-6-26---start
	class SDFGradientField {
	private:
		std::vector<Eigen::Vector3d> m_gradient; //梯度场
		std::vector<Eigen::Vector3d> m_potentialField; //几何势场
		Eigen::AlignedBox3d m_domain;
		std::array<unsigned int, 3> m_resolution;
		Eigen::Vector3d m_cell_size; // 网格单元大小
		Eigen::Vector3d m_inv_cell_size;//单位长度含网格的数量
	public:
		//默认造函数
		SDFGradientField()
		{
			m_cell_size = { 0.1,0.1,0.1 };
			m_inv_cell_size = { 10,10,10 };
		}
		SDFGradientField(Discregrid::CubicLagrangeDiscreteGrid sdf);
		FORCE_INLINE ~SDFGradientField(void) {
			m_gradient.clear();
		}
		Eigen::Vector3d indexToPosition(unsigned int l);
		Eigen::Vector3d getGradient(Eigen::Vector3d &x); //generate the gradient at given location x
		void normGradient(); //将梯度场归一化处理
		unsigned int multiToSingleIndex(Eigen::Vector3d &x);
		void computPotentialField(Discregrid::CubicLagrangeDiscreteGrid & sdf); //计算几何势场
		Eigen::Vector3d getPotential(Eigen::Vector3d &x); //获得给定位置的几何势
		// 读取m_gradient变量
		std::vector<Eigen::Vector3d> getMGradient()
		{
			return m_gradient;
		}

		Real & getSignDistanceOfPosition(Discregrid::CubicLagrangeDiscreteGrid &sdf, Eigen::Vector3d & pos)
		{
			double distance;
			distance = sdf.interpolate(0, pos);
			return distance;
		}
	};
	//zzl---------2018-6-26----end
}

#endif
