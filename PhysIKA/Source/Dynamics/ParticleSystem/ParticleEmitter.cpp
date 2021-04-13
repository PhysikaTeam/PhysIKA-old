#include "ParticleEmitter.h"
#include "ParticleFluid.h"
namespace PhysIKA
{
	template<typename TDataType>
	ParticleEmitter<TDataType>::ParticleEmitter(std::string name)
		: ParticleSystem<TDataType>(name)
	{
	}

	template<typename TDataType>
	ParticleEmitter<TDataType>::~ParticleEmitter()
	{
		
	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::generateParticles()
	{

	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::advance2(Real dt)
	{
		generateParticles();

		int cur_size = this->currentPosition()->getElementCount();

		if (cur_size > 0)
		{
			DeviceArray<Coord>& cur_points0 = this->currentPosition()->getValue();
			DeviceArray<Coord>& cur_vels0 = this->currentVelocity()->getValue();
			DeviceArray<Coord>& cur_forces0 = this->currentForce()->getValue();

			pos_buf.resize(cur_size);
			vel_buf.resize(cur_size);
			force_buf.resize(cur_size);

			Function1Pt::copy(pos_buf, cur_points0);
			Function1Pt::copy(vel_buf, cur_vels0);
			Function1Pt::copy(force_buf, cur_forces0);
		}


		int total_num = cur_size + gen_pos.size();

		if (total_num > 0)
		{
			this->currentPosition()->setElementCount(cur_size + gen_pos.size());
			this->currentVelocity()->setElementCount(cur_size + gen_pos.size());
			this->currentForce()->setElementCount(cur_size + gen_pos.size());

			//printf("%d %d %d\n", cur_size, gen_pos.size(), this->currentPosition()->getElementCount());

			DeviceArray<Coord>& cur_points = this->currentPosition()->getValue();
			DeviceArray<Coord>& cur_vels = this->currentVelocity()->getValue();
			DeviceArray<Coord>& cur_forces = this->currentForce()->getValue();

			cur_points.reset();
			cur_vels.reset();
			cur_forces.reset();

			cudaMemcpy(cur_points.getDataPtr(), pos_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(cur_points.getDataPtr() + cur_size, gen_pos.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

			cudaMemcpy(cur_vels.getDataPtr(), vel_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(cur_vels.getDataPtr() + cur_size, gen_vel.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

			cudaMemcpy(cur_forces.getDataPtr(), force_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(cur_forces.getDataPtr() + cur_size, gen_pos.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

		}
		//return;
	}
	template<typename TDataType>
	void ParticleEmitter<TDataType>::advance(Real dt)
	{
		return;
		generateParticles();
		DeviceArray<Coord>& cur_points0 = this->currentPosition()->getValue();
		DeviceArray<Coord>& cur_vels0 = this->currentVelocity()->getValue();
		DeviceArray<Coord>& cur_forces0 = this->currentForce()->getValue();

		int cur_size = this->currentPosition()->getElementCount();

		pos_buf.resize(cur_size);
		vel_buf.resize(cur_size);
		force_buf.resize(cur_size);

		Function1Pt::copy(pos_buf, cur_points0);
		Function1Pt::copy(vel_buf, cur_vels0);
		Function1Pt::copy(force_buf, cur_forces0);


		this->currentPosition()->setElementCount(cur_size + gen_pos.size());
		this->currentVelocity()->setElementCount(cur_size + gen_pos.size());
		this->currentForce()->setElementCount(cur_size + gen_pos.size());

		//printf("%d %d %d\n", cur_size, gen_pos.size(), this->currentPosition()->getElementCount());

		DeviceArray<Coord>& cur_points = this->currentPosition()->getValue();
		DeviceArray<Coord>& cur_vels = this->currentVelocity()->getValue();
		DeviceArray<Coord>& cur_forces = this->currentForce()->getValue();

		cur_points.reset();
		cur_vels.reset();
		cur_forces.reset();

		cudaMemcpy(cur_points.getDataPtr(), pos_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
		cudaMemcpy(cur_points.getDataPtr() + cur_size, gen_pos.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

		cudaMemcpy(cur_vels.getDataPtr(), vel_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
		cudaMemcpy(cur_vels.getDataPtr() + cur_size, gen_vel.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

		cudaMemcpy(cur_forces.getDataPtr(), force_buf.getDataPtr(), cur_size * sizeof(Coord), cudaMemcpyDeviceToDevice);
		cudaMemcpy(cur_forces.getDataPtr() + cur_size, gen_pos.getDataPtr(), gen_pos.size() * sizeof(Coord), cudaMemcpyDeviceToDevice);

	}

	template<typename TDataType>
	void ParticleEmitter<TDataType>::updateTopology()
	{

	}

	template<typename TDataType>
	bool PhysIKA::ParticleEmitter<TDataType>::resetStatus()
	{
		this->currentPosition()->setElementCount(0);
		this->currentVelocity()->setElementCount(0);
		this->currentForce()->setElementCount(0);

		return true;
	}

}