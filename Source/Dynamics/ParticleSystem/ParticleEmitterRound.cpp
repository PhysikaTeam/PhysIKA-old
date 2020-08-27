#include "ParticleEmitterRound.h"
#include <time.h>

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleEmitterRound, TDataType)

	template<typename TDataType>
	ParticleEmitterRound<TDataType>::ParticleEmitterRound(std::string name)
		: ParticleEmitter<TDataType>(name)
	{

		
	}

	
	
	template<typename TDataType>
	ParticleEmitterRound<TDataType>::~ParticleEmitterRound()
	{
		gen_pos.release();
	}

	template<typename TDataType>
	void ParticleEmitterRound<TDataType>::setInfo(Coord pos, Coord direction, Real r, Real distance)
	{
		printf("setInfo inside\n");
		radius = r;
		sampling_distance = distance;
		centre = pos;
		dir = direction;

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Real lo = -radius;
		Real hi = +radius;

		for (Real x = lo; x <= hi; x += sampling_distance)
		{
			for (Real y = lo; y <= hi; y += sampling_distance)
			{
				Coord p = Coord(x, y, 0);
				if ((p - Coord(0)).norm() < radius)
				{
					pos_list.push_back(Coord(x, 0, y) + centre);
					vel_list.push_back(direction);
				}
			}
		}

		gen_pos.resize(pos_list.size());
		gen_vel.resize(pos_list.size());

		Function1Pt::copy(gen_pos, pos_list);
		Function1Pt::copy(gen_vel, vel_list);


		

		printf("setInfo outside 0\n");

		this->currentPosition()->setElementCount(pos_list.size());
		this->currentVelocity()->setElementCount(pos_list.size());
		this->currentForce()->setElementCount(pos_list.size());

		//printf("setInfo outside 1 %d\n", this->currentPosition()->getElementCount());
		
		Function1Pt::copy(this->currentPosition()->getValue(), gen_pos);
		Function1Pt::copy(this->currentVelocity()->getValue(), gen_vel);

		this->currentForce()->getReference()->reset();

		//printf("setInfo outside %d~~\n", this->currentPosition()->getElementCount());
		//this->advance(0.001);
		//printf("setInfo outside 1\n");
		pos_list.clear();
		vel_list.clear();

		if (true)
		{
			std::vector<Coord> pos_list;
			std::vector<Coord> vel_list;

			Real lo = -radius;
			Real hi = +radius;

			for (Real x = lo; x <= hi; x += sampling_distance)
			{
				for (Real y = lo; y <= hi; y += sampling_distance)
				{
					Coord p = Coord(x, y, 0);
					if ((p - Coord(0)).norm() < radius && rand() % 4 == 0)
					{
						Real aa, bb, cc;
						do
						{
							aa = Real(rand() % 2000 - 1000) / 1000.0;
							bb = Real(rand() % 2000 - 1000) / 1000.0;
							cc = Real(rand() % 2000 - 1000) / 1000.0;
						} while (aa * aa + bb * bb + cc * cc < 1.0);
						pos_list.push_back(Coord(x, 0, y) + centre);
						vel_list.push_back(Coord(aa, bb, cc));
					}
				}
			}

			gen_pos.resize(pos_list.size());
			gen_vel.resize(pos_list.size());

			Function1Pt::copy(gen_pos, pos_list);
			Function1Pt::copy(gen_vel, vel_list);




			printf("setInfo outside 0\n");

			this->currentPosition()->setElementCount(pos_list.size());
			this->currentVelocity()->setElementCount(pos_list.size());
			this->currentForce()->setElementCount(pos_list.size());

			Function1Pt::copy(this->currentPosition()->getValue(), gen_pos);
			Function1Pt::copy(this->currentVelocity()->getValue(), gen_vel);

			this->currentForce()->getReference()->reset();

			printf("setInfo outside %d~~\n", this->currentPosition()->getElementCount());
			//this->advance(0.001);
			//printf("setInfo outside 1\n");
			pos_list.clear();
			vel_list.clear();
		}
	}

	template<typename TDataType>
	void ParticleEmitterRound<TDataType>::gen_random()
	{

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Real lo = -radius;
		Real hi = +radius;

		for (Real x = lo; x <= hi; x += sampling_distance)
		{
			for (Real y = lo; y <= hi; y += sampling_distance)
			{
				Coord p = Coord(x, y, 0);
				if ((p - Coord(0)).norm() < radius && rand() % 400 == 0)
				{
					Real aa, bb, cc;
					do
					{
						aa = Real(rand() % 2000 - 1000) / 1000.0;
						bb = Real(rand() % 2000 - 1000) / 1000.0;
						cc = Real(rand() % 2000 - 1000) / 1000.0;
					} 
					while (aa * aa + bb * bb + cc * cc < 1.0);
					pos_list.push_back(Coord(x, 0, y) + centre);
					vel_list.push_back(Coord(aa ,bb,cc));
				}
			}
		}

		gen_pos.resize(pos_list.size());
		gen_vel.resize(pos_list.size());

		Function1Pt::copy(gen_pos, pos_list);
		Function1Pt::copy(gen_vel, vel_list);




		printf("setInfo outside 0\n");

		
		pos_list.clear();
		vel_list.clear();
	}

	template<typename TDataType>
	void ParticleEmitterRound<TDataType>::advance(Real dt)
	{
		sum++;
		bool random_gen = true;
		if(! random_gen)
		{ 
			if (sum % 40 != 0) return;
		}
		else
		{
			gen_random();
		}
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

		printf("%d %d %d\n", cur_size, gen_pos.size(), this->currentPosition()->getElementCount());

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
}