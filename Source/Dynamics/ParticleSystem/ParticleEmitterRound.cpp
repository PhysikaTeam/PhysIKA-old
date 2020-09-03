#include "ParticleEmitterRound.h"
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <stdlib.h>

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ParticleEmitterRound, TDataType)

	template<typename TDataType>
	ParticleEmitterRound<TDataType>::ParticleEmitterRound(std::string name)
		: ParticleEmitter<TDataType>(name)
	{

		srand(time(0));
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

		getRotMat(dir / dir.norm());

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;

		Real lo = -radius;
		Real hi = +radius;

		for (Real x = lo; x <= hi; x += sampling_distance)
		{
			for (Real y = lo; y <= hi; y += sampling_distance)
			{
				Coord p = Coord(x, 0, y);
				if ((p - Coord(0)).norm() < radius)
				{
					Coord q = cos(angle) * p + (1 - cos(angle))*(p.dot(axis))*axis + sin(angle) * axis.cross(p);
					pos_list.push_back(q + centre);
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
					Coord p = Coord(x, 0, y);
					if ((p - Coord(0)).norm() < radius && rand() % 4 == 0)
					{/*
						Real aa, bb, cc;
						do
						{
							aa = Real(rand() % 2000 - 1000) / 1000.0;
							bb = Real(rand() % 2000 - 1000) / 1000.0;
							cc = Real(rand() % 2000 - 1000) / 1000.0;
						} while (aa * aa + bb * bb + cc * cc < 1.0);
						*/
						Coord q = cos(angle) * p + (1 - cos(angle)) * (p.dot(axis)) * axis + sin(angle) * axis.cross(p);
						pos_list.push_back(q + centre);
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
				Coord p = Coord(x, 0, y);
				if ((p - Coord(0)).norm() < radius && rand() % 40 == 0)
				{/*
					Real aa, bb, cc;
					do
					{
						aa = Real(rand() % 2000 - 1000) / 1000.0;
						bb = Real(rand() % 2000 - 1000) / 1000.0;
						cc = Real(rand() % 2000 - 1000) / 1000.0;
					} while (aa * aa + bb * bb + cc * cc < 1.0);
					*/
					Coord q = cos(angle) * p + (1 - cos(angle)) * (p.dot(axis)) * axis + sin(angle) * axis.cross(p);
					pos_list.push_back(q + centre);
					vel_list.push_back(dir);
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


}