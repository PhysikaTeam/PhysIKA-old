#ifndef __PARTICLEDATA_H__
#define __PARTICLEDATA_H__

#include <vector>
#include "Common/Common.h"


namespace PBD
{
	class PotentialField
	{
		const int X_RESOLUTION = 64;
		const int Y_RESOLUTION = 64;
		const int Z_RESOLUTION = 64;
	private:
		Vector3r m_size; //x,y,z方向的大小
		std::vector<std::vector<std::vector<Vector3r>>> m_force;
	public:
		PotentialField() 
		{
		}

		PotentialField(Vector3r size,Vector3r resolution)
		{
			m_size = size;
			m_force.resize(resolution.x());
			for (int i = 0; i < resolution.x(); ++i)
			{
				m_force[i].resize(resolution.y());
				for (int j = 0; j < resolution.y(); ++j)
				{
					m_force[i][j].resize(resolution.z());
				}
			}
			
		}
		~PotentialField()
		{

		}

		std::vector<std::vector<std::vector<Vector3r>>> & getForce()
		{
			return m_force;
		}

		void setForceForOneGrid(Vector3r &val, Vector3r & position)
		{
			m_force[position[0]][position[1]][position[2]] = val;
		}

		Vector3r & getSize()
		{
			return m_size;
		}

		void setSize(Vector3r & size)
		{
			m_size = size;
		}


	};

	class gridProperty
	{
	public:
		struct velocity
		{
			Real Ui_1;
			Real Ui_2;
			Real Vj_1;
			Real Vj_2;
			Real Wk_1;
			Real Wk_2;
			velocity()
			{
				Ui_1 = 0.0;
				Ui_2 = 0.0;
				Vj_1 = 0.0;
				Vj_2 = 0.0;
				Wk_1 = 0.0;
				Wk_2 = 0.0;
			}
		} V;

		struct driving_force
		{
			Real Fu_1;
			Real Fu_2;
			Real Fv_1;
			Real Fv_2;
			Real Fw_1;
			Real Fw_2;
			driving_force()
			{
				Fu_1 = 0.0;
				Fu_2 = 0.0;
				Fv_1 = 0.0;
				Fv_2 = 0.0;
				Fw_1 = 0.0;
				Fw_2 = 0.0;
			}
		} F;

		Real density;
		int numOfGird;

		gridProperty() { density = 0.0;		numOfGird = 0; }
		gridProperty(Real den, int num) :density(den), numOfGird(num) {}
	};


}
#endif