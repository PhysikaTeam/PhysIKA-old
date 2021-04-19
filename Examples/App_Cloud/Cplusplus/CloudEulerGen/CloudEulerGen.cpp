#include "CloudEulerGen.h"
#include<random>

CloudEulerGen::CloudEulerGen(const std::vector<int>& arg, const float epsilon)
{
	mode_flag_ = arg.at(0);
	n_ = arg.at(1);
	start_x_ = arg.at(2);
	end_x_ = arg.at(3);
	start_y_ = arg.at(4);
	end_y_ = arg.at(5);
	noise_type_ = arg.at(6);
	epsilon_ = epsilon;

	//std::cout << mode_flag_ << " " << n_ << " " << start_x_ << " " << end_x_ << " " << start_y_ << " " << end_y_ << " " << noise_type_ << " " << epsilon_ << std::endl;
	//system("pause");

	init_vapor_density_ = INIT_VAPOR_DENSITY;
	init_material_temp_ = INIT_MATERIAL_TEMP;
	init_velocity_w_ = INIT_VELOCITY_W;

	size_= (n_ + 2) * (n_ + 2) * (n_ + 2);
	dt_ = 1.0f;
	frame_count_ = 0;

	velocity_u_.resize(size_);
	velocity_v_.resize(size_);
	velocity_w_.resize(size_);

	//烟
	if (mode_flag_ == 0)
	{
		vapor_density_.resize(size_);
		ambient_temp_.resize(n_ + 1);  //环境温度随高度变化，不随水平位置变化
		material_temp_.resize(size_);

		//初始化环境温度
		for (int k = 0; k < n_ + 1; k++)
		{
			ambient_temp_[k] = MAX_AMBIENT_TEMP + ((MAX_AMBIENT_TEMP - MIN_AMBIENT_TEMP) / n_ - 1) * (1 - k);
		}
	}

	//云
	if (mode_flag_ == 1)
	{
		vapor_density_.resize(size_);
		cloud_density_.resize(size_);
		ambient_temp_.resize(n_ + 1);   //环境温度随高度变化，不随水平位置变化
		material_temp_.resize(size_);

		//初始化环境温度
		for (int k = 0; k <= n_ + 1; k++)
		{
			ambient_temp_[k] = MAX_AMBIENT_TEMP + ((MAX_AMBIENT_TEMP - MIN_AMBIENT_TEMP) / n_ - 1) * (1 - k);
		}
	}
}

CloudEulerGen::~CloudEulerGen()
{
}

void CloudEulerGen::CloudGenFinalRun(int times)
{
	for (int i = 0; i < times; i++)
	{
		SourceControl();
		GetVelocity();
		GetDensity();
		CorrectTimestep();
		frame_count_++;
	}

}

void CloudEulerGen::CloudGenFrameRun()
{
	SourceControl();
	GetVelocity();
	GetDensity();
	CorrectTimestep();
	frame_count_++;
}

const int CloudEulerGen::GetModeFlag()
{
    return mode_flag_;
}

const int CloudEulerGen::GetN()
{
    return n_;
}

const float CloudEulerGen::GetVelocityU(int i, int j, int k)
{
	return velocity_u_[Position(i, j, k)];
}

const float CloudEulerGen::GetVelocityV(int i, int j, int k)
{
	return velocity_v_[Position(i, j, k)];
}

const float CloudEulerGen::GetVelocityW(int i, int j, int k)
{
	return velocity_w_[Position(i, j, k)];
}

const float CloudEulerGen::GetVaporDensity(int i, int j, int k)
{
	return vapor_density_[Position(i, j, k)];
}

const float CloudEulerGen::GetCloudDensity(int i, int j, int k)
{
	return cloud_density_[Position(i, j, k)];
}

const float CloudEulerGen::GetAmbientTemp(int k)
{
	return ambient_temp_[k];
}

const float CloudEulerGen::GetMaterialTemp(int i, int j, int k)
{
	return material_temp_[Position(i, j, k)];
}

const int CloudEulerGen::GetFrameCount()
{
	return frame_count_;
}

const int CloudEulerGen::Position(int i, int j, int k)
{
	return i + (n_ + 2) * j + (n_ + 2) * (n_ + 2) * k;
}

void CloudEulerGen::GetVelocity()
{
	std::vector<float> pre_velocity_u = velocity_u_;
	std::vector<float> pre_velocity_v = velocity_v_;
	std::vector<float> pre_velocity_w = velocity_w_;

	Advect(velocity_u_, pre_velocity_u, pre_velocity_u, pre_velocity_v, pre_velocity_w);
	Advect(velocity_v_, pre_velocity_v, pre_velocity_u, pre_velocity_v, pre_velocity_w);
	Advect(velocity_w_, pre_velocity_w, pre_velocity_u, pre_velocity_v, pre_velocity_w);

	Project();

	if (mode_flag_ == 0)
		AddBouyancySmoke();
	if (mode_flag_ == 1)
		AddBouyancyCloud();

	VorticityConfinement();

	Project();
}

void CloudEulerGen::GetDensity()
{
	int i, j, k;

	//烟
	if (mode_flag_ == 0)
	{
		float adiabatic_lapse_rate;   //绝热失效率
		adiabatic_lapse_rate = 0.02f;   //-----------
		std::vector<float> pre_vapor_density = vapor_density_;
		std::vector<float> pre_material_temp = material_temp_;

		Advect(vapor_density_, pre_vapor_density, velocity_u_, velocity_v_, velocity_w_);
		Advect(material_temp_, pre_material_temp, velocity_u_, velocity_v_, velocity_w_);

		for (i = 1; i <= n_; i++)
		{
			for (j = 1; j <= n_; j++)
			{
				for (k = 1; k <= n_; k++)
				{
					//物质温度减去绝热失效
					material_temp_[Position(i, j, k)] -= adiabatic_lapse_rate * dt_ * velocity_w_[Position(i, j, k)];
					//相对温度不得小于0！
					if (material_temp_[Position(i, j, k)] < MATERIAL_MIN)
						material_temp_[Position(i, j, k)] = 0.0f;
				}
			}
		}
	}

	//云
	if (mode_flag_ == 1)
	{
		std::vector<float> pre_vapor_density = vapor_density_;
		std::vector<float> pre_material_temp = material_temp_;
		std::vector<float> pre_cloud_density = cloud_density_;

		Advect(vapor_density_, pre_vapor_density, velocity_u_, velocity_v_, velocity_w_);
		Advect(material_temp_, pre_material_temp, velocity_u_, velocity_v_, velocity_w_);
		Advect(cloud_density_, pre_cloud_density, velocity_u_, velocity_v_, velocity_w_);

		PhaseTransitionCloud();
	}
}

void CloudEulerGen::BoundaryCondition(std::vector<float>& value, const int velocity_project_flag)
{
	int count_a, count_b;

	for (count_a = 1; count_a <= n_; count_a++)
	{
		for (count_b = 1; count_b <= n_; count_b++)
		{

			if (velocity_project_flag == 1)
			{
				value[Position(0, count_a, count_b)] = -value[Position(1, count_a, count_b)];
				value[Position(n_ + 1, count_a, count_b)] = -value[Position(n_, count_a, count_b)];
			}
			else
			{
				value[Position(0, count_a, count_b)] = value[Position(1, count_a, count_b)];
				value[Position(n_ + 1, count_a, count_b)] = value[Position(n_, count_a, count_b)];
			}

			if (velocity_project_flag == 2)
			{
				value[Position(count_a, 0, count_b)] = -value[Position(count_a, 1, count_b)];
				value[Position(count_a, n_ + 1, count_b)] = -value[Position(count_a, n_, count_b)];
			}
			else
			{
				value[Position(count_a, 0, count_b)] = value[Position(count_a, 1, count_b)];
				value[Position(count_a, n_ + 1, count_b)] = value[Position(count_a, n_, count_b)];
			}

			if (velocity_project_flag == 3)
			{
				//!!!!!!!
				//value[Position(count_a, count_b, 0)] = -value[Position(count_a, count_b, 1)];
				value[Position(count_a, count_b, n_ + 1)] = -value[Position(count_a, count_b, n_)];
			}
			else
			{
				value[Position(count_a, count_b, 0)] = value[Position(count_a, count_b, 1)];
				value[Position(count_a, count_b, n_ + 1)] = value[Position(count_a, count_b, n_)];
			}
		}
	}

	value[Position(0, 0, 0)] = 1.0 / 3.0 * (value[Position(1, 0, 0)] + value[Position(0, 1, 0)] + value[Position(0, 0, 1)]);
	value[Position(0, n_ + 1, 0)] = 1.0 / 3.0 * (value[Position(1, n_ + 1, 0)] + value[Position(0, n_, 0)] + value[Position(0, n_ + 1, 1)]);

	value[Position(n_ + 1, 0, 0)] = 1.0 / 3.0 * (value[Position(n_, 0, 0)] + value[Position(n_ + 1, 1, 0)] + value[Position(n_ + 1, 0, 1)]);
	value[Position(n_ + 1, n_ + 1, 0)] = 1.0 / 3.0 * (value[Position(n_, n_ + 1, 0)] + value[Position(n_ + 1, n_, 0)] + value[Position(n_ + 1, n_ + 1, 1)]);

	value[Position(0, 0, n_ + 1)] = 1.0 / 3.0 * (value[Position(1, 0, n_ + 1)] + value[Position(0, 1, n_ + 1)] + value[Position(0, 0, n_)]);
	value[Position(0, n_ + 1, n_ + 1)] = 1.0 / 3.0 * (value[Position(1, n_ + 1, n_ + 1)] + value[Position(0, n_, n_ + 1)] + value[Position(0, n_ + 1, n_)]);

	value[Position(n_ + 1, 0, n_ + 1)] = 1.0 / 3.0 * (value[Position(n_, 0, n_ + 1)] + value[Position(n_ + 1, 1, n_ + 1)] + value[Position(n_ + 1, 0, n_)]);
	value[Position(n_ + 1, n_ + 1, n_ + 1)] = 1.0 / 3.0 * (value[Position(n_, n_ + 1, n_ + 1)] + value[Position(n_ + 1, n_, n_ + 1)] + value[Position(n_ + 1, n_ + 1, n_)]);

}

void CloudEulerGen::Advect(std::vector<float>& value, const std::vector<float>& pre_value, const std::vector<float>& velocity_u, const std::vector<float>& velocity_v, const std::vector<float>& velocity_w)
{
	int i, j, k;
	int i0, j0, k0;   //相对于回溯点空间位置靠后的点的编号
	float a, b, c;   //相对于回溯点空间位置靠后的点在回朔点处的权值
	float pastX, pastY, pastZ;   //回溯点的坐标
	float diffX, diffY, diffZ;   //回溯点坐标与所在单元格底部距离

	//将原数据清空
	for (i = 0; i < size_; i++)
	{
		value[i] = 0.0f;
	}

	//回溯
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				pastX = i - dt_ * velocity_u[Position(i, j, k)];
				pastY = j - dt_ * velocity_v[Position(i, j, k)];
				pastZ = k - dt_ * velocity_w[Position(i, j, k)];

				if (pastX < 0.5)
				{
					pastX = 0.5f;
				}
				if (pastX > n_ + 0.5)
				{
					pastX = (float)(n_ + 0.5);
				}

				if (pastY < 0.5)
				{
					pastY = 0.5f;
				}
				if (pastY > n_ + 0.5)
				{
					pastY = (float)(n_ + 0.5);
				}

				//if (pastZ < 1.0)
				//{
				//	pastZ = 0.5f;
				//}
				if (pastZ < 0.5)
				{
					pastZ = 0.5f;
				}
				if (pastZ > n_ + 0.5)
				{
					pastZ = (float)(n_ + 0.5);
				}

				diffX = pastX - int(pastX);
				diffY = pastY - int(pastY);
				diffZ = pastZ - int(pastZ);

				i0 = int(pastX);
				j0 = int(pastY);
				k0 = int(pastZ);
				a = 1 - diffX;
				b = 1 - diffY;
				c = 1 - diffZ;

				value[Position(i, j, k)] = a * b * c * pre_value[Position(i0, j0, k0)] + a * b * (1 - c) * pre_value[Position(i0, j0, k0 + 1)] + a * (1 - b) * c * pre_value[Position(i0, j0 + 1, k0)] + a * (1 - b) * (1 - c) * pre_value[Position(i0, j0 + 1, k0 + 1)]
					+ (1 - a) * b * c * pre_value[Position(i0 + 1, j0, k0)] + (1 - a) * b * (1 - c) * pre_value[Position(i0 + 1, j0, k0 + 1)] + (1 - a) * (1 - b) * c * pre_value[Position(i0 + 1, j0 + 1, k0)] + (1 - a) * (1 - b) * (1 - c) * pre_value[Position(i0 + 1, j0 + 1, k0 + 1)];
			}
		}
	}

	//边界控制
	BoundaryCondition(value, 0);
}

void CloudEulerGen::GaussSeidelIteration(std::vector<float>& p, const std::vector<float>& div)
{
	int i, j, k;
	int count;   //迭代次数

	for (count = 0; count < 30; count++)
	{
		for (i = 1; i <= n_; i++)
		{
			for (j = 1; j <= n_; j++)
			{
				for (k = 1; k <= n_; k++)
				{
					p[Position(i, j, k)] = (p[Position(i - 1, j, k)] + p[Position(i + 1, j, k)] + p[Position(i, j - 1, k)] + p[Position(i, j + 1, k)] + p[Position(i, j, k - 1)] + p[Position(i, j, k + 1)] - div[Position(i, j, k)]) / 6.0f;
				}
			}
		}

		BoundaryCondition(p, 0);
	}
}

void CloudEulerGen::Project()
{
	int i, j, k;
	std::vector<float> p(size_);   //压强
	std::vector<float> div(size_);   //速度散度

	//计算每个网格点的速度散度
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				div[Position(i, j, k)] = float(1.0 / 2.0) * (velocity_u_[Position(i + 1, j, k)] - velocity_u_[Position(i - 1, j, k)] + velocity_v_[Position(i, j + 1, k)] - velocity_v_[Position(i, j - 1, k)] + velocity_w_[Position(i, j, k + 1)] - velocity_w_[Position(i, j, k - 1)]);
			}
		}
	}

	BoundaryCondition(div, 0);

	//计算压强
	GaussSeidelIteration(p, div);

	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				velocity_u_[Position(i, j, k)] -= 0.5f * (p[Position(i + 1, j, k)] - p[Position(i - 1, j, k)]);
				velocity_v_[Position(i, j, k)] -= 0.5f * (p[Position(i, j + 1, k)] - p[Position(i, j - 1, k)]);
				velocity_w_[Position(i, j, k)] -= 0.5f * (p[Position(i, j, k + 1)] - p[Position(i, j, k - 1)]);
			}
		}
	}

	BoundaryCondition(velocity_u_, 1);
	BoundaryCondition(velocity_v_, 2);
	BoundaryCondition(velocity_w_, 3);
}

void CloudEulerGen::AddBouyancySmoke()
{
	int i, j, k;
	float kb, g;   //kb：浮力系数 g：重力系数
	kb = 0.9f;
	//!!!!!!!!!!!!
	g = 0.0f;

	//kb = 0.4;
	//g = 0.4;

	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				if (vapor_density_[Position(i, j, k)] <= MATERIAL_MIN)
					continue;
				else
					velocity_w_[Position(i, j, k)] = velocity_w_[Position(i, j, k)] + dt_ * (kb * (material_temp_[Position(i, j, k)] / ambient_temp_[k]) - g * vapor_density_[Position(i, j, k)]);
			}
		}
	}

	BoundaryCondition(velocity_w_, 0);
}

void CloudEulerGen::AddBouyancyCloud()
{
	int i, j, k;
	float kb, g;   //kb：浮力系数 g：重力系数
	kb = 1.0f;
	g = 0.6f;

	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				if ((vapor_density_[Position(i, j, k)] <= MATERIAL_MIN) && (cloud_density_[Position(i, j, k)] <= MATERIAL_MIN))
					continue;
				else
					velocity_w_[Position(i, j, k)] = velocity_w_[Position(i, j, k)] + dt_ * (kb * (material_temp_[Position(i, j, k)] / ambient_temp_[k]) - g * cloud_density_[Position(i, j, k)]);
			}
		}
	}

	BoundaryCondition(velocity_w_, 0);
}

void CloudEulerGen::VorticityConfinement()
{
	int i, j, k;
	//中心速度
	std::vector<float> U(size_);
	std::vector<float> V(size_);
	std::vector<float> W(size_);

	//旋度场
	std::vector<float> rotation_field_U(size_);
	std::vector<float> rotation_field_V(size_);
	std::vector<float> rotation_field_W(size_);

	//旋度场强度
	std::vector<float> rf_strength(size_);

	//旋度场强度的梯度
	std::vector<float> rfs_gradient_U(size_);
	std::vector<float> rfs_gradient_V(size_);
	std::vector<float> rfs_gradient_W(size_);

	//旋度场强度的梯度的单位向量
	std::vector<float> rfsg_unit_U(size_);
	std::vector<float> rfsg_unit_V(size_);
	std::vector<float> rfsg_unit_W(size_);

	//旋度限制力
	std::vector<float> vc_U(size_);
	std::vector<float> vc_V(size_);
	std::vector<float> vc_W(size_);

	//求网格中心速度
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				U[Position(i, j, k)] = velocity_u_[Position(i, j, k)];
				V[Position(i, j, k)] = velocity_v_[Position(i, j, k)];
				W[Position(i, j, k)] = velocity_w_[Position(i, j, k)];
			}
		}
	}

	//求网格旋度
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				rotation_field_U[Position(i, j, k)] = 0.5f * ((W[Position(i, j + 1, k)] - W[Position(i, j - 1, k)]) - (V[Position(i, j, k + 1)] - V[Position(i, j, k - 1)]));
				rotation_field_V[Position(i, j, k)] = 0.5f * ((U[Position(i, j, k + 1)] - U[Position(i, j, k - 1)]) - (W[Position(i + 1, j, k)] - W[Position(i - 1, j, k)]));
				rotation_field_W[Position(i, j, k)] = 0.5f * ((V[Position(i + 1, j, k)] - V[Position(i - 1, j, k)]) - (U[Position(i, j + 1, k)] - U[Position(i, j - 1, k)]));
			}
		}
	}

	//求旋度场大小
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				rf_strength[Position(i, j, k)] = sqrtf(powf(rotation_field_U[Position(i, j, k)], 2.0) + powf(rotation_field_V[Position(i, j, k)], 2.0) + powf(rotation_field_W[Position(i, j, k)], 2.0));
			}
		}
	}

	//求旋度场强度的梯度
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				rfs_gradient_U[Position(i, j, k)] = 0.5f * (rf_strength[Position(i + 1, j, k)] - rf_strength[Position(i - 1, j, k)]);
				rfs_gradient_V[Position(i, j, k)] = 0.5f * (rf_strength[Position(i, j + 1, k)] - rf_strength[Position(i, j - 1, k)]);
				rfs_gradient_W[Position(i, j, k)] = 0.5f * (rf_strength[Position(i, j, k + 1)] - rf_strength[Position(i, j, k - 1)]);
			}
		}
	}

	//求旋度场强度的梯度的单位向量
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				float rfsg_unit_strength = sqrtf(powf(rfs_gradient_U[Position(i, j, k)], 2.0) + powf(rfs_gradient_V[Position(i, j, k)], 2.0) + powf(rfs_gradient_W[Position(i, j, k)], 2.0));
				if (rfsg_unit_strength < MATERIAL_MIN)
				{
					rfsg_unit_U[Position(i, j, k)] = 0.0f;
					rfsg_unit_V[Position(i, j, k)] = 0.0f;
					rfsg_unit_W[Position(i, j, k)] = 0.0f;
				}
				else
				{
					rfsg_unit_U[Position(i, j, k)] = rfs_gradient_U[Position(i, j, k)] / rfsg_unit_strength;
					rfsg_unit_V[Position(i, j, k)] = rfs_gradient_V[Position(i, j, k)] / rfsg_unit_strength;
					rfsg_unit_W[Position(i, j, k)] = rfs_gradient_W[Position(i, j, k)] / rfsg_unit_strength;
				}
			}
		}
	}

	//求旋度限制力
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				vc_U[Position(i, j, k)] = epsilon_ * 1.0f * (rfsg_unit_V[Position(i, j, k)] * rotation_field_W[Position(i, j, k)] - rfsg_unit_W[Position(i, j, k)] * rotation_field_V[Position(i, j, k)]);
				vc_V[Position(i, j, k)] = epsilon_ * 1.0f * (rfsg_unit_W[Position(i, j, k)] * rotation_field_U[Position(i, j, k)] - rfsg_unit_U[Position(i, j, k)] * rotation_field_W[Position(i, j, k)]);
				vc_W[Position(i, j, k)] = epsilon_ * 1.0f * (rfsg_unit_U[Position(i, j, k)] * rotation_field_V[Position(i, j, k)] - rfsg_unit_V[Position(i, j, k)] * rotation_field_U[Position(i, j, k)]);
			}
		}
	}

	//更新速度
	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				velocity_u_[Position(i, j, k)] += dt_ * vc_U[Position(i, j, k)];
				velocity_v_[Position(i, j, k)] += dt_ * vc_V[Position(i, j, k)];
				velocity_w_[Position(i, j, k)] += dt_ * vc_W[Position(i, j, k)];
			}
		}
	}
}

void CloudEulerGen::PhaseTransitionCloud()
{
	int i, j, k;
	float alpha;   //相变率
	float adiabatic_lapse_rate;   //绝热失效率
	float latent_heat_coefficient;   //潜热系数
	alpha = 0.8f;
	adiabatic_lapse_rate = 0.02f;
	latent_heat_coefficient = 0.8f;
	std::vector<float> phase_transition_cloud(size_);

	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				//计算当前网格的饱和水气值
				float temp = 100 * expf(-25.0f / (material_temp_[Position(i, j, k)] + ambient_temp_[k]));

				//正相变
				if (vapor_density_[Position(i, j, k)] >= temp)
				{
					phase_transition_cloud[Position(i, j, k)] = alpha * (vapor_density_[Position(i, j, k)] - temp);
					//更新水气值
					vapor_density_[Position(i, j, k)] -= (vapor_density_[Position(i, j, k)] - temp);
				}
				//反相变
				else
				{
					phase_transition_cloud[Position(i, j, k)] = -alpha * (temp - vapor_density_[Position(i, j, k)]);
					//如果云不够填补剩余水气值
					if ((cloud_density_[Position(i, j, k)] + phase_transition_cloud[Position(i, j, k)]) <= MATERIAL_MIN)
					{
						phase_transition_cloud[Position(i, j, k)] = -cloud_density_[Position(i, j, k)];
						//更新水气值
						vapor_density_[Position(i, j, k)] += (1.0f / alpha) * cloud_density_[Position(i, j, k)];
					}
					//如果云够填补剩余水气值(应该给空间内初始化一些水蒸气？)
					else
					{
						//更新水气值
						vapor_density_[Position(i, j, k)] = temp;
					}
				}

				//更新云密度
				cloud_density_[Position(i, j, k)] += phase_transition_cloud[Position(i, j, k)];

				//更新物质温度
				temp = material_temp_[Position(i, j, k)] - adiabatic_lapse_rate * velocity_w_[Position(i, j, k)] + latent_heat_coefficient * phase_transition_cloud[Position(i, j, k)];
				if (temp <= MATERIAL_MIN) {
					//相对温度不得小于0！
					material_temp_[Position(i, j, k)] = 0.0f;
				}
				else
				{
					material_temp_[Position(i, j, k)] = temp;
				}
			}
		}
	}

	BoundaryCondition(vapor_density_, 0);
	BoundaryCondition(cloud_density_, 0);
	BoundaryCondition(material_temp_, 0);
}

void CloudEulerGen::CorrectTimestep()
{
	int i, j, k;
	float current_velocity = 0.0f;   //当前网格速度
	float current_dt = dt_;   //当前网格纠正后时间步长
	float min_dt = dt_;  //当前最小时间步长

	for (i = 1; i <= n_; i++)
	{
		for (j = 1; j <= n_; j++)
		{
			for (k = 1; k <= n_; k++)
			{
				current_velocity = sqrtf(powf(velocity_u_[Position(i, j, k)], 2.0) + powf(velocity_v_[Position(i, j, k)], 2.0) + powf(velocity_w_[Position(i, j, k)], 2.0));
				//1.2为阈值控制时间步长减少程度
				current_dt = 1.0f / (current_velocity * 1.2);
				if (min_dt > current_dt)
					min_dt = current_dt;
			}
		}
	}

	//更新时间步长
	dt_ = min_dt;
	//std:: cout << "下一次迭代时间步长： " << dt_ << std::endl;
}

void CloudEulerGen::SourceControl()
{
	switch (noise_type_)
	{
	case 0:
		NoneNoise();
		break;
	case 1:
		GaussNoise();
		break;
	default:
		NoneNoise();
		break;
	}
}

void CloudEulerGen::NoneNoise()
{
	for (int i = start_x_; i <= end_x_; i++)
	{
		for (int j = start_y_; j <= end_y_; j++)
		{
			vapor_density_[Position(i, j, 0)] = init_vapor_density_;
			material_temp_[Position(i, j, 0)] = init_material_temp_;
			velocity_w_[Position(i, j, 0)] = init_velocity_w_;
			velocity_w_[Position(i, j, 1)] = init_velocity_w_;
		}
	}
}

void CloudEulerGen::GaussNoise()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> normal(0.0f, 1.0f);

	for (int i = start_x_; i <= end_x_; i++)
	{
		for (int j = start_y_; j <= end_y_; j++)
		{
			float normal_value = normal(rd);
			if (normal_value > MATERIAL_MIN)
			{
				vapor_density_[Position(i, j, 0)] = init_vapor_density_;
				material_temp_[Position(i, j, 0)] = init_material_temp_;
				velocity_w_[Position(i, j, 0)] = init_velocity_w_;
				velocity_w_[Position(i, j, 1)] = init_velocity_w_;
			}
			else
			{
				vapor_density_[Position(i, j, 0)] = 0.0f;
				material_temp_[Position(i, j, 0)] = 0.0f;
				velocity_w_[Position(i, j, 0)] = 0.0f;
				velocity_w_[Position(i, j, 1)] = 0.0f;
			}
		}
	}
}
