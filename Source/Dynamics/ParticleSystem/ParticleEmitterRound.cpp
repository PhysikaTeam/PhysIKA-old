
/**
 * @author     : Chang Yue (changyue@buaa.edu.cn)
 * @date       : 2020-08-27
 * @description: Implementation of ParticleEmitterRound class, which emits particles from a circle
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: poslish code
 * @version    : 1.1
 */

#include "ParticleEmitterRound.h"

#include <ctime>
#include <cstdlib>

namespace PhysIKA {
IMPLEMENT_CLASS_1(ParticleEmitterRound, TDataType)

template <typename TDataType>
ParticleEmitterRound<TDataType>::ParticleEmitterRound(std::string name)
    : ParticleEmitter<TDataType>(name)
{
    srand(time(0));
}

template <typename TDataType>
ParticleEmitterRound<TDataType>::~ParticleEmitterRound()
{
    gen_pos.release();
}

template <typename TDataType>
void ParticleEmitterRound<TDataType>::generateParticles()
{
    auto sampling_distance = this->varSamplingDistance()->getValue();
    if (sampling_distance < EPSILON)
        sampling_distance = 0.005;
    auto center = this->varLocation()->getValue();

    std::vector<Coord> pos_list;
    std::vector<Coord> vel_list;

    auto rot_vec = this->varRotation()->getValue();

    Quaternion<Real> quat  = Quaternion<float>::Identity();
    float            x_rad = rot_vec[0] / 180.0f * M_PI;
    float            y_rad = rot_vec[1] / 180.0f * M_PI;
    float            z_rad = rot_vec[2] / 180.0f * M_PI;

    quat = quat * Quaternion<Real>(x_rad, Coord(1, 0, 0));
    quat = quat * Quaternion<Real>(y_rad, Coord(0, 1, 0));
    quat = quat * Quaternion<Real>(z_rad, Coord(0, 0, 1));

    auto rot_mat = quat.get3x3Matrix();

    Coord v0 = this->varVelocityMagnitude()->getValue() * rot_mat * Vector3f(0, -1, 0);

    auto r  = this->varRadius()->getValue();
    Real lo = -r;
    Real hi = r;

    for (Real x = lo; x <= hi; x += sampling_distance)
    {
        for (Real y = lo; y <= hi; y += sampling_distance)
        {
            Coord p = Coord(x, 0, y);
            if ((p - Coord(0)).norm() < r && rand() % 40 == 0)
            {
                Coord q = rot_mat * p;
                pos_list.push_back(q + center);
                vel_list.push_back(v0);
            }
        }
    }

    if (pos_list.size() > 0)
    {
        gen_pos.resize(pos_list.size());
        gen_vel.resize(pos_list.size());

        Function1Pt::copy(gen_pos, pos_list);
        Function1Pt::copy(gen_vel, vel_list);
    }

    pos_list.clear();
    vel_list.clear();
}

}  // namespace PhysIKA