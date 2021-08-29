/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of SurfaceTension class, which implements the surface tension force
 *               introduced in the paper <Robust Simulation of Sparsely Sampled Thin Features in SPH-Based Free Surface Flows>
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/ModuleForce.h"
#include "Framework/Framework/FieldArray.h"

namespace PhysIKA {

template <typename TDataType>
class SurfaceTension : public ForceModule
{
    /**
 * SurfaceTension, applying the surface tension force
 * Usage:
 * 1. Define a SurfaceTension instance
 * 2. Initialize the velocity field and position field, seems to have some problems here
 * 3. Call execute() when needed
 */
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SurfaceTension();
    ~SurfaceTension() override{};

    /**
    *   seems to be incompleted on current version
    */
    bool execute() override;
    /**
    *   seems to be incompleted on current version
    */
    bool applyForce() override;

    /**
    * seems to be used to initialize position field
    */
    void setPositionID(FieldID id)
    {
        m_posID = id;
    }

    /**
    * seems to be used to initialize velocity field
    */
    void setVelocityID(FieldID id)
    {
        m_velID = id;
    }

    /**
    * seems to be used to initialize neighbor list
    */
    void setNeighborhoodID(FieldID id)
    {
        m_neighborhoodID = id;
    }

    /**
    * seems to be used to initialize coef(?)
    */
    void setIntensity(Real intensity)
    {
        m_intensity = intensity;
    }

    /**
    * seems to be used to initialize searching radius
    */
    void setSmoothingLength(Real len)
    {
        m_soothingLength = len;
    }

protected:
    FieldID m_posID;
    FieldID m_velID;
    FieldID m_neighborhoodID;

private:
    Real m_intensity;
    Real m_soothingLength;

    DeviceArrayField<Real>* m_energy;
};

#ifdef PRECISION_FLOAT
template class SurfaceTension<DataType3f>;
#else
template class SurfaceTension<DataType3d>;
#endif
}  // namespace PhysIKA