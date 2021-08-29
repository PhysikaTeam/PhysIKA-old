/**
 * @author     : Ben Xu (xuben@mail.nankai.edu.cn)
 * @date       : 2021-02-01
 * @description: swe model library
 * @version    : 1.0
 */
#pragma once
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"

namespace PhysIKA {
/*!
*    \class    ShallowWaterEquationModel
*    \brief    SWE model. This class can only be called by class HeightFieldNode's member function.
    */
template <typename TDataType>
class ShallowWaterEquationModel : public NumericalModel
{
    DECLARE_CLASS_1(ShallowWaterEquationModel, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ShallowWaterEquationModel();
    virtual ~ShallowWaterEquationModel();

    void step(Real dt) override;

    void setDistance(Real distance)
    {
        this->distance = distance;
    }
    void setRelax(Real relax)
    {
        this->relax = relax;
    }
    void setZcount(int zcount)
    {
        this->zcount = zcount;
    }

public:
    DeviceArrayField<Coord> m_position;
    DeviceArrayField<Coord> m_velocity;
    DeviceArrayField<Coord> m_accel;
    //staggered grid
    DeviceArrayField<Real> grid_vel_x;
    DeviceArrayField<Real> grid_vel_z;
    DeviceArrayField<Real> grid_accel_x;
    DeviceArrayField<Real> grid_accel_z;

    DeviceArrayField<Real>  m_solid;
    DeviceArrayField<Coord> m_normal;
    DeviceArrayField<int>   m_isBound;
    DeviceArrayField<Real>  m_height;  //solid_pos + h*Normal = m_position
    DeviceArrayField<Coord> buffer;

protected:
    bool initializeImpl() override;

private:
    int  m_pNum;    //grid point numbers
    Real distance;  //the distance between two neighbor grid points
    Real relax;     //System damping coefficient, should always be less than or equal to 1
    int  zcount;    //grid point number along z direction
    int  xcount;    //grid point number along x direction

    float sumtimes = 0;
    int   sumnum   = 0;
};

#ifdef PRECISION_FLOAT
template class ShallowWaterEquationModel<DataType3f>;
#else
template class ShallowWaterEquationModel<DataType3d>;
#endif
};  // namespace PhysIKA