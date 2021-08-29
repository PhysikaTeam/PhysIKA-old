/**
 * @author     : Chen Xiaosong (xiaosong0911@gmail.com)
 * @date       : 2019-06-13
 * @description: Declaration of MultipleFluidModel class, which implements the paper
 *               <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-27
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Framework/Framework/NumericalModel.h"
#include "CahnHilliard.h"

namespace PhysIKA {
template <typename TDataType>
class ParticleIntegrator;
template <typename TDataType>
class NeighborQuery;
template <typename TDataType>
class DensityPBD;
template <typename TDataType>
class ImplicitViscosity;

/**
 * MultipleFluidModel, implementation of the paper <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * Usage:
 * 1. Define a MultipleFluidModel instance
 * 2. Bind the instance with a ParticleFluid node by calling Node::setNumericalModel()
 * 3. Connect fields of ParticleFluid with MultipleFluidModel by calling Field::connect()
 * We're done. MultipleFluidModel will be employed in advance() of the ParticleFluid.
 *
 * TODO(Zhu Fei): complete the code comments.
 */
template <typename TDataType>
class MultipleFluidModel : public NumericalModel
{
    DECLARE_CLASS_1(MultipleFluidModel, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    using PhaseVector = typename CahnHilliard<TDataType>::PhaseVector;

    MultipleFluidModel();
    ~MultipleFluidModel() override;

    void step(Real dt) override;

    void setSmoothingLength(Real len)
    {
        m_smoothingLength.setValue(len);
    }
    void setRestDensity(PhaseVector rho)
    {
        m_restDensity = rho;
    }

public:
    VarField<Real>        m_smoothingLength;
    VarField<PhaseVector> m_restDensity;

    DeviceArrayField<Coord>       m_position;
    DeviceArrayField<Vector3f>    m_color;
    DeviceArrayField<Coord>       m_velocity;
    DeviceArrayField<Real>        m_massInv;  // for pbd constraints
    DeviceArrayField<PhaseVector> m_concentration;

    DeviceArrayField<Coord> m_forceDensity;

protected:
    bool initializeImpl() override;

private:
    std::shared_ptr<CahnHilliard<TDataType>>       m_phaseSolver;
    std::shared_ptr<DensityPBD<TDataType>>         m_pbdModule;
    std::shared_ptr<ImplicitViscosity<TDataType>>  m_visModule;
    std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
    std::shared_ptr<NeighborQuery<TDataType>>      m_nbrQuery;
};

#ifdef PRECISION_FLOAT
template class MultipleFluidModel<DataType3f>;
#else
template class MultipleFluidModel<DataType3d>;
#endif
}  // namespace PhysIKA
