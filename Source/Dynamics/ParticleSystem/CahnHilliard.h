/**
 * @author     : Chen Xiaosong (xiaosong0911@gmail.com)
 * @date       : 2019-06-02
 * @description: Declaration of CahnHilliard class, which implements the CahnHilliard model
 *               introduced in the paper <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-27
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Framework/Framework/Module.h"
namespace PhysIKA {
/**
 * CahnHilliard implements the CahnHilliard model of the paper
 * <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * It is used in MultipleFluidModel class
 * TODO(Zhu Fei): complete the code comments.
 */
template <typename TDataType, int PhaseCount = 2>
class CahnHilliard : public Module
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    using PhaseVector = Vector<Real, PhaseCount>;

    CahnHilliard();
    ~CahnHilliard() override;

    /**
     * update states, generally called in simulation advance() calls
     *
     * @return  true if succeeds, false otherwise
     */
    bool integrate();

protected:
    bool initializeImpl() override;

public:
    VarField<Real> m_particleVolume;
    VarField<Real> m_smoothingLength;

    VarField<Real> m_degenerateMobilityM;
    VarField<Real> m_interfaceEpsilon;

    DeviceArrayField<Coord> m_position;

    NeighborField<int> m_neighborhood;

    DeviceArrayField<PhaseVector> m_chemicalPotential;
    DeviceArrayField<PhaseVector> m_concentration;
};
#ifdef PRECISION_FLOAT
template class CahnHilliard<DataType3f>;
#else
template class CahnHilliard<DataType3d>;
#endif
}  // namespace PhysIKA
