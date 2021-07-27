/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of Attribute class, a mechanism to tag properties
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-26
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include <cstdint>
#include "Core/Platform.h"
namespace PhysIKA {
/*!
    *    \class    Attribute
    *    \brief    particle attribute 0x00000000: [31-30]material; [29]motion; [28]Dynamic; [27-8]undefined yet, for future use; [7-0]correspondding to the id of a fluid phase in multiphase fluid or an object in a multibody system
    */

/**
 * Attribute, a 32-bit tag to store multiple properties
 * TODO(Zhu Fei): clarify the bit fields
 */
class Attribute
{
public:
    COMM_FUNC Attribute()
    {
        m_tag = 0;
    }
    COMM_FUNC ~Attribute(){};

    enum MaterialType
    {
        MATERIAL_MASK    = 0xC0000000,
        MATERIAL_FLUID   = 0x00000000,
        MATERIAL_RIGID   = 0xA0000000,
        MATERIAL_ELASTIC = 0xB0000000,
        MATERIAL_PLASTIC = 0xC0000000
    };

    enum KinematicType
    {
        KINEMATIC_MASK     = 0x30000000,
        KINEMATIC_FIXED    = 0x00000000,
        KINEMATIC_PASSIVE  = 0x10000000,
        KINEMATIC_POSITIVE = 0x20000000
    };

    enum ObjectID
    {
        OBJECTID_MASK = 0x000000FF
    };

    COMM_FUNC inline void SetMaterialType(MaterialType type)
    {
        m_tag = ((~MATERIAL_MASK) & m_tag) | type;
    }
    COMM_FUNC inline void SetKinematicType(KinematicType type)
    {
        m_tag = ((~KINEMATIC_MASK) & m_tag) | type;
    }
    COMM_FUNC inline void SetObjectId(unsigned id)
    {
        m_tag = ((~OBJECTID_MASK) & m_tag) | id;
    }

    COMM_FUNC inline MaterialType GetMaterialType()
    {
        return MaterialType(m_tag & MATERIAL_MASK);
    }
    COMM_FUNC inline KinematicType GetKinematicType()
    {
        return KinematicType(m_tag & KINEMATIC_MASK);
    }
    COMM_FUNC inline unsigned GetObjectId()
    {
        return ( unsigned )(m_tag & OBJECTID_MASK);
    }

    COMM_FUNC inline bool IsFluid()
    {
        return MaterialType::MATERIAL_FLUID == GetMaterialType();
    }
    COMM_FUNC inline bool IsRigid()
    {
        return MaterialType::MATERIAL_RIGID == GetMaterialType();
    }
    COMM_FUNC inline bool IsElastic()
    {
        return MaterialType::MATERIAL_ELASTIC == GetMaterialType();
    }
    COMM_FUNC inline bool IsPlastic()
    {
        return MaterialType::MATERIAL_PLASTIC == GetMaterialType();
    }

    COMM_FUNC inline void SetFluid()
    {
        SetMaterialType(MaterialType::MATERIAL_FLUID);
    }
    COMM_FUNC inline void SetRigid()
    {
        SetMaterialType(MaterialType::MATERIAL_RIGID);
    }
    COMM_FUNC inline void SetElastic()
    {
        SetMaterialType(MaterialType::MATERIAL_ELASTIC);
    }
    COMM_FUNC inline void SetPlastic()
    {
        SetMaterialType(MaterialType::MATERIAL_PLASTIC);
    }

    COMM_FUNC inline bool IsFixed()
    {
        return KinematicType::KINEMATIC_FIXED == GetKinematicType();
    }
    COMM_FUNC inline bool IsPassive()
    {
        return KinematicType::KINEMATIC_PASSIVE == GetKinematicType();
    }
    COMM_FUNC inline bool IsDynamic()
    {
        return KinematicType::KINEMATIC_POSITIVE == GetKinematicType();
    }

    COMM_FUNC inline void SetFixed()
    {
        SetKinematicType(KinematicType::KINEMATIC_FIXED);
    }
    COMM_FUNC inline void SetPassive()
    {
        SetKinematicType(KinematicType::KINEMATIC_PASSIVE);
    }
    COMM_FUNC inline void SetDynamic()
    {
        SetKinematicType(KinematicType::KINEMATIC_POSITIVE);
    }

private:
    uint32_t m_tag;
};
}  // namespace PhysIKA
