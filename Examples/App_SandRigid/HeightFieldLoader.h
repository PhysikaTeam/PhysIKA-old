#pragma once

#ifndef PK_HEIGHTFIELDLOADER_H
#define PK_HEIGHTFIELDLOADER_H

//#include "IO/Image_IO/image_io.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include <string>

namespace PhysIKA {

class HeightFieldLoader
{
public:
    void setRange(float minh, float maxh)
    {
        m_minH = minh;
        m_maxH = maxh;

        if (m_minH > m_maxH)
        {
            float temp = m_minH;
            m_minH     = m_maxH;
            m_maxH     = temp;
        }
    }

    template <typename TReal, DeviceType deviceType>
    bool load(HeightFieldGrid<TReal, TReal, deviceType>& hf, const std::string& img);

private:
    float m_minH = 0.0;
    float m_maxH = 1.0;
};

}  // namespace PhysIKA

#endif  // PK_HEIGHTFIELDLOADER_H
