#pragma once

#ifndef PK_HEIGHTFIELDLOADER_H
#define PK_HEIGHTFIELDLOADER_H

//#include "IO/Image_IO/image_io.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "IO/Image_IO/image_io.h"
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

template <typename TReal, DeviceType deviceType>
bool HeightFieldLoader::load(HeightFieldGrid<TReal, TReal, deviceType>& hf, const std::string& img)
{
    // Load image.
    Image imgdata;
    if (!ImageIO::load(img, &imgdata))
        return false;

    int stride = imgdata.dataFormat() == Image::DataFormat::RGB ? 3 : 4;

    // Save the image in a temp height field.
    // Heightfield size = (width, height),   space = (1,1)
    HeightFieldGrid<TReal, TReal, deviceType> midhf;
    midhf.resize(imgdata.width(), imgdata.height());
    midhf.setSpace(1.0, 1.0);
    midhf.setOrigin(imgdata.width() / 2.0, 0.0, imgdata.height() / 2.0);
    for (int i = 0; i < imgdata.width(); ++i)
    {
        for (int j = 0; j < imgdata.height(); ++j)
        {
            const unsigned char* curdata = imgdata.rawData() + (j * imgdata.width() + i) * stride;
            double               val     = (( int )(curdata[0]) + ( int )(curdata[1]) + ( int )(curdata[2])) / (3.0 * 255.0);
            val                          = m_minH + (m_maxH - m_minH) * val;

            midhf(i, j) = (TReal)(val);
        }
    }

    // Resize height field data, and save it in hf.
    if (hf.Nx() <= 0 || hf.Ny() <= 0)
        hf.resize(imgdata.width(), imgdata.height());
    for (int i = 0; i < hf.Nx(); ++i)
    {
        for (int j = 0; j < hf.Ny(); ++j)
        {
            double gx = i * ( double )midhf.Nx() / hf.Nx();
            double gz = j * ( double )midhf.Ny() / hf.Ny();

            TReal val = midhf.get(gx, gz);
            hf(i, j)  = val;
        }
    }

    return true;
}

}  // namespace PhysIKA

#endif  // PK_HEIGHTFIELDLOADER_H
