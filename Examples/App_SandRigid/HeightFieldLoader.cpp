#include "HeightFieldLoader.h"

#include "IO/Image_IO/image_io.h"

namespace PhysIKA {
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
    HeightFieldGrid<T, TReal, deviceType> midhf;
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