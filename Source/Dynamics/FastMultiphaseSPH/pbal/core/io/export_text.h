#pragma once

#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

namespace pbal {

namespace io {

template <typename T>
void exportParticlesAsText(
    const std::vector<std::vector<T>> particles,
    const std::string&                rootDir,
    int                               frameCnt)
{

    char basename[256];
    snprintf(basename, sizeof(basename), "%d.txt", frameCnt);
    std::string   filename = rootDir + "/" + basename;
    std::ofstream file(filename.c_str());
    if (file)
    {
        printf("Writing %s...\n", filename.c_str());
        file << std::fixed << std::showpoint << std::setprecision(6);
        for (auto p : particles)
        {
            for (size_t i = 0; i < p.size(); i++)
            {
                if (i + 1 == p.size())
                    file << p[i] << std::endl;
                else
                    file << p[i] << ' ';
            }
        }

        file.close();
    }
}

}  // namespace io
}  // namespace pbal
