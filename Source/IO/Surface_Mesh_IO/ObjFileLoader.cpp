#include "ObjFileLoader.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace PhysIKA {

ObjFileLoader::ObjFileLoader(std::string filename)
{
    load(filename);
}

bool ObjFileLoader::load(const std::string& filename)
{
    if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj"))
    {
        std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
        exit(-1);
    }

    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Failed to open. Terminating.\n";
        exit(-1);
    }

    int         ignored_lines = 0;
    std::string line;
    while (!infile.eof())
    {
        std::getline(infile, line);

        //.obj files sometimes contain vertex normals indicated by "vn"
        if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn"))
        {
            std::stringstream data(line);
            char              c;
            Vector3f          point;
            data >> c >> point[0] >> point[1] >> point[2];
            vertList.push_back(point);
        }
        else if (line.substr(0, 1) == std::string("f"))
        {
            std::stringstream data(line);
            char              c;
            int               v0, v1, v2;
            data >> c >> v0 >> v1 >> v2;
            faceList.push_back(Face(v0 - 1, v1 - 1, v2 - 1));
        }
        //else if (line.substr(0, 2) == std::string("vn")) {
        //    std::cerr << "Obj-loader is not able to parse vertex normals, please strip them from the input file. \n";
        //    exit(-2);
        //}
        else
        {
            ++ignored_lines;
        }
    }
    infile.close();
}

bool ObjFileLoader::save(const std::string& filename)
{
    return true;
}

std::vector<Vector3f>& ObjFileLoader::getVertexList()
{
    return vertList;
}

std::vector<Face>& ObjFileLoader::getFaceList()
{
    return faceList;
}

}  //end of namespace PhysIKA
