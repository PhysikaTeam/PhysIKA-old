#include "smesh.h"
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;

namespace PhysIKA {

void Smesh::loadFile(string filename)
{
    fstream filein(filename);
    if (!filein.is_open())
    {
        cout << "can't open smesh file:" << filename << endl;
        exit(0);
    }

    string part_str;
    filein >> part_str;
    if (part_str != "*VERTICES")
    {
        cout << "first non-empty line must be '*VERTICES'." << endl;
        exit(0);
    }
    int num_points = 0, point_dim = 0;
    filein >> num_points >> point_dim;
    m_points.resize(num_points, Vector3f(0.0f));
    for (int i = 0; i < num_points; ++i)
    {
        int vert_index;
        filein >> vert_index;
        for (int j = 0; j < point_dim; ++j)
        {
            filein >> m_points[i][j];
        }
    }

    filein >> part_str;
    if (part_str != "*ELEMENTS")
    {
        cout << "after vertices, the first non-empty line must be '*ELEMENTS'." << endl;
        return;
    }

    while (!filein.eof())
    {
        string ele_type = "";
        int    num_eles = 0, ele_dim = 0;
        filein >> ele_type >> num_eles >> ele_dim;
        if (ele_type == "LINE")
        {
            m_edges.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_edges[i][j];
                }
            }
        }
        else if (ele_type == "TRIANGLE")
        {
            m_triangles.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_triangles[i][j];
                }
            }
        }
        else if (ele_type == "QUAD")
        {
            m_quads.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_quads[i][j];
                }
            }
        }
        else if (ele_type == "TET")
        {
            m_tets.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_tets[i][j];
                }
            }
        }
        else if (ele_type == "HEX")
        {
            m_hexs.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_hexs[i][j];
                }
            }
        }
        else
        {
            cout << "unrecognized element type:" << ele_type << endl;
        }
    }
}

}  // namespace PhysIKA
