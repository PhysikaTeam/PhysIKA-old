#include <Eigen/Core>
#include "FEMGeometrySubdivid.h"
#include <vector>

#define SUBDIV_EPSILON 1e-5

const int HexDivTable[8][8] = {
    8,
    1,
    9,
    24,
    20,
    17,
    21,
    26,
    24,
    9,
    2,
    10,
    26,
    21,
    18,
    22,
    0,
    8,
    24,
    11,
    16,
    20,
    26,
    23,
    11,
    24,
    10,
    3,
    23,
    26,
    22,
    19,
    20,
    17,
    21,
    26,
    12,
    5,
    13,
    25,
    26,
    21,
    18,
    22,
    25,
    13,
    6,
    14,
    16,
    20,
    26,
    23,
    4,
    12,
    25,
    15,
    23,
    26,
    22,
    19,
    15,
    25,
    14,
    7
};

template <typename T>
void HexDiv1To8(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 27>& NewP, Eigen::Matrix<int, 8, 8>& C)
{
    /* add vertexes */
    int offset;
    /* all original vertexes */
    for (int i = 0; i < 8; i++)
        NewP.col(i) = P.col(i);
    offset = 8;
    /* midpoints of edges at first layer */
    for (int i = 0; i < 4; i++)
        NewP.col(offset + i) = 0.5 * (P.col(i) + P.col((i + 1) % 4));
    offset += 4;
    /* midpoints of edges at second layer */
    for (int i = 0; i < 4; i++)
        NewP.col(offset + i) = 0.5 * (P.col(i + 4) + P.col((i + 1) % 4 + 4));
    offset += 4;
    /* midpoints of lateral edges */
    for (int i = 0; i < 4; i++)
        NewP.col(offset + i) = 0.5 * (P.col(i) + P.col(i + 4));
    offset += 4;
    /* center of lateral faces */
    for (int i = 0; i < 4; i++)
        NewP.col(offset + i) = 0.25 * (P.col(i) + P.col((i + 1) % 4) + P.col(i + 4) + P.col((i + 1) % 4 + 4));
    offset += 4;
    /* center of top and bottom faces */
    NewP.col(offset++) = 0.25 * (P.col(0) + P.col(1) + P.col(2) + P.col(3));
    NewP.col(offset++) = 0.25 * (P.col(4) + P.col(5) + P.col(6) + P.col(7));
    /* center of cell */
    NewP.col(offset++) = 0.125 * (P.col(0) + P.col(1) + P.col(2) + P.col(3) + P.col(4) + P.col(5) + P.col(6) + P.col(7));

    /* add cells */
    for (int c = 0; c < 8; c++)
        for (int p = 0; p < 8; p++)
            C(p, c) = HexDivTable[c][p];
};

template void HexDiv1To8(const Eigen::Matrix<float, 3, 8>& P, Eigen::Matrix<float, 3, 27>& NewP, Eigen::Matrix<int, 8, 8>& C);
template void HexDiv1To8(const Eigen::Matrix<double, 3, 8>& P, Eigen::Matrix<double, 3, 27>& NewP, Eigen::Matrix<int, 8, 8>& C);

template <typename T>
inline int getPIdx(Eigen::Matrix<T, 3, 1>& p, std::vector<Eigen::Matrix<T, 3, 1>>& P)
{
    for (size_t i = 0; i < P.size(); i++)
    {
        Eigen::Matrix<T, 3, 1>& curP = P.at(i);
        if ((p - curP).squaredNorm() < SUBDIV_EPSILON)
            return i;
    }
    return -1;
}

template int getPIdx(Eigen::Matrix<float, 3, 1>& p, std::vector<Eigen::Matrix<float, 3, 1>>& P);
template int getPIdx(Eigen::Matrix<double, 3, 1>& p, std::vector<Eigen::Matrix<double, 3, 1>>& P);

template <typename T>
void HexDiv1To64(const Eigen::Matrix<T, 3, 8>& P, Eigen::Matrix<T, 3, 125>& NewP, Eigen::Matrix<int, 8, 64>& C)
{
    std::vector<Eigen::Matrix<T, 3, 1>> UniqueP;
    Eigen::Matrix<T, 3, 27>             TempP = Eigen::Matrix<T, 3, 27>::Zero();
    Eigen::Matrix<int, 8, 8>            TempC = Eigen::Matrix<int, 8, 8>::Zero();
    Eigen::Matrix<T, 3, 8>              SubCellP;
    Eigen::Matrix<T, 3, 27>             SubCellNewP;
    Eigen::Matrix<int, 8, 8>            SubC;

    int NewPIdx;
    int CountC = 0;

    /* subdivide big cell into 8 subcells */
    HexDiv1To8<T>(P, TempP, TempC);

    /* subdivide 8 subcells */
    for (int i = 0; i < 8; i++)
    {
        /* set subcell */
        for (int j = 0; j < 8; j++)
        {
            SubCellP.col(j) = TempP.col(TempC(j, i));
        }
        /* subdivide subcell */
        HexDiv1To8<T>(SubCellP, SubCellNewP, SubC);
        /* add subcell into cells matrix */
        for (int CIdx = 0; CIdx < 8; CIdx++)
        {
            for (int PIdx = 0; PIdx < 8; PIdx++)
            {
                Eigen::Matrix<T, 3, 1> p = SubCellNewP.col(SubC(PIdx, CIdx));
                /* determine whether the point is already in points vector */
                if ((NewPIdx = getPIdx(p, UniqueP)) == -1)
                {
                    UniqueP.push_back(p);
                    NewPIdx = UniqueP.size() - 1;
                }
                C(PIdx, CountC) = NewPIdx;
            }
            CountC++;
        }
    }
    for (int i = 0; i < 125; i++)
        NewP.col(i) = UniqueP.at(i);
}

template void HexDiv1To64(const Eigen::Matrix<float, 3, 8>& P, Eigen::Matrix<float, 3, 125>& NewP, Eigen::Matrix<int, 8, 64>& C);
template void HexDiv1To64(const Eigen::Matrix<double, 3, 8>& P, Eigen::Matrix<double, 3, 125>& NewP, Eigen::Matrix<int, 8, 64>& C);