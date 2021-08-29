#include "geometry.h"

inline HDFUNC float eigenMax(cmat3 mat)
{
    cfloat3 guess(1, 1, 1);
    cfloat3 tmp;
    float   s;
    for (int i = 0; i < 5; i++)
    {
        mvprod(mat, guess, tmp);
        s = tmp.z;
        if (abs(s) < EPSILON)
            return -1;
        tmp   = tmp / s;
        guess = tmp;
    }
    mvprod(mat, guess, tmp);
    auto rq = dot(guess, tmp) / dot(guess, guess);
    return rq;
}