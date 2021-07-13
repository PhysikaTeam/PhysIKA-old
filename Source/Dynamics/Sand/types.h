#pragma once

#include "vector_types.h"
#include "vector_functions.h"
#include <string>
#include <vector>
#define MAX_NUM 4
#define SAMPLING_DEPTH 4

typedef uchar4 rgb;

typedef float4 vertex;  // x, h, z

typedef float4 gridpoint;  // h, uh, vh, b

typedef int reflection;
using namespace std;
struct Samples
{
    float3 a[MAX_NUM];
    rgb    b[MAX_NUM];
    rgb    wrgb;
};
struct RendererParam
{
    float4 background;
    float3 cameraPos;
    float3 cameraUp;
    float  cameraYaw;
    float  cameraPitch;
    int    mode;
};
struct space
{
    float3 a;
    float3 b;
    int    x;
    int    y;
    int    z;
};
struct sphere
{
    space  spa;
    float3 a;
    float  b;
    bool   c;
};
struct shapefrom
{
    std::string filename;
    float3      trans;
};
struct addsdf_message
{
    bool        ex;
    std::string filename;
    float2      v;
    int         skip;
    int         num;
    //vector<int> shape;
    int               numsphere;
    vector<sphere>    sp;
    int               numfrom;
    vector<shapefrom> shfrom;
    void              qingk()
    {
        vector<sphere>    sp1;
        vector<shapefrom> shfrom1;
        swap(sp, sp1);
        swap(shfrom, shfrom1);
    }
};
struct m4_4
{
    float m[4][4];
};
struct obj_m
{
    bool           ex;
    int            num;
    vector<string> type;
    vector<string> filename;
    vector<rgb>    color;
    vector<float>  zoom_m;
    vector<m4_4>   rot_m;
    vector<m4_4>   tran_m;
    vector<float3> v;
    void           qingk()
    {
        vector<string> type1;
        vector<string> filename1;
        vector<rgb>    color1;
        vector<float>  zoom_m1;
        vector<m4_4>   rot_m1;
        vector<m4_4>   tran_m1;
        vector<float3> v1;
        swap(type, type1);
        swap(filename, filename1);
        swap(color, color1);
        swap(zoom_m, zoom_m1);
        swap(rot_m, rot_m1);
        swap(tran_m, tran_m1);
        swap(v, v1);
    }
};
