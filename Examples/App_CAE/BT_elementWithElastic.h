#include "GaussIntegration.h"
int BT_elementWithElastic(int** gix, double** gx, double** gv, double** gtk, double dt, int icyc, int mg, int llkt, double qhg, double** gff, double** gfm)
{
#define lg 64
#define LLKT 3

    //define local variables
    int illt;
    int inft;

    int    in[lg];
    int    imat[lg];
    int    ix1[lg], ix2[lg], ix3[lg], ix4[lg];  //node number of element
    double dx1[lg], dx2[lg], dx3[lg], dx4[lg];  //x displacement increment of node
    double dy1[lg], dy2[lg], dy3[lg], dy4[lg];
    double dz1[lg], dz2[lg], dz3[lg], dz4[lg];
    double wxx1[lg], wxx2[lg], wxx3[lg], wxx4[lg];  //x rotation increment of node
    double wyy1[lg], wyy2[lg], wyy3[lg], wyy4[lg];
    double wzz1[lg], wzz2[lg], wzz3[lg], wzz4[lg];

    //     double cn1x1[lg],cn1y1[lg],cn1z1[lg];        //current coordinates at n+1
    //     double cn1x2[lg],cn1y2[lg],cn1z2[lg];
    //     double cn1x3[lg],cn1y3[lg],cn1z3[lg];
    //     double cn1x4[lg],cn1y4[lg],cn1z4[lg];

    double cx1[lg], cy1[lg], cz1[lg];  //coordinates at n+1/2
    double cx2[lg], cy2[lg], cz2[lg];
    double cx3[lg], cy3[lg], cz3[lg];
    double cx4[lg], cy4[lg], cz4[lg];

    //     double xh21[lg],yh21[lg],zh21[lg];        //vector difference of nodal coordinates at n+1/2
    //     double xh31[lg],yh31[lg],zh31[lg];
    //     double xh41[lg],yh41[lg],zh41[lg];
    //     double xh42[lg],yh42[lg],zh42[lg];

    double c1, c2, c3, al;
    double xhh;
    double gh13[lg], gh23[lg], gh33[lg];  //e3(n+1/2),unit normal vector to lamina surface at n+1/2
    double gh11[lg], gh21[lg], gh31[lg];  //e1(n+1/2)
    double gh12[lg], gh22[lg], gh32[lg];  //e2(n+1/2)

    double x21[lg], y21[lg], z21[lg];  //vector difference of nodal coordinates at n+1
    double x31[lg], y31[lg], z31[lg];
    double x41[lg], y41[lg], z41[lg];
    double x42[lg], y42[lg], z42[lg];

    double xll;
    double gl13[lg], gl23[lg], gl33[lg];  //e3(n+1),unit vectors tangent to lamina surface at n+1
    double gl12[lg], gl22[lg], gl32[lg];  //e2(n+1)
    double gl11[lg], gl21[lg], gl31[lg];  //e1(n+1)

    double vx1[lg], vy1[lg], vz1[lg];  //translation increments of node1 in local system
    double vx2[lg], vy2[lg], vz2[lg];
    double vx3[lg], vy3[lg], vz3[lg];
    double vx4[lg], vy4[lg], vz4[lg];

    double vx5[lg], vy5[lg];  //rotation increments of node1 in local system
    double vx6[lg], vy6[lg];
    double vx7[lg], vy7[lg];
    double vx8[lg], vy8[lg];

    double x2[lg], y2[lg], z2[lg];  //node2 coordinates in local coordinate system
    double x3[lg], y3[lg], z3[lg];
    double x4[lg], y4[lg], z4[lg];

    double px1[lg], px2[lg], py1[lg], py2[lg];
    double area[lg];                      //the area of the element
    double vx13[lg], vy13[lg], vz13[lg];  //difference in displacements in local coordinate system
    double vx24[lg], vy24[lg], vz24[lg];
    double wxx13[lg], wyy13[lg];  //difference in rotations in local coordinate system
    double wxx24[lg], wyy24[lg];

    double areain[lg];  //the reciprocan of the element area
    double px1a[lg], py1a[lg], px2a[lg], py2a[lg];
    double g1[lg], g2[lg], g3[lg];

    double b1vx[lg], b1vy[lg], b1vz[lg];
    double b2vx[lg], b2vy[lg], b2vz[lg];
    double b1tx[lg], b1ty[lg];
    double b2tx[lg], b2ty[lg];
    double b3tx[lg], b3ty[lg];
    double bxyv[lg], bxyt[lg];

    double sx[lg], sy[lg];      //for compute the incremental shear strain
    double epyz[lg], epzx[lg];  //incremental shear strain yz and zx

    double htx[lg], hty[lg];
    double gm1[lg], gm2[lg], gm3[lg], gm4[lg];  //define gamma
    double gmz[lg];

    double sg1m, sg2m, sg4m;  //integrate resultant moments
    double sg1n, sg2n, sg4n;  //integrate resultant forces in plane
    double sg5n, sg6n;        //integrate resultant shear forces
    double sg5l, sg6l;

    double thick[LLKT];   //the thickness of shell element
    double ztop, zbot;    //Z coordinate of the upper surface and the lower surface
    double zpoint[LLKT];  //Z coordinate of integration point
    double gjw[LLKT];     //integration weight through the thickness

    double d1[lg], d2[lg], d4[lg], d5[lg], d6[lg];  //in plane and shear strain increments

    double g;
    double d11, d12, d22, d33;  //material p

    double fac1[lg];

    double fmr11[lg], fmr12[lg], fmr21[lg], fmr22[lg];
    double ft11[lg], ft12[lg], ft13[lg], ft21[lg],
        ft22[lg], ft23[lg];  //forces for nodes 1 and 2 in local
    double ft31[lg], ft32[lg], ft33[lg], ft41[lg],
        ft42[lg], ft43[lg];  //forces for nodes 3 and 4 in local

    double fm11[lg], fm12[lg], fm21[lg], fm22[lg];  //moments for nodes 1 and 2 in local
    double fm31[lg], fm32[lg], fm41[lg], fm42[lg];  //moments for nodes 3 and 4 in local

    double tmode;  //hourglass p
    double wmode;
    double mmode;

    double qhx[lg], qhy[lg], qwz[lg];  //generalized hourglass strains
    double qtx[lg], qty[lg];

    double bsum[lg], xl[lg];
    //    double qs[5][NUMEL];
    double fac2[lg], fac3[lg];

    double rx1[lg], ry1[lg], rz1[lg];  //int forces of node1 in globe
    double rx2[lg], ry2[lg], rz2[lg];  //int forces of node2 in globe
    double rx3[lg], ry3[lg], rz3[lg];
    double rx4[lg], ry4[lg], rz4[lg];
    double rx5[lg], ry5[lg], rz5[lg];  //int moments of node1 in globe
    double rx6[lg], ry6[lg], rz6[lg];  //int moments of node2 in globe
    double rx7[lg], ry7[lg], rz7[lg];
    double rx8[lg], ry8[lg], rz8[lg];

    static double **gsig1, **gsig2, **gsig3, **gsig4, **gsig5, **gsig6;
    static double** qs;
    if (icyc == 1)
    {
        gsig1 = alloc_2d_dbl(llkt, numel);
        gsig2 = alloc_2d_dbl(llkt, numel);
        gsig3 = alloc_2d_dbl(llkt, numel);
        gsig4 = alloc_2d_dbl(llkt, numel);
        gsig5 = alloc_2d_dbl(llkt, numel);
        gsig6 = alloc_2d_dbl(llkt, numel);

        qs = alloc_2d_dbl(5, numnp);
    }

    //**********************************************************************************************************************
    // start to initial all matrix , we have to make all matrix zero, otherwise, fatal error should occur such as
    // if x=-8.123123123e49, and y=x*-1 >>> y=8.123123123e49 correspondingly.

    memset(ix1, 0, sizeof(ix1)), memset(ix2, 0, sizeof(ix2)), memset(ix3, 0, sizeof(ix3)), memset(ix4, 0, sizeof(ix4));
    memset(dx1, 0, sizeof(dx1)), memset(dx2, 0, sizeof(dx2)), memset(dx3, 0, sizeof(dx3)), memset(dx4, 0, sizeof(dx4));
    memset(dy1, 0, sizeof(dy2)), memset(dy2, 0, sizeof(dy2)), memset(dy3, 0, sizeof(dy3)), memset(dy4, 0, sizeof(dy4));
    memset(dz1, 0, sizeof(dz3)), memset(dz2, 0, sizeof(dz2)), memset(dz3, 0, sizeof(dz3)), memset(dz4, 0, sizeof(dz4));
    memset(wxx1, 0, sizeof(wxx1)), memset(wxx2, 0, sizeof(wxx2)), memset(wxx3, 0, sizeof(wxx3)), memset(wxx4, 0, sizeof(wxx4));
    memset(wyy1, 0, sizeof(wyy1)), memset(wyy2, 0, sizeof(wyy2)), memset(wyy3, 0, sizeof(wyy3)), memset(wyy4, 0, sizeof(wyy4));
    memset(wzz1, 0, sizeof(wzz1)), memset(wzz2, 0, sizeof(wzz2)), memset(wzz3, 0, sizeof(wzz3)), memset(wzz4, 0, sizeof(wzz4));

    memset(d1, 0, sizeof(d1)), memset(d2, 0, sizeof(d2)),
        memset(d4, 0, sizeof(d4)), memset(d5, 0, sizeof(d5)), memset(d6, 0, sizeof(d6));
    memset(zpoint, 0, sizeof(zpoint));
    memset(gmz, 0, sizeof(gmz));
    //ZeroMemory(gmz,sizeof(gmz));

    //////////////Added by CaiYong///////////////
    //     memset(cn1x1,0,sizeof(cn1x1)),memset(cn1y1,0,sizeof(cn1y1)),memset(cn1z1,0,sizeof(cn1z1));
    //     memset(cn1x2,0,sizeof(cn1x2)),memset(cn1y2,0,sizeof(cn1y2)),memset(cn1z2,0,sizeof(cn1z2));
    //     memset(cn1x3,0,sizeof(cn1x3)),memset(cn1y3,0,sizeof(cn1y3)),memset(cn1z3,0,sizeof(cn1z3));
    //     memset(cn1x4,0,sizeof(cn1x4)),memset(cn1y4,0,sizeof(cn1y4)),memset(cn1z4,0,sizeof(cn1z4));

    memset(cx1, 0, sizeof(cx1)), memset(cy1, 0, sizeof(cy1)), memset(cz1, 0, sizeof(cz1));
    memset(cx2, 0, sizeof(cx2)), memset(cy2, 0, sizeof(cy2)), memset(cz2, 0, sizeof(cz2));
    memset(cx3, 0, sizeof(cx3)), memset(cy3, 0, sizeof(cy3)), memset(cz3, 0, sizeof(cz3));
    memset(cx4, 0, sizeof(cx4)), memset(cy4, 0, sizeof(cy4)), memset(cz4, 0, sizeof(cz4));

    //     memset(xh21,0,sizeof(xh21)),memset(yh21,0,sizeof(yh21)),memset(zh21,0,sizeof(zh21));
    //     memset(xh31,0,sizeof(xh31)),memset(yh31,0,sizeof(yh31)),memset(zh31,0,sizeof(zh31));
    //     memset(xh41,0,sizeof(xh41)),memset(yh41,0,sizeof(yh41)),memset(zh41,0,sizeof(zh41));
    //     memset(xh42,0,sizeof(xh42)),memset(yh42,0,sizeof(yh42)),memset(zh42,0,sizeof(zh42));

    memset(gh13, 0, sizeof(gh13)), memset(gh23, 0, sizeof(gh23)), memset(gh33, 0, sizeof(gh33));
    memset(gh11, 0, sizeof(gh11)), memset(gh21, 0, sizeof(gh21)), memset(gh31, 0, sizeof(gh31));
    memset(gh12, 0, sizeof(gh12)), memset(gh22, 0, sizeof(gh22)), memset(gh32, 0, sizeof(gh32));

    memset(x21, 0, sizeof(x21)), memset(y21, 0, sizeof(y21)), memset(z21, 0, sizeof(z21));
    memset(x31, 0, sizeof(x31)), memset(y31, 0, sizeof(y31)), memset(z31, 0, sizeof(z31));
    memset(x41, 0, sizeof(x41)), memset(y41, 0, sizeof(y41)), memset(z41, 0, sizeof(z41));
    memset(x42, 0, sizeof(x42)), memset(y42, 0, sizeof(y42)), memset(z42, 0, sizeof(z42));

    memset(gl13, 0, sizeof(gl13)), memset(gl23, 0, sizeof(gl23)), memset(gl33, 0, sizeof(gl33));
    memset(gl12, 0, sizeof(gl12)), memset(gl22, 0, sizeof(gl22)), memset(gl32, 0, sizeof(gl32));
    memset(gl11, 0, sizeof(gl11)), memset(gl21, 0, sizeof(gl21)), memset(gl31, 0, sizeof(gl31));

    memset(vx1, 0, sizeof(vx1)), memset(vy1, 0, sizeof(vy1)), memset(vz1, 0, sizeof(vz1));
    memset(vx2, 0, sizeof(vx2)), memset(vy2, 0, sizeof(vy2)), memset(vz2, 0, sizeof(vz2));
    memset(vx3, 0, sizeof(vx3)), memset(vy3, 0, sizeof(vy3)), memset(vz3, 0, sizeof(vz3));
    memset(vx4, 0, sizeof(vx4)), memset(vy4, 0, sizeof(vy4)), memset(vz4, 0, sizeof(vz4));

    memset(vx5, 0, sizeof(vx5)), memset(vy5, 0, sizeof(vy5));
    memset(vx6, 0, sizeof(vx6)), memset(vy6, 0, sizeof(vy6));
    memset(vx7, 0, sizeof(vx7)), memset(vy7, 0, sizeof(vy7));
    memset(vx8, 0, sizeof(vx8)), memset(vy8, 0, sizeof(vy8));

    memset(x2, 0, sizeof(x2)), memset(y2, 0, sizeof(y2)), memset(z2, 0, sizeof(z2));
    memset(x3, 0, sizeof(x3)), memset(y3, 0, sizeof(y3)), memset(z3, 0, sizeof(z3));
    memset(x4, 0, sizeof(x4)), memset(y4, 0, sizeof(y4)), memset(z4, 0, sizeof(z4));

    memset(px1, 0, sizeof(px1)), memset(px2, 0, sizeof(px2)), memset(py1, 0, sizeof(py1)), memset(py2, 0, sizeof(py2));
    memset(area, 0, sizeof(area));
    memset(vx13, 0, sizeof(vx13)), memset(vy13, 0, sizeof(vy13)), memset(vz13, 0, sizeof(vz13));
    memset(vx24, 0, sizeof(vx24)), memset(vy24, 0, sizeof(vy24)), memset(vz24, 0, sizeof(vz24));
    memset(wxx13, 0, sizeof(wxx13)), memset(wyy13, 0, sizeof(wyy13));
    memset(wxx24, 0, sizeof(wxx24)), memset(wyy24, 0, sizeof(wyy24));

    memset(areain, 0, sizeof(areain));
    memset(px1a, 0, sizeof(px1a)), memset(py1a, 0, sizeof(py1a)), memset(px2a, 0, sizeof(px2a)), memset(py2a, 0, sizeof(py2a));
    memset(g1, 0, sizeof(g1)), memset(g2, 0, sizeof(g2)), memset(g3, 0, sizeof(g3));

    memset(b1vx, 0, sizeof(b1vx)), memset(b1vy, 0, sizeof(b1vy)), memset(b1vz, 0, sizeof(b1vz));
    memset(b2vx, 0, sizeof(b2vx)), memset(b2vy, 0, sizeof(b2vy)), memset(b2vz, 0, sizeof(b2vz));
    memset(b1tx, 0, sizeof(b1tx)), memset(b1ty, 0, sizeof(b1ty));
    memset(b2tx, 0, sizeof(b2tx)), memset(b2ty, 0, sizeof(b2ty));
    memset(b3tx, 0, sizeof(b3tx)), memset(b3ty, 0, sizeof(b3ty));
    memset(bxyv, 0, sizeof(bxyv)), memset(bxyt, 0, sizeof(bxyt));

    memset(sx, 0, sizeof(sx)), memset(sy, 0, sizeof(sy));
    memset(epyz, 0, sizeof(epyz)), memset(epzx, 0, sizeof(epzx));

    memset(htx, 0, sizeof(htx)), memset(hty, 0, sizeof(hty));
    memset(gm1, 0, sizeof(gm1)), memset(gm2, 0, sizeof(gm2)), memset(gm3, 0, sizeof(gm3)), memset(gm4, 0, sizeof(gm4));
    //memset(gmz,0,sizeof(gmz));

    memset(thick, 0, sizeof(thick));
    //memset(zpoint,0,sizeof(zpoint));
    memset(gjw, 0, sizeof(gjw));

    //memset(d1,0,sizeof(d1)),memset(d2,0,sizeof(d2));
    memset(d4, 0, sizeof(d4)), memset(d5, 0, sizeof(d5)), memset(d6, 0, sizeof(d6));

    memset(fac1, 0, sizeof(fac1));

    memset(fmr11, 0, sizeof(fmr11)), memset(fmr12, 0, sizeof(fmr12)), memset(fmr21, 0, sizeof(fmr21)), memset(fmr22, 0, sizeof(fmr22));
    memset(ft11, 0, sizeof(ft11)), memset(ft12, 0, sizeof(ft12)), memset(ft13, 0, sizeof(ft13)), memset(ft21, 0, sizeof(ft21)), memset(ft22, 0, sizeof(ft22)), memset(ft23, 0, sizeof(ft23));
    memset(ft31, 0, sizeof(ft31)), memset(ft32, 0, sizeof(ft32)), memset(ft33, 0, sizeof(ft33)), memset(ft41, 0, sizeof(ft41)), memset(ft42, 0, sizeof(ft42)), memset(ft43, 0, sizeof(ft43));

    memset(fm11, 0, sizeof(fm11)), memset(fm12, 0, sizeof(fm12)), memset(fm21, 0, sizeof(fm21)), memset(fm22, 0, sizeof(fm22));
    memset(fm31, 0, sizeof(fm31)), memset(fm32, 0, sizeof(fm32)), memset(fm41, 0, sizeof(fm41)), memset(fm42, 0, sizeof(fm42));

    memset(qhx, 0, sizeof(qhx)), memset(qhy, 0, sizeof(qhy)), memset(qwz, 0, sizeof(qwz));
    memset(qtx, 0, sizeof(qtx)), memset(qty, 0, sizeof(qty));

    memset(bsum, 0, sizeof(bsum)), memset(xl, 0, sizeof(xl));
    memset(fac2, 0, sizeof(fac2)), memset(fac3, 0, sizeof(fac3));

    memset(rx1, 0, sizeof(rx1)), memset(ry1, 0, sizeof(ry1)), memset(rz1, 0, sizeof(rz1));
    memset(rx2, 0, sizeof(rx2)), memset(ry2, 0, sizeof(ry2)), memset(rz2, 0, sizeof(rz2));
    memset(rx3, 0, sizeof(rx3)), memset(ry3, 0, sizeof(ry3)), memset(rz3, 0, sizeof(rz3));
    memset(rx4, 0, sizeof(rx4)), memset(ry4, 0, sizeof(ry4)), memset(rz4, 0, sizeof(rz4));
    memset(rx5, 0, sizeof(rx5)), memset(ry5, 0, sizeof(ry5)), memset(rz5, 0, sizeof(rz5));
    memset(rx6, 0, sizeof(rx6)), memset(ry6, 0, sizeof(ry6)), memset(rz6, 0, sizeof(rz6));
    memset(rx7, 0, sizeof(rx7)), memset(ry7, 0, sizeof(ry7)), memset(rz7, 0, sizeof(rz7));
    memset(rx8, 0, sizeof(rx8)), memset(ry8, 0, sizeof(ry8)), memset(rz8, 0, sizeof(rz8));
    ///////////////THE END////////////////////////////

    //**********************************************************************************************************************

    gsssh();  // Gauss integration

    llkt = gllkt;

    for (int i = 0; i < gng + 1; i++)  //ing main loop
    {
        illt = 64;  //illt
        if (i > gng - 1)
            illt = gmg;   //mg
        inft = ( i )*64;  //inft

        for (int j = 0; j < illt; j++)
        {
            in[j]   = inft + j;
            imat[j] = gix[4][in[j]];
        }
        //gather initial coordinates into local vectors
        for (int k = 0; k < illt; k++)
        {
            // 获得每个单元的节点编号
            ix1[k] = gix[0][in[k]];  //the first node of element in[k]
            ix2[k] = gix[1][in[k]];  //the second node of element in[k]
            ix3[k] = gix[2][in[k]];
            ix4[k] = gix[3][in[k]];
        }

        for (int k = 0; k < illt; k++)
        {
            //unpack displacement and rotation increments
            dx1[k]  = gv[0][ix1[k]] * dt;  //x displacement increment of the first node
            dy1[k]  = gv[1][ix1[k]] * dt;
            dz1[k]  = gv[2][ix1[k]] * dt;
            wxx1[k] = gv[3][ix1[k]] * dt;  //x rotation increment of the first node
            wyy1[k] = gv[4][ix1[k]] * dt;
            wzz1[k] = gv[5][ix1[k]] * dt;

            //printf("\n wxx1 = %13.8e\n",wxx1[1]);
            //system("pause");

            dx2[k]  = gv[0][ix2[k]] * dt;  //x displacement increment of the second node
            dy2[k]  = gv[1][ix2[k]] * dt;
            dz2[k]  = gv[2][ix2[k]] * dt;
            wxx2[k] = gv[3][ix2[k]] * dt;  //x rotation increment of the second node
            wyy2[k] = gv[4][ix2[k]] * dt;
            wzz2[k] = gv[5][ix2[k]] * dt;

            dx3[k]  = gv[0][ix3[k]] * dt;  //x displacement increment of the third node
            dy3[k]  = gv[1][ix3[k]] * dt;
            dz3[k]  = gv[2][ix3[k]] * dt;
            wxx3[k] = gv[3][ix3[k]] * dt;  //x rotation increment of the third node
            wyy3[k] = gv[4][ix3[k]] * dt;
            wzz3[k] = gv[5][ix3[k]] * dt;

            dx4[k]  = gv[0][ix4[k]] * dt;  //x displacement increment of the forth node
            dy4[k]  = gv[1][ix4[k]] * dt;
            dz4[k]  = gv[2][ix4[k]] * dt;
            wxx4[k] = gv[3][ix4[k]] * dt;  //x rotation increment of the forth node
            wyy4[k] = gv[4][ix4[k]] * dt;
            wzz4[k] = gv[5][ix4[k]] * dt;
        }  //increments loop end

        for (int k = 0; k < illt; k++)
        {
            //current coordinates at n+1
            /* The original temper variables have been removed
        cn1x1[k]=gx[0][ix1[k]];
        cn1y1[k]=gx[1][ix1[k]];
        cn1z1[k]=gx[2][ix1[k]];
        cn1x2[k]=gx[0][ix2[k]];
        cn1y2[k]=gx[1][ix2[k]];
        cn1z2[k]=gx[2][ix2[k]];
        cn1x3[k]=gx[0][ix3[k]];
        cn1y3[k]=gx[1][ix3[k]];
        cn1z3[k]=gx[2][ix3[k]];
        cn1x4[k]=gx[0][ix4[k]];
        cn1y4[k]=gx[1][ix4[k]];
        cn1z4[k]=gx[2][ix4[k]];
        */

            //compute coordinates at n+1/2 by subtracting 1/2
            //the displacement increment from the coordinates at n+1
            cx1[k] = gx[0][ix1[k]] - 0.5 * dx1[k];
            cy1[k] = gx[1][ix1[k]] - 0.5 * dy1[k];
            cz1[k] = gx[2][ix1[k]] - 0.5 * dz1[k];
            cx2[k] = gx[0][ix2[k]] - 0.5 * dx2[k];
            cy2[k] = gx[1][ix2[k]] - 0.5 * dy2[k];
            cz2[k] = gx[2][ix2[k]] - 0.5 * dz2[k];
            cx3[k] = gx[0][ix3[k]] - 0.5 * dx3[k];
            cy3[k] = gx[1][ix3[k]] - 0.5 * dy3[k];
            cz3[k] = gx[2][ix3[k]] - 0.5 * dz3[k];
            cx4[k] = gx[0][ix4[k]] - 0.5 * dx4[k];
            cy4[k] = gx[1][ix4[k]] - 0.5 * dy4[k];
            cz4[k] = gx[2][ix4[k]] - 0.5 * dz4[k];
        }
        //node coordinates loop end

        //compute the lamina coordinate system at both n+1/2 and n+1 geometry
        for (int k = 0; k < illt; k++)
        {
            /* The original temper variables have been removed
        xh21[k]=cx2[k]-cx1[k];
        yh21[k]=cy2[k]-cy1[k];
        zh21[k]=cz2[k]-cz1[k];
        xh31[k]=cx3[k]-cx1[k];
        yh31[k]=cy3[k]-cy1[k];
        zh31[k]=cz3[k]-cz1[k];
        xh41[k]=cx4[k]-cx1[k];
        yh41[k]=cy4[k]-cy1[k];
        zh41[k]=cz4[k]-cz1[k];
        xh42[k]=cx4[k]-cx2[k];
        yh42[k]=cy4[k]-cy2[k];
        zh42[k]=cz4[k]-cz2[k];
        */

            //unit normal vector to lamina surface at n+1/2
            c1      = (cy3[k] - cy1[k]) * (cz4[k] - cz2[k]) - (cz3[k] - cz1[k]) * (cy4[k] - cy2[k]);
            c2      = (cz3[k] - cz1[k]) * (cx4[k] - cx2[k]) - (cx3[k] - cx1[k]) * (cz4[k] - cz2[k]);
            c3      = (cx3[k] - cx1[k]) * (cy4[k] - cy2[k]) - (cy3[k] - cy1[k]) * (cx4[k] - cx2[k]);
            al      = 1.0 / sqrt(c1 * c1 + c2 * c2 + c3 * c3);
            gh13[k] = c1 * al;  //normal vector n+1/2, e3(n+1/2)
            gh23[k] = c2 * al;
            gh33[k] = c3 * al;
        }

        for (int k = 0; k < illt; k++)
        {
            xhh     = (cx2[k] - cx1[k]) * gh13[k] + (cy2[k] - cy1[k]) * gh23[k] + (cz2[k] - cz1[k]) * gh33[k];
            c1      = (cx2[k] - cx1[k]) - gh13[k] * xhh;
            c2      = (cy2[k] - cy1[k]) - gh23[k] * xhh;
            c3      = (cz2[k] - cz1[k]) - gh33[k] * xhh;
            al      = 1.0 / sqrt(c1 * c1 + c2 * c2 + c3 * c3);
            gh11[k] = c1 * al;  //e1(n+1/2)
            gh21[k] = c2 * al;
            gh31[k] = c3 * al;

            gh12[k] = gh23[k] * gh31[k] - gh33[k] * gh21[k];
            gh22[k] = gh33[k] * gh11[k] - gh13[k] * gh31[k];
            gh32[k] = gh13[k] * gh21[k] - gh23[k] * gh11[k];
        }

        for (int k = 0; k < illt; k++)
        {
            //vector difference of nodal coordinates at n+1
            x21[k] = gx[0][ix2[k]] - gx[0][ix1[k]];
            y21[k] = gx[1][ix2[k]] - gx[1][ix1[k]];
            z21[k] = gx[2][ix2[k]] - gx[2][ix1[k]];
            x31[k] = gx[0][ix3[k]] - gx[0][ix1[k]];
            y31[k] = gx[1][ix3[k]] - gx[1][ix1[k]];
            z31[k] = gx[2][ix3[k]] - gx[2][ix1[k]];
            x41[k] = gx[0][ix4[k]] - gx[0][ix1[k]];
            y41[k] = gx[1][ix4[k]] - gx[1][ix1[k]];
            z41[k] = gx[2][ix4[k]] - gx[2][ix1[k]];
            x42[k] = gx[0][ix4[k]] - gx[0][ix2[k]];
            y42[k] = gx[1][ix4[k]] - gx[1][ix2[k]];
            z42[k] = gx[2][ix4[k]] - gx[2][ix2[k]];
        }

        for (int k = 0; k < illt; k++)
        {
            //find unit vectors tangent to lamina surface at n+1
            c1      = y31[k] * z42[k] - z31[k] * y42[k];
            c2      = z31[k] * x42[k] - x31[k] * z42[k];
            c3      = x31[k] * y42[k] - y31[k] * x42[k];
            al      = 1.0 / sqrt(c1 * c1 + c2 * c2 + c3 * c3);
            gl13[k] = c1 * al;  //e3(n+1/2)
            gl23[k] = c2 * al;
            gl33[k] = c3 * al;
        }

        for (int k = 0; k < illt; k++)
        {
            xll     = x21[k] * gl13[k] + y21[k] * gl23[k] + z21[k] * gl33[k];
            c1      = x21[k] - gl13[k] * xll;
            c2      = y21[k] - gl23[k] * xll;
            c3      = z21[k] - gl33[k] * xll;
            al      = 1.0 / sqrt(c1 * c1 + c2 * c2 + c3 * c3);
            gl11[k] = c1 * al;  //e1(n+1/2)
            gl21[k] = c2 * al;
            gl31[k] = c3 * al;

            gl12[k] = gl23[k] * gl31[k] - gl33[k] * gl21[k];  //e2(n+1/2)
            gl22[k] = gl33[k] * gl11[k] - gl13[k] * gl31[k];
            gl32[k] = gl13[k] * gl21[k] - gl23[k] * gl11[k];
        }
        //lamina coordinate system loop end

        //compute translation and rotation increments in local system
        for (int k = 0; k < illt; k++)
        {
            //translation increments in local system
            vx1[k] = gh11[k] * dx1[k] + gh21[k] * dy1[k] + gh31[k] * dz1[k];
            vy1[k] = gh12[k] * dx1[k] + gh22[k] * dy1[k] + gh32[k] * dz1[k];
            vz1[k] = gh13[k] * dx1[k] + gh23[k] * dy1[k] + gh33[k] * dz1[k];
            vx2[k] = gh11[k] * dx2[k] + gh21[k] * dy2[k] + gh31[k] * dz2[k];
            vy2[k] = gh12[k] * dx2[k] + gh22[k] * dy2[k] + gh32[k] * dz2[k];
            vz2[k] = gh13[k] * dx2[k] + gh23[k] * dy2[k] + gh33[k] * dz2[k];
            vx3[k] = gh11[k] * dx3[k] + gh21[k] * dy3[k] + gh31[k] * dz3[k];
            vy3[k] = gh12[k] * dx3[k] + gh22[k] * dy3[k] + gh32[k] * dz3[k];
            vz3[k] = gh13[k] * dx3[k] + gh23[k] * dy3[k] + gh33[k] * dz3[k];
            vx4[k] = gh11[k] * dx4[k] + gh21[k] * dy4[k] + gh31[k] * dz4[k];
            vy4[k] = gh12[k] * dx4[k] + gh22[k] * dy4[k] + gh32[k] * dz4[k];
            vz4[k] = gh13[k] * dx4[k] + gh23[k] * dy4[k] + gh33[k] * dz4[k];

            //rotation increments in local system
            vx5[k] = gh11[k] * wxx1[k] + gh21[k] * wyy1[k] + gh31[k] * wzz1[k];
            vy5[k] = gh12[k] * wxx1[k] + gh22[k] * wyy1[k] + gh32[k] * wzz1[k];
            vx6[k] = gh11[k] * wxx2[k] + gh21[k] * wyy2[k] + gh31[k] * wzz2[k];
            vy6[k] = gh12[k] * wxx2[k] + gh22[k] * wyy2[k] + gh32[k] * wzz2[k];
            vx7[k] = gh11[k] * wxx3[k] + gh21[k] * wyy3[k] + gh31[k] * wzz3[k];
            vy7[k] = gh12[k] * wxx3[k] + gh22[k] * wyy3[k] + gh32[k] * wzz3[k];
            vx8[k] = gh11[k] * wxx4[k] + gh21[k] * wyy4[k] + gh31[k] * wzz4[k];
            vy8[k] = gh12[k] * wxx4[k] + gh22[k] * wyy4[k] + gh32[k] * wzz4[k];
        }

        //compute the coordinates of nodes in local coordinate system
        for (int k = 0; k < illt; k++)
        {
            x2[k] = gh11[k] * (cx2[k] - cx1[k]) + gh21[k] * (cy2[k] - cy1[k]) + gh31[k] * (cz2[k] - cz1[k]);
            y2[k] = gh12[k] * (cx2[k] - cx1[k]) + gh22[k] * (cy2[k] - cy1[k]) + gh32[k] * (cz2[k] - cz1[k]);
            x3[k] = gh11[k] * (cx3[k] - cx1[k]) + gh21[k] * (cy3[k] - cy1[k]) + gh31[k] * (cz3[k] - cz1[k]);
            y3[k] = gh12[k] * (cx3[k] - cx1[k]) + gh22[k] * (cy3[k] - cy1[k]) + gh32[k] * (cz3[k] - cz1[k]);
            x4[k] = gh11[k] * (cx4[k] - cx1[k]) + gh21[k] * (cy4[k] - cy1[k]) + gh31[k] * (cz4[k] - cz1[k]);
            y4[k] = gh12[k] * (cx4[k] - cx1[k]) + gh22[k] * (cy4[k] - cy1[k]) + gh32[k] * (cz4[k] - cz1[k]);
        }  //local coordinates of nodes loop end

        for (int k = 0; k < illt; k++)
        {
            px1[k] = 0.5 * (y2[k] - y4[k]);
            px2[k] = 0.5 * y3[k];
            py1[k] = -0.5 * (x2[k] - x4[k]);
            py2[k] = -0.5 * x3[k];

            //compute the area of the element
            area[k] = 2.0 * (py2[k] * px1[k] - py1[k] * px2[k]);

            //compute difference in displacements, rotations for b x d calculation
            vx13[k]  = vx1[k] - vx3[k];  //x difference between node1 and node3 in displacements in local system
            vx24[k]  = vx2[k] - vx4[k];  //x difference between node2 and node4 in displacements in local system
            wxx13[k] = vx5[k] - vx7[k];  //x difference between node1 and node3 in rotations in local system
            wxx24[k] = vx6[k] - vx8[k];  //x difference between node1 and node3 in rotations in local system
            vy13[k]  = vy1[k] - vy3[k];
            vy24[k]  = vy2[k] - vy4[k];
            wyy13[k] = vy5[k] - vy7[k];
            wyy24[k] = vy6[k] - vy8[k];
            vz13[k]  = vz1[k] - vz3[k];
            vz24[k]  = vz2[k] - vz4[k];

            //areain is the reciprocan of the element area
            //Modify by CaiYong
            //areain[k]=1.0 / (area[i]+1.0E-20);
            areain[k] = 1.0 / (area[k] + 1.0e-20);
            px1a[k]   = areain[k] * px1[k];
            py1a[k]   = areain[k] * py1[k];
            px2a[k]   = areain[k] * px2[k];
            py2a[k]   = areain[k] * py2[k];

            g1[k] = py2[k] * vx13[k] + py1[k] * vx24[k];
            g2[k] = -px2[k] * vy13[k] - px1[k] * vy24[k];
            g3[k] = py2[k] * vy13[k] + py1[k] * vy24[k] - px2[k] * vx13[k] - px1[k] * vx24[k];

        }  //

        //compute membrane and bending strain, elements of shear strain
        for (int k = 0; k < illt; k++)
        {
            b1vx[k] = px1a[k] * vx13[k] + px2a[k] * vx24[k];
            b1vy[k] = px1a[k] * vy13[k] + px2a[k] * vy24[k];
            b1vz[k] = px1a[k] * vz13[k] + px2a[k] * vz24[k];
            b2vx[k] = py1a[k] * vx13[k] + py2a[k] * vx24[k];
            b2vy[k] = py1a[k] * vy13[k] + py2a[k] * vy24[k];
            b2vz[k] = py1a[k] * vz13[k] + py2a[k] * vz24[k];
            b1tx[k] = px1a[k] * wxx13[k] + px2a[k] * wxx24[k];
            b1ty[k] = px1a[k] * wyy13[k] + px2a[k] * wyy24[k];
            b2tx[k] = py1a[k] * wxx13[k] + py2a[k] * wxx24[k];
            b2ty[k] = py1a[k] * wyy13[k] + py2a[k] * wyy24[k];
            bxyv[k] = b2vx[k] + b1vy[k];
            bxyt[k] = b2ty[k] - b1tx[k];
        }  //

        //compute incremental shear strains
        for (int k = 0; k < illt; k++)
        {
            sx[k]   = vx5[k] + vx6[k] + vx7[k] + vx8[k];
            sy[k]   = vy5[k] + vy6[k] + vy7[k] + vy8[k];
            epyz[k] = b2vz[k] - 0.25 * sx[k];  //incremental shear strain yz
            epzx[k] = b1vz[k] + 0.25 * sy[k];  //incremental shear strain zx
        }                                      //incremental shear strains

        // gamma define loop
        for (int k = 0; k < illt; k++)
        {
            htx[k] = areain[k] * (x3[k] - x2[k] - x4[k]);
            hty[k] = areain[k] * (y3[k] - y2[k] - y4[k]);

            //define gamma
            gm1[k] = 1.0 - px1[k] * htx[k] - py1[k] * hty[k];
            gm2[k] = -1.0 - px2[k] * htx[k] - py2[k] * hty[k];
            gm3[k] = 2.0 - gm1[k];
            gm4[k] = -2.0 - gm2[k];
            gmz[k] = gm2[k] * z2[k] + gm3[k] * z3[k] + gm4[k] * z4[k];
        }  // gamma define loop end

        for (int k = 0; k < illt; k++)
        {
            //clear integrate resultant forces and moments
            sg1m = 0.0;
            sg2m = 0.0;
            sg4m = 0.0;
            sg1n = 0.0;
            sg2n = 0.0;
            sg4n = 0.0;
            sg5n = 0.0;
            sg6n = 0.0;
            sg5l = 0.0;
            sg6l = 0.0;

            //loop over through thickness integrations
            //Following codes need to consider accuracy of C and Fortran modified Steven Wang

            for (int ipt = 0; ipt < llkt; ipt++)
            {
                int ni      = llkt - 1;
                thick[ipt]  = gtk[0][in[k]];
                ztop        = 0.5 * thick[ipt];
                zbot        = -0.5 * thick[ipt];
                zpoint[ipt] = 0.5 * (1.0 - gzeta[ipt][ni]) * zbot + 0.5 * (1.0 + gzeta[ipt][ni]) * ztop;
                gjw[ipt]    = gtk[0][in[k]] / 2.0 * ggw[ipt][ni] * area[k];

                //compute the in plane and shear strain increments at integration point
                d1[k] = b1vx[k] + zpoint[ipt] * (b1ty[k] + gmz[k] * g1[k] / (area[k] * area[k]));  //dexx
                d2[k] = b2vy[k] - zpoint[ipt] * (b2tx[k] + gmz[k] * g2[k] / (area[k] * area[k]));  //deyy
                d4[k] = bxyv[k] + zpoint[ipt] * (bxyt[k] + gmz[k] * g3[k] / (area[k] * area[k]));  //dezz
                d5[k] = epyz[k];                                                                   //deyz
                d6[k] = epzx[k];                                                                   //dezx

                g   = gyms / (2.0 * (1.0 + gpro));
                d11 = gyms / (1.0 - gpro * gpro);
                d12 = gpro * d11;
                d22 = d11;
                d33 = gyms / (2.0 * (1.0 + gpro));
                c3  = 5.0 * g / 6.0;

                gsig3[ipt][in[k]] = 0.0;  //z stress

                //shear stress zx and yz
                gsig5[ipt][in[k]] = gsig5[ipt][in[k]] + c3 * d5[k];  //stress yz
                gsig6[ipt][in[k]] = gsig6[ipt][in[k]] + c3 * d6[k];  //stress zx

                //stresses with elastic material
                gsig1[ipt][in[k]] = d11 * d1[k] + d12 * d2[k] + gsig1[ipt][in[k]];  //stress x
                gsig2[ipt][in[k]] = d12 * d1[k] + d22 * d2[k] + gsig2[ipt][in[k]];  //stress y
                gsig4[ipt][in[k]] = d33 * d4[k] + gsig4[ipt][in[k]];                //stress xy

                fac1[ipt] = gjw[ipt] * zpoint[ipt];

                sg1m = sg1m + fac1[ipt] * gsig1[ipt][in[k]];
                sg2m = sg2m + fac1[ipt] * gsig2[ipt][in[k]];  // less accuracy compared with Fortran, Attention
                sg4m = sg4m + fac1[ipt] * gsig4[ipt][in[k]];
                sg1n = sg1n + gjw[ipt] * gsig1[ipt][in[k]];
                sg2n = sg2n + gjw[ipt] * gsig2[ipt][in[k]];
                sg4n = sg4n + gjw[ipt] * gsig4[ipt][in[k]];
                sg5n = sg5n + gjw[ipt] * gsig5[ipt][in[k]];
                sg6n = sg6n + gjw[ipt] * gsig6[ipt][in[k]];
                sg5l = sg5l - gjw[ipt] * gsig5[ipt][in[k]];
                sg6l = sg6l + gjw[ipt] * gsig6[ipt][in[k]];
            }  //end loop over through-thickness integration

            //find force components in local system

            //for compute moments
            fmr11[k] = -py1a[k] * sg2m - px1a[k] * sg4m;
            fmr12[k] = px1a[k] * sg1m + py1a[k] * sg4m;
            fmr21[k] = -py2a[k] * sg2m - px2a[k] * sg4m;
            fmr22[k] = px2a[k] * sg1m + py2a[k] * sg4m;

            //compute forces for nodes 1 and 2 (opposite in sign as 3 and 4)
            ft11[k] = px1a[k] * sg1n + py1a[k] * sg4n;
            ft12[k] = py1a[k] * sg2n + px1a[k] * sg4n;
            ft21[k] = px2a[k] * sg1n + py2a[k] * sg4n;
            ft22[k] = py2a[k] * sg2n + px2a[k] * sg4n;
            ft13[k] = px1a[k] * sg6n + py1a[k] * sg5n;
            ft23[k] = px2a[k] * sg6n + py2a[k] * sg5n;

            //compute moments all nodes
            fm11[k] = 0.25 * sg5l + fmr11[k];
            fm12[k] = 0.25 * sg6l + fmr12[k];
            fm21[k] = 0.25 * sg5l + fmr21[k];
            fm22[k] = 0.25 * sg6l + fmr22[k];
            fm31[k] = 0.25 * sg5l - fmr11[k];
            fm32[k] = 0.25 * sg6l - fmr12[k];
            fm41[k] = 0.25 * sg5l - fmr21[k];
            fm42[k] = 0.25 * sg6l - fmr22[k];

        }  //end loop integrate resultant forces and moments

        if (qhg > 1.0e-04)  //start if compute the hourglass forces and moments
        {
            tmode = qhg * gyms / 1920.0;
            wmode = qhg * g / 120.0;
            mmode = qhg * gyms / 80.0;

            //compute the hourglass forces and moments
            for (int k = 0; k < illt; k++)
            {
                htx[k] = areain[k] * (x3[k] - x2[k] - x4[k]);
                hty[k] = areain[k] * (y3[k] - y2[k] - y4[k]);

                gm1[k] = 1.0 - px1[k] * htx[k] - py1[k] * hty[k];
                gm2[k] = -1.0 - px2[k] * htx[k] - py2[k] * hty[k];
                gm3[k] = 2.0 - gm1[k];
                gm4[k] = -2.0 - gm2[k];

                //Modify by CaiYong
                bsum[k] = 2. * (px1[k] * px1[k] + px2[k] * px2[k] + py1[k] * py1[k] + py2[k] * py2[k]);
                xl[k]   = areain[k] * bsum[k] * gtk[0][in[k]];

                fac1[k] = xl[k] * gtk[0][in[k]] * gtk[0][in[k]];
                fac2[k] = wmode * fac1[k] * areain[k];
                fac3[k] = mmode * xl[k];
                fac1[k] = tmode * fac1[k];

                //define generalized hourglass strains
                qhx[k] = gm1[k] * vx1[k] + gm2[k] * vx2[k] + gm3[k] * vx3[k] + gm4[k] * vx4[k];
                qhy[k] = gm1[k] * vy1[k] + gm2[k] * vy2[k] + gm3[k] * vy3[k] + gm4[k] * vy4[k];
                qwz[k] = gm1[k] * vz1[k] + gm2[k] * vz2[k] + gm3[k] * vz3[k] + gm4[k] * vz4[k];
                qtx[k] = gm1[k] * vx5[k] + gm2[k] * vx6[k] + gm3[k] * vx7[k] + gm4[k] * vx8[k];
                qty[k] = gm1[k] * vy5[k] + gm2[k] * vy6[k] + gm3[k] * vy7[k] + gm4[k] * vy8[k];
            }

            //Modify
            //memset(qs,0.0E0,sizeof(qs));
            for (int k = 0; k < illt; k++)
            {
                //add hourglass control to forces and moments
                qs[0][in[k]] = qs[0][in[k]] + fac3[k] * qhx[k];
                qs[1][in[k]] = qs[1][in[k]] + fac3[k] * qhy[k];
                qs[2][in[k]] = qs[2][in[k]] + fac2[k] * qwz[k];
                qs[3][in[k]] = qs[3][in[k]] + fac1[k] * qtx[k];
                qs[4][in[k]] = qs[4][in[k]] + fac1[k] * qty[k];

                ft31[k] = -ft11[k] + gm3[k] * qs[0][in[k]];
                ft32[k] = -ft12[k] + gm3[k] * qs[1][in[k]];
                ft33[k] = -ft13[k] + gm3[k] * qs[2][in[k]];
                ft41[k] = -ft21[k] + gm4[k] * qs[0][in[k]];
                ft42[k] = -ft22[k] + gm4[k] * qs[1][in[k]];
                ft43[k] = -ft23[k] + gm4[k] * qs[2][in[k]];
                ft11[k] = ft11[k] + gm1[k] * qs[0][in[k]];
                ft12[k] = ft12[k] + gm1[k] * qs[1][in[k]];
                ft13[k] = ft13[k] + gm1[k] * qs[2][in[k]];
                ft21[k] = ft21[k] + gm2[k] * qs[0][in[k]];
                ft22[k] = ft22[k] + gm2[k] * qs[1][in[k]];
                ft23[k] = ft23[k] + gm2[k] * qs[2][in[k]];
                fm11[k] = fm11[k] + gm1[k] * qs[3][in[k]];
                fm12[k] = fm12[k] + gm1[k] * qs[4][in[k]];
                fm21[k] = fm21[k] + gm2[k] * qs[3][in[k]];
                fm22[k] = fm22[k] + gm2[k] * qs[4][in[k]];
                fm31[k] = fm31[k] + gm3[k] * qs[3][in[k]];
                fm32[k] = fm32[k] + gm3[k] * qs[4][in[k]];
                fm41[k] = fm41[k] + gm4[k] * qs[3][in[k]];
                fm42[k] = fm42[k] + gm4[k] * qs[4][in[k]];
            }

        }  //end if compute the hourglass forces and moments
        else
        {
            for (int k = 0; k < illt; k++)
            {
                //define forces for nodes 3 and 4 if hourglass control is not used
                ft31[k] = -ft11[k];
                ft32[k] = -ft12[k];
                ft33[k] = -ft13[k];
                ft41[k] = -ft21[k];  //x force in local
                ft42[k] = -ft22[k];
                ft43[k] = -ft23[k];
            }  //end loop forces for nodes 3 and 4
        }      //end if no hourglass forces and moments

        for (int k = 0; k < illt; k++)
        {
            //transform forces to global system
            rx1[k] = gl11[k] * ft11[k] + gl12[k] * ft12[k] + gl13[k] * ft13[k];  //x int force of node1 in globe
            ry1[k] = gl21[k] * ft11[k] + gl22[k] * ft12[k] + gl23[k] * ft13[k];  //y int force of node1 in globe
            rz1[k] = gl31[k] * ft11[k] + gl32[k] * ft12[k] + gl33[k] * ft13[k];  //z int force of node1 in globe

            rx2[k] = gl11[k] * ft21[k] + gl12[k] * ft22[k] + gl13[k] * ft23[k];  //x int force of node2 in globe
            ry2[k] = gl21[k] * ft21[k] + gl22[k] * ft22[k] + gl23[k] * ft23[k];
            rz2[k] = gl31[k] * ft21[k] + gl32[k] * ft22[k] + gl33[k] * ft23[k];

            rx3[k] = gl11[k] * ft31[k] + gl12[k] * ft32[k] + gl13[k] * ft33[k];  //x int force of node3 in globe
            ry3[k] = gl21[k] * ft31[k] + gl22[k] * ft32[k] + gl23[k] * ft33[k];
            rz3[k] = gl31[k] * ft31[k] + gl32[k] * ft32[k] + gl33[k] * ft33[k];

            rx4[k] = gl11[k] * ft41[k] + gl12[k] * ft42[k] + gl13[k] * ft43[k];  //x int force of node4 in globe
            ry4[k] = gl21[k] * ft41[k] + gl22[k] * ft42[k] + gl23[k] * ft43[k];
            rz4[k] = gl31[k] * ft41[k] + gl32[k] * ft42[k] + gl33[k] * ft43[k];

            //transform moments to global system
            rx5[k] = gl11[k] * fm11[k] + gl12[k] * fm12[k];  //x int moment of node1 in globe
            ry5[k] = gl21[k] * fm11[k] + gl22[k] * fm12[k];
            rz5[k] = gl31[k] * fm11[k] + gl32[k] * fm12[k];

            rx6[k] = gl11[k] * fm21[k] + gl12[k] * fm22[k];  //x int moment of node2 in globe
            ry6[k] = gl21[k] * fm21[k] + gl22[k] * fm22[k];
            rz6[k] = gl31[k] * fm21[k] + gl32[k] * fm22[k];

            rx7[k] = gl11[k] * fm31[k] + gl12[k] * fm32[k];  //x int moment of node3 in globe
            ry7[k] = gl21[k] * fm31[k] + gl22[k] * fm32[k];
            rz7[k] = gl31[k] * fm31[k] + gl32[k] * fm32[k];

            rx8[k] = gl11[k] * fm41[k] + gl12[k] * fm42[k];  //x int moment of node4 in globe
            ry8[k] = gl21[k] * fm41[k] + gl22[k] * fm42[k];
            rz8[k] = gl31[k] * fm41[k] + gl32[k] * fm42[k];
        }  //end loop int forces and moments in global system

        //compute the force
        for (int k = 0; k < illt; k++)
        {
            gff[0][ix1[k]] = gff[0][ix1[k]] - rx1[k];
            gff[1][ix1[k]] = gff[1][ix1[k]] - ry1[k];
            gff[2][ix1[k]] = gff[2][ix1[k]] - rz1[k];
            gfm[0][ix1[k]] = gfm[0][ix1[k]] - rx5[k];
            gfm[1][ix1[k]] = gfm[1][ix1[k]] - ry5[k];
            gfm[2][ix1[k]] = gfm[2][ix1[k]] - rz5[k];
            gff[0][ix2[k]] = gff[0][ix2[k]] - rx2[k];
            gff[1][ix2[k]] = gff[1][ix2[k]] - ry2[k];
            gff[2][ix2[k]] = gff[2][ix2[k]] - rz2[k];
            gfm[0][ix2[k]] = gfm[0][ix2[k]] - rx6[k];
            gfm[1][ix2[k]] = gfm[1][ix2[k]] - ry6[k];
            gfm[2][ix2[k]] = gfm[2][ix2[k]] - rz6[k];
            gff[0][ix3[k]] = gff[0][ix3[k]] - rx3[k];
            gff[1][ix3[k]] = gff[1][ix3[k]] - ry3[k];
            gff[2][ix3[k]] = gff[2][ix3[k]] - rz3[k];
            gfm[0][ix3[k]] = gfm[0][ix3[k]] - rx7[k];
            gfm[1][ix3[k]] = gfm[1][ix3[k]] - ry7[k];
            gfm[2][ix3[k]] = gfm[2][ix3[k]] - rz7[k];
            gff[0][ix4[k]] = gff[0][ix4[k]] - rx4[k];
            gff[1][ix4[k]] = gff[1][ix4[k]] - ry4[k];
            gff[2][ix4[k]] = gff[2][ix4[k]] - rz4[k];
            gfm[0][ix4[k]] = gfm[0][ix4[k]] - rx8[k];
            gfm[1][ix4[k]] = gfm[1][ix4[k]] - ry8[k];
            gfm[2][ix4[k]] = gfm[2][ix4[k]] - rz8[k];
        }  //end loop nodes forces

    }  //main loop end

    return 1;

}  //subroutine bt end
