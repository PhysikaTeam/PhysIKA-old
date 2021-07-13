#pragma once

#include <unordered_map>

#include <core/math/matrix_csr.h>
#include <core/utils/parallel.h>
#include <core/utils/timer.h>

#if 0
namespace pbal {

    // CPU GPU hybrid

    const int kMinLevelSize = 16 * 16;

    class AmgpcgSolver2 {
        public:
            std::vector<MatrixCsrd> aP, rP, pP;
            std::vector<Size2> sP;
            int levelsNum = 0;

            bool solve(
                const Size2 size,
                const MatrixCsrd& A,
                double * dX, 
                const double * dB, 
                int maxIterations = 100, 
                double tolerence = 1e-6) {
            
                Timer timer;
                generatePyramids(size, A);
                std::cout << "pyramids : " << timer.durationInSeconds() << std::endl;

                int n = size.x * size.y;

                auto copyToHost = [&](double * dR, std::vector<double>& r) {
                    thrust::device_ptr<double> pR = thrust::device_pointer_cast(dR);
                    thrust::device_vector<double> dvR(n);
                    thrust::copy(pR, pR + n, dvR.begin());

                    thrust::host_vector<double> hvR = dvR;
                    r.resize(hvR.size());
                    parallelForEachIndex(hvR.size(), [&](int i) {
                        r[i] = hvR[i];
                    });
                };

                auto copyToDevice = [&](const std::vector<double>& r, double * dR) {
                    thrust::device_ptr<double> pR = thrust::device_pointer_cast(dR);
                    thrust::host_vector<double> hvR(r.size());
                    parallelForEachIndex(hvR.size(), [&](int i) {
                        hvR[i] = r[i];
                    });
                    thrust::device_vector<double> dvR = hvR;
                    thrust::copy(dvR.begin(), dvR.end(), pR);
                };

                timer.reset();
                bool flag = pcg<true>(n, 
                    [=](double * dR, double * dZ) {
                        std::vector<double> r, z;
                        copyToHost(dR, r);
                        precond(z, r);
                        copyToDevice(z, dZ);
                    },
                    [=](double * dX, double * dAx) {
                        std::vector<double> x, ax;
                        copyToHost(dX, x);
                        ax = A.mul(x);
                        copyToDevice(ax, dAx);
                    },
                    dX, dB,
                    maxIterations,
                    tolerence
                );
                std::cout << "pcg : " << timer.durationInSeconds() << std::endl;

                return flag;
            }


        private:

            void generatePyramids(Size2 size, MatrixCsrd A) {
                levelsNum = 1;
                aP.push_back(A);
                sP.push_back(size);
                while (size.x * size.y > kMinLevelSize) {
                    aP.emplace_back(MatrixCsrd());
                    rP.emplace_back(MatrixCsrd());
                    pP.emplace_back(MatrixCsrd());
                    size = Size2((size.x + 1) / 2, (size.y + 1) / 2);
                    sP.push_back(size);
                    levelsNum ++;
                }
                for (int iter = 0; iter < levelsNum - 1; iter ++) {
                    Size2 curSize = sP[iter];
                    Size2 downSize = sP[iter + 1];
                    // generate R P;
                    std::unordered_map<int, int> indexMap;
                    forEachIndex(curSize,
                        [&](int i, int j) {
                            int curIdx = i + curSize.x * j;
                            int downIdx = i / 2 + downSize.x * (j / 2);
                            pP[iter].setElement(curIdx, downIdx, 1.0);
                            rP[iter].setElement(downIdx, curIdx, 0.25);
                        }
                    );
                    // A = R * A * P;
                    aP[iter + 1] = rP[iter].mul(aP[iter].mul(pP[iter]));
                }
            }

            void rbgs(const MatrixCsrd& A,
                const std::vector<double>& b,
                std::vector<double>& x,
                const Size2& size,
                int iterationNum) {

                for (int iter = 0; iter < iterationNum; iter++) {
                    parallelForEachIndex(size, [&](int i, int j) {
                        if ((i + j) % 2 == 1) {
                            int index = i + size.x * j;

                            double sum = 0.0, diag = 0.0;
                            for (int ii = A.rowPointers[index]; ii < A.rowPointers[index + 1]; ii++) {
                                if (A.columnIndices[ii] != index) { // none diagonal terms
                                    sum += A.values[ii] * x[A.columnIndices[ii]];
                                }
                                else { // record diagonal value A(i,i)
                                    diag = A.values[ii];
                                }
                            } // A(i,:)*x for off-diag terms
                            if (diag != 0.0) {
                                x[index] = (b[index] - sum) / diag;
                            }
                            else {
                                x[index] = 0.0;
                            }
                        }
                    });
                    parallelForEachIndex(size, [&](int i, int j) {
                        if ((i + j) % 2 == 0) {
                            unsigned int index = i + size.x * j;

                            double sum = 0.0, diag = 0.0;
                            for (int ii = A.rowPointers[index]; ii < A.rowPointers[index + 1]; ii++) {
                                if (A.columnIndices[ii] != index) { // none diagonal terms
                                    sum += A.values[ii] * x[A.columnIndices[ii]];
                                }
                                else { // record diagonal value A(i,i)
                                    diag = A.values[ii];
                                }
                            } // A(i,:)*x for off-diag terms
                            if (diag != 0) {
                                x[index] = (b[index] - sum) / diag;
                            }
                            else {
                                x[index] = 0.0;
                            }
                        }
                    });
                }
            }

            void restriction(const MatrixCsrd& A,
                const MatrixCsrd& R,
                const std::vector<double> x,
                const std::vector<double> b,
                std::vector<double>& bDown) {
                // b = R(b - Ax)
                auto ax = A.mul(x);
                std::vector<double> r(ax.size(), 0.0);
                parallelForEachIndex(ax.size(), [&](int i) {
                    r[i] = b[i] - ax[i];
                });
                bDown = R.mul(r);
            }

            void prolongation(const MatrixCsrd& P,
                const std::vector<double> x,
                std::vector<double> xUp) {
                // x = x + Px
                auto px = P.mul(x);
                parallelForEachIndex(px.size(), [&](int i) {
                    xUp[i] = xUp[i] + px[i];
                });
            }
 
            void precond(std::vector<double>& x, const std::vector<double>& b) {
                std::vector<std::vector<double> > xP(levelsNum), bP(levelsNum);
                xP[0].resize(b.size(), 0.0);
                bP[0] = b;

                for (int iter = 1; iter < levelsNum; iter++) {
                    int n = sP[iter].x * sP[iter].y;
                    xP[iter].resize(n, 0.0);
                    bP[iter].resize(n, 0.0);
                }
                for (int iter = 0; iter < levelsNum - 1; iter ++) {
                    rbgs(aP[iter], bP[iter], xP[iter], sP[iter], 4);
                    restriction(aP[iter], rP[iter], xP[iter], bP[iter], bP[iter + 1]);
                }
                {
                    int iter = levelsNum - 1;
                    rbgs(aP[iter], bP[iter], xP[iter], sP[iter], 200);
                }
                for (int iter = levelsNum - 2; iter >= 0; iter --) {
                    prolongation(pP[iter], xP[iter + 1], xP[iter]);
                    rbgs(aP[iter], bP[iter], xP[iter], sP[iter], 4);
                }
                x = xP[0];
                for (int iter = 0; iter < levelsNum; iter++) {
                    xP[iter].resize(0);
                    xP[iter].shrink_to_fit();
                    bP[iter].resize(0);
                    bP[iter].shrink_to_fit();
                }
            }

    };

}

#endif