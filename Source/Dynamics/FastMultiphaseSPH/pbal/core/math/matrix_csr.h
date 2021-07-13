#pragma once

#include <algorithm>

#include <core/math/matrix_csr_parallel.h>
#include <core/math/vec.h>
#include <core/utils/utils.h>

namespace pbal {

template <typename T>
class MatrixCsr
{
public:
    Size2            size;
    std::vector<T>   values;
    std::vector<int> rowPointers;
    std::vector<int> columnIndices;

    MatrixCsr()
    {
        clear();
    }

    MatrixCsr(const MatrixCsr& other)
    {
        set(other);
    }

    void clear()
    {
        size = Size2();
        values.clear();
        rowPointers.clear();
        columnIndices.clear();
        rowPointers.push_back(0);
    }

    void set(const MatrixCsr& other)
    {
        size          = other.size;
        values        = other.values;
        rowPointers   = other.rowPointers;
        columnIndices = other.columnIndices;
    }

    MatrixCsr(const MatrixCsrParallel<T>& other)
    {
        clear();
        for (int i = 0; i < other.rowSize; i++)
        {
            addRow(other.values[i], other.columnIndices[i]);
        }
    }

    MatrixCsr<T> mul(const MatrixCsr<T>& m) const
    {
        MatrixCsrParallel<T> ret(size.x);
        parallelForEachIndex(size.x,
                             [&](int i) {
                                 for (int j = rowPointers[i]; j < rowPointers[i + 1]; j++)
                                 {
                                     int k = columnIndices[j];
                                     T   a = values[j];
                                     for (int kk = m.rowPointers[k]; kk < m.rowPointers[k + 1]; kk++)
                                     {
                                         T b = m.values[kk];
                                         ret.incElement(i, m.columnIndices[kk], a * b);
                                     }
                                 }
                             });
        return MatrixCsr<T>(ret);
    }

    std::vector<T> mul(const std::vector<T>& v) const
    {
        std::vector<T> ret(size.x, T());
        parallelForEachIndex(size.x,
                             [&](int i) {
                                 T sum = T();
                                 for (int j = rowPointers[i]; j < rowPointers[i + 1]; j++)
                                 {
                                     sum += values[j] * v[columnIndices[j]];
                                 }
                                 ret[i] = sum;
                             });
        return ret;
    }

    void addRow(const std::vector<T>&   rowValues,
                const std::vector<int>& rowColumnIndices)
    {
        ++size.x;

        std::vector<std::pair<T, int>> zipped;
        for (int i = 0; i < rowValues.size(); ++i)
        {
            zipped.emplace_back(rowValues[i], rowColumnIndices[i]);
            size.y = std::max(size.y, rowColumnIndices[i] + 1);
        }
        std::sort(zipped.begin(), zipped.end(), [](std::pair<T, int> a, std::pair<T, int> b) {
            return a.second < b.second;
        });
        for (int i = 0; i < zipped.size(); ++i)
        {
            values.push_back(zipped[i].first);
            columnIndices.push_back(zipped[i].second);
        }

        rowPointers.push_back(values.size());
    }

    int hasElement(int i, int j)
    {
        if (i >= size.x || j >= size.y)
        {
            return -1;
        }

        int rowBegin = rowPointers[i];
        int rowEnd   = rowPointers[i + 1];

        auto iter = binaryFind(columnIndices.begin() + rowBegin,
                               columnIndices.begin() + rowEnd,
                               j);
        if (iter != columnIndices.begin() + rowEnd)
        {
            return static_cast<int>(iter - columnIndices.begin());
        }
        else
        {
            return -1;
        }
    }

    void incElement(int i, int j, const T& v)
    {
        int nzIndex = hasElement(i, j);
        if (nzIndex == -1)
        {
            addElement(i, j, v);
        }
        else
        {
            values[nzIndex] += v;
        }
    }

    void setElement(int i, int j, const T& v)
    {
        int nzIndex = hasElement(i, j);
        if (nzIndex == -1)
        {
            addElement(i, j, v);
        }
        else
        {
            values[nzIndex] = v;
        }
    }

    void addElement(int i, int j, const T& v)
    {
        int numRowsToAdd = i - size.x + 1;
        if (numRowsToAdd > 0)
        {
            for (int i = 0; i < numRowsToAdd; i++)
            {
                addRow({}, {});
            }
        }
        size.y = std::max(size.y, j + 1);

        int rowBegin = rowPointers[i];
        int rowEnd   = rowPointers[i + 1];

        auto colIdxIter = std::lower_bound(
            columnIndices.begin() + rowBegin,
            columnIndices.begin() + rowEnd,
            j);
        auto offset = colIdxIter - columnIndices.begin();

        columnIndices.insert(colIdxIter, j);
        values.insert(values.begin() + offset, v);

        for (int idx = i + 1; idx < rowPointers.size(); ++idx)
        {
            ++rowPointers[idx];
        }
    }
};

using MatrixCsrd = MatrixCsr<double>;

}  // namespace pbal
