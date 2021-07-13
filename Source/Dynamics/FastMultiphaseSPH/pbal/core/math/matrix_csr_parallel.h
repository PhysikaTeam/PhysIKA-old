#pragma once

#include <algorithm>

#include <core/math/vec.h>
#include <core/utils/parallel.h>
#include <core/utils/utils.h>

namespace pbal {

template <typename T>
class MatrixCsr;

template <typename T>
class MatrixCsrParallel
{
private:
    friend class MatrixCsr<T>;

public:
    int                           rowSize;
    std::vector<std::vector<T>>   values;
    std::vector<std::vector<int>> columnIndices;

    MatrixCsrParallel()
    {
        clear();
    }

    MatrixCsrParallel(int n)
    {
        clear();
        rowSize = n;
        values.resize(n);
        columnIndices.resize(n);
    }

    MatrixCsrParallel(const MatrixCsrParallel& other)
    {
        set(other);
    }

    void clear()
    {
        rowSize = 0;
        values.clear();
        columnIndices.clear();
    }

    void set(const MatrixCsrParallel& other)
    {
        rowSize       = other.rowSize;
        values        = other.values;
        columnIndices = other.columnIndices;
    }

    void addRow(const std::vector<T>&   rowValues,
                const std::vector<int>& rowColumnIndices)
    {

        std::vector<std::pair<T, int>> zipped;
        for (int i = 0; i < rowValues.size(); ++i)
        {
            zipped.emplace_back(rowValues[i], rowColumnIndices[i]);
        }
        std::sort(zipped.begin(), zipped.end(), [](std::pair<T, int> a, std::pair<T, int> b) {
            return a.second < b.second;
        });
        values.emplace_back(std::vector<double>());
        columnIndices.emplace_back(std::vector<int>());
        for (int i = 0; i < zipped.size(); ++i)
        {
            values[rowSize].push_back(zipped[i].first);
            columnIndices[rowSize].push_back(zipped[i].second);
        }
        rowSize++;
    }

    int hasElement(int i, int j)
    {
        if (i >= rowSize)
        {
            return -1;
        }

        auto iter = binaryFind(columnIndices[i].begin(),
                               columnIndices[i].end(),
                               j);
        if (iter != columnIndices[i].end())
        {
            return static_cast<int>(iter - columnIndices[i].begin());
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
            values[i][nzIndex] += v;
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
            values[i][nzIndex] = v;
        }
    }

    void addElement(int i, int j, const T& v)
    {
        int numRowsToAdd = i - rowSize + 1;
        if (numRowsToAdd > 0)
        {
            for (int i = 0; i < numRowsToAdd; i++)
            {
                addRow({}, {});
            }
        }

        auto colIdxIter = std::lower_bound(
            columnIndices[i].begin(),
            columnIndices[i].end(),
            j);
        auto offset = colIdxIter - columnIndices[i].begin();

        columnIndices[i].insert(colIdxIter, j);
        values[i].insert(values[i].begin() + offset, v);
    }
};

using MatrixCsrParalleld = MatrixCsrParallel<double>;

}  // namespace pbal
