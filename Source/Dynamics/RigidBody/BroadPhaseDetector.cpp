#include "Dynamics/RigidBody/BroadPhaseDetector.h"
#include <set>
#include <algorithm>

namespace PhysIKA {
template <typename T>
void SortSweepDetector<T>::reset()
{
    this->m_n = 0;
    //this->m_sorted_x.clear();
    //this->m_sorted_y.clear();
    //this->m_sorted_z.clear();

    this->m_sorted_pair.clear();
}

template <typename T>
inline bool SortSweepDetector<T>::detect(const std::vector<BoxAABB3d<T>>& boxes, std::vector<std::pair<int, int>>& collision_pairs)
{

    //for (int axis = 0; axis < 3; ++axis)
    {
        int                             axis        = m_sort_axis;
        std::vector<std::pair<int, T>>& sorted_pair = m_sorted_pair;
        //std::vector<std::pair<int, T>>& sorted_pair = ( axis == 0 ? (m_sorted_x) : (axis == 1 ? m_sorted_y : m_sorted_z) );
        //std::vector<bool>& is_begin_value = (axis == 0 ? (m_is_begin_x) : (axis == 1 ? m_is_begin_y : m_is_begin_z));

        double t0 = clock() / 1000.0;

        // sort the beginning and end value of intervals
        if (m_n == 0)
        {
            this->m_n = boxes.size();

            sorted_pair.resize(boxes.size());
            for (int i = 0; i < boxes.size(); ++i)
            {
                sorted_pair[i] = std::make_pair(i, boxes[i].getl(axis));
            }

            // sort
            std::sort(sorted_pair.begin(), sorted_pair.end(), [](std::pair<int, T>& p1, std::pair<int, T>& p2) {
                return p1.second < p2.second;
            });
        }
        else
        {
            // update the interval value of in sorted_pair
            for (int i = 0; i < (boxes.size()); ++i)
            {
                int cur_id = sorted_pair[i].first;

                sorted_pair[i].second = boxes[cur_id].getl(axis);
            }

            // insertion sort
            for (int i = 1; i < (boxes.size()); ++i)
            {
                for (int j = i; j > 0; --j)
                {
                    if (sorted_pair[j].second < sorted_pair[j - 1].second)
                    {
                        auto tmp_pair      = sorted_pair[j];
                        sorted_pair[j]     = sorted_pair[j - 1];
                        sorted_pair[j - 1] = tmp_pair;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        double t1 = clock() / 1000.0;
        std::cout << "TIME insertion sort:   " << t1 - t0 << std::endl;

        // find intersections
        collision_pairs.clear();

        //std::vector<bool> is_active(boxes.size(), false);
        //std::set<int> active_id;

        int n_intersect_test = 0;
        for (int i = 0; i < (boxes.size()); ++i)
        {
            int cur_id = sorted_pair[i].first;

            for (int j = i + 1; j < boxes.size(); ++j)
            {
                int id_j = sorted_pair[j].first;
                if (boxes[cur_id].getu(axis) < boxes[id_j].getl(axis))
                {
                    break;
                }

                ++n_intersect_test;
                if (boxes[cur_id].isIntersect(boxes[id_j]))
                {
                    if (cur_id < id_j)
                    {
                        collision_pairs.push_back(std::make_pair(cur_id, id_j));
                    }
                    else
                    {
                        collision_pairs.push_back(std::make_pair(id_j, cur_id));
                    }
                }
            }

            ////if (is_beginning)
            //{
            //    // record all potential collision pair
            //    for (auto iter = active_id.begin(); iter != active_id.end(); ++iter)
            //    {
            //        int collide_to = (*iter);

            //        ++n_intersect_test;
            //        if (boxes[cur_id].isIntersect(boxes[collide_to]))
            //        {
            //            if (cur_id < collide_to)
            //            {
            //                collision_pairs.push_back(std::make_pair(cur_id, collide_to));
            //            }
            //            else
            //            {
            //                collision_pairs.push_back(std::make_pair(collide_to, cur_id));
            //            }
            //        }

            //    }

            //    //active_id.insert(cur_id);
            //}
            //else
            //{
            //    active_id.erase(cur_id);
            //}
        }
        std::cout << n_intersect_test << std::endl;

        double t2 = clock() / 1000.0;
        std::cout << "TIME find intersection:    " << t2 - t1 << std::endl;
    }

    return !collision_pairs.empty();
}

template <typename T>
void SortSweepDetector<T>::updateAxis(const std::vector<BoxAABB3d<T>>& boxes)
{
    // compute variance
    T sx = 0, sy = 0, sz = 0;
    T s2x = 0, s2y = 0, s2z = 0;

    for (int i = 0; i < boxes.size(); ++i)
    {
        T px = 0.5 * (boxes[i].getl(0) + boxes[i].getu(0));
        T py = 0.5 * (boxes[i].getl(1) + boxes[i].getu(1));
        T pz = 0.5 * (boxes[i].getl(2) + boxes[i].getu(2));

        sx += px;
        s2x += px * px;
        sy += py;
        s2y += py * py;
        sz += pz;
        s2z += pz * pz;
    }

    T variance[3];
    variance[0] = (s2x - (sx * sx) / boxes.size());
    variance[1] = (s2y - (sy * sy) / boxes.size());
    variance[2] = (s2z - (sz * sz) / boxes.size());

    int new_axis = m_sort_axis;
    if (variance[1] > variance[0])
    {
        new_axis = 1;
    }
    if (variance[2] > variance[new_axis])
    {
        new_axis = 2;
    }

    // update the array of sorted pairs
    if ((new_axis != m_sort_axis) || (m_n != boxes.size()))
    {
        m_sort_axis                                 = new_axis;
        m_n                                         = boxes.size();
        int                             axis        = m_sort_axis;
        std::vector<std::pair<int, T>>& sorted_pair = m_sorted_pair;

        // sort the beginning and end value of intervals
        {
            this->m_n = boxes.size();

            sorted_pair.resize(boxes.size());
            for (int i = 0; i < boxes.size(); ++i)
            {
                sorted_pair[i] = std::make_pair(i, boxes[i].getl(axis));
            }

            // sort
            std::sort(sorted_pair.begin(), sorted_pair.end(), [](std::pair<int, T>& p1, std::pair<int, T>& p2) {
                return p1.second < p2.second;
            });
        }
    }
}

}  // namespace PhysIKA