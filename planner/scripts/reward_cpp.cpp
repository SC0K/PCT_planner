#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_set>
#include <tuple>
#include <functional>

// Custom hash for tuple<int,int,int>
namespace std {
    template <>
    struct hash<std::tuple<int, int, int>> {
        std::size_t operator()(const std::tuple<int, int, int>& k) const {
            return std::get<0>(k) ^ (std::get<1>(k) << 8) ^ (std::get<2>(k) << 16);
        }
    };
}

namespace py = pybind11;

py::tuple batched_reward(
    py::array_t<int> visible_hits,  // (n, n_hits, 3)
    py::array_t<int> hit_flags,     // (n, n_hits)
    py::array_t<bool> explored,     // (gx, gy, gz)
    py::array_t<int> min_idx        // (3,)
) {
    auto vh = visible_hits.unchecked<3>();
    auto hf = hit_flags.unchecked<2>();
    auto ex = explored.unchecked<3>();
    auto mi = min_idx.unchecked<1>();
    ssize_t n = vh.shape(0);
    py::list rewards;
    py::list visible_voxels_list;
    for (ssize_t i = 0; i < n; ++i) {
        std::unordered_set<std::tuple<int,int,int>> s;
        for (ssize_t j = 0; j < vh.shape(1); ++j) {
            if (hf(i, j)) {
                int x = vh(i, j, 0) + mi(0);
                int y = vh(i, j, 1) + mi(1);
                int z = vh(i, j, 2) + mi(2);
                s.emplace(x, y, z);
            }
        }
        int reward = 0;
        py::set vis_set;
        for (const auto& v : s) {
            int x, y, z;
            std::tie(x, y, z) = v;
            if (!ex(x - mi(0), y - mi(1), z - mi(2))) reward++;
            vis_set.add(py::make_tuple(x, y, z));
        }
        rewards.append(reward);
        visible_voxels_list.append(vis_set);
    }
    return py::make_tuple(rewards, visible_voxels_list);
}

PYBIND11_MODULE(reward_cpp, m) {
    m.def("batched_reward", &batched_reward);
}