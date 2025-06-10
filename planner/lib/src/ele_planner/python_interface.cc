#include "ele_planner/offline_ele_planner.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

PYBIND11_MODULE(ele_planner, m) {
  auto pyOfflineElePlanner =
      py::class_<OfflineElePlanner>(m, "OfflineElePlanner");
  pyOfflineElePlanner
      .def(py::init<double, bool>(), py::arg("max_heading_rate"),
           py::arg("use_quintic") = false)
      .def("init_map", &OfflineElePlanner::InitMap)
      .def("reinit_map", &OfflineElePlanner::ReInitMap,
           py::arg("a_start_cost_threshold"), py::arg("safe_cost_margin"),
           py::arg("resolution"), py::arg("num_layers"),
           py::arg("step_cost_weight"), py::arg("cost_map"),
           py::arg("height_map"), py::arg("ceiling"), py::arg("ele_map"),
           py::arg("grad_x"), py::arg("grad_y"))
      .def("init_planner", &OfflineElePlanner::InitPlanner)
      .def("plan", &OfflineElePlanner::Plan)
      .def("debug", &OfflineElePlanner::Debug)
      .def("set_reference_height", &OfflineElePlanner::SetReferenceHeight)
      .def("set_max_iterations", &OfflineElePlanner::set_max_iterations)
      .def("get_path_finder", &OfflineElePlanner::get_path_finder)
      .def("get_map", &OfflineElePlanner::get_map)
      .def("get_trajectory_optimizer",
           &OfflineElePlanner::get_trajectory_optimizer)
      .def("get_trajectory_optimizer_wnoj",
           &OfflineElePlanner::get_trajectory_optimizer_wnoj)
      .def("get_debug_path", &OfflineElePlanner::GetDebugPath);
}