// Triangle mesh (binary STL, meters) and the model file of T1 (points with normals).
#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace trk {

struct Mesh {
    std::vector<Eigen::Vector3d> v0, v1, v2;   // triangle corners (object frame, m)
    Eigen::Vector3d bmin = Eigen::Vector3d::Zero(), bmax = Eigen::Vector3d::Zero();
    std::size_t size() const { return v0.size(); }
};
bool load_stl(const std::string& path, Mesh& m, std::string& err);

struct Model {
    std::vector<Eigen::Vector3d> points, normals;   // object frame
    Eigen::Vector3d bmin = Eigen::Vector3d::Zero(), bmax = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();   // of the surface points
    double radius = 0;                                     // max distance of a point from the centroid
};
bool load_model(const std::string& path, Model& m, std::string& err);

}  // namespace trk
