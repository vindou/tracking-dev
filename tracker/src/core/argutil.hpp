// Small helpers for the command-line tools.
#pragma once
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace trk {
inline std::vector<double> parse_list(const std::string& s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::atof(item.c_str()));
    return v;
}
}  // namespace trk
