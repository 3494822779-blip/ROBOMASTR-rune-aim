#pragma once

#include "core/camera.hpp"
#include "core/types.hpp"

#include <array>

namespace rmcs::util {

struct SingleRunePnpSolution {
    struct Input {
        CameraFeature cam;

        Point2d center;
        Point2d icon;
        std::array<Point2d, 4> corners;
    } input;

    struct Result {
        Translation translation;
        Orientation orientation;
    } result;

    auto solve() -> bool;
};

}
