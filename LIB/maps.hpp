#ifndef MAPS_HPP
#define MAPS_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>

[[deprecated("Use GridKey instead")]]
std::string get_key(const float* coor);
[[deprecated("Use GridKey instead")]]
void parse_key(std::string key, float* coor);

// Integer-based coordinate key for reproducible deduplication.
// Snaps coordinates to milliangstrom grid to avoid floating-point
// formatting differences across platforms.
struct GridKey {
    int ix, iy, iz;

    GridKey() : ix(0), iy(0), iz(0) {}
    GridKey(float x, float y, float z)
        : ix(snap(x)), iy(snap(y)), iz(snap(z)) {}
    explicit GridKey(const float* coor)
        : ix(snap(coor[0])), iy(snap(coor[1])), iz(snap(coor[2])) {}

    bool operator<(const GridKey& o) const {
        if (ix != o.ix) return ix < o.ix;
        if (iy != o.iy) return iy < o.iy;
        return iz < o.iz;
    }
    bool operator==(const GridKey& o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }

    void to_coor(float* coor) const {
        coor[0] = ix * 0.001f;
        coor[1] = iy * 0.001f;
        coor[2] = iz * 0.001f;
    }

private:
    static int snap(float v) {
        return static_cast<int>(std::round(v * 1000.0f));
    }
};

#endif // MAPS_HPP
