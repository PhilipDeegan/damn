#ifndef DAMN_HDF5_VTKHDF_HPP
#define DAMN_HDF5_VTKHDF_HPP

// VTK's OverlappingAMR in HDF5 (VTKHDF 2.2), as a time series: one file, a step per dump, each
// level's boxes and the point data on their primal nodes (as PHARE's pharevtkhdf diagnostics)
//
//   init(file, VtkInit{n_components, max_levels, spacing})   the groups, datasets, attributes
//                                     of every level up to max_levels (if not there yet)
//   append_step(file, time)           a step: its time, NSteps + 1
//   append_level(file, VtkLevel{ln, level, rank, n_components, data})   collective: level ln
//                                     at the last step: its boxes (rank major: every rank's
//                                     patches one after another, by rank), their offsets, and
//                                     each rank's rows at its place, one write per rank; a level
//                                     without patches (not there yet) appends an empty entry
//
//   points_of(box)                    a box's primal nodes, cells + 1 per direction
//   rows_of(box)                      its rows: its points, x_times<dim> times (1D, 2D: a 3D
//                                     view, the block repeated for each dummy direction's two
//                                     nodes)
//
// rows: points in Fortran order (x fastest), one column per component (vectors x y z, tensors
// xx xy xz yy yz zz), the caller's data in its patches' order (level order). AMRBox rows: the
// cells, lower and upper per direction (upper inclusive), 0 for directions past dim. Every rank
// calls every resize with the same sizes; rank 0 writes the steps' entries and the boxes

#include "damn/core/def.hpp"
#include "damn/hdf5/h5_file.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include "highfive/H5File.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <initializer_list>

namespace damn::amr
{
template<std::size_t dim>
struct Level;
} // namespace damn::amr
namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace damn::hdf5::vtkhdf
{
inline constexpr char base[]             = "/VTKHDF";
std::size_t inline constexpr steps_chunk = 1024, boxes_chunk = 1024;

// a 1D or 2D box as a 3D view: its block of points once per dummy direction node (2 each)
template<std::size_t dim>
std::size_t constexpr x_times = std::array<std::size_t, 3>{4, 2, 1}[dim - 1];

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::size_t points_of(core::Box<int, dim> const& cells)
{
    std::size_t n = 1;
    for (std::size_t i = 0; i < dim; ++i)
        n *= static_cast<std::size_t>(cells.upper[i] - cells.lower[i] + 2);
    return n;
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::size_t rows_of(core::Box<int, dim> const& cells)
{
    return points_of(cells) * x_times<dim>;
}

// what init needs: columns per point, levels the file holds, level 0's cell widths (a level's
// is level 0's / 2^level)
template<std::size_t dim>
struct VtkInit
{
    std::size_t n_components = 1;
    int max_levels           = 1;
    std::array<double, dim> spacing{};
    std::size_t chunk_bytes = 1 << 20; // PointData chunks
};

namespace detail
{
    MKN_KUL_FN_SIG_RET inline std::string level_path(int const ln)
    {
        return std::string{base} + "/Level" + std::to_string(ln);
    }
    MKN_KUL_FN_SIG_RET inline std::string step_level_path(int const ln)
    {
        return std::string{base} + "/Steps/Level" + std::to_string(ln);
    }

    template<typename T>
    auto resizable_1d(h5::HighFiveFile& file, std::string const& path)
    {
        return file.create_chunked_data_set<T>(
            path, std::vector<hsize_t>{steps_chunk},
            HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}));
    }

    template<typename T>
    auto resizable_2d(h5::HighFiveFile& file, std::string const& path, std::size_t const cols,
                      std::size_t const chunk_rows)
    {
        return file.create_chunked_data_set<T>(
            path, std::vector<hsize_t>{chunk_rows, cols},
            HighFive::DataSpace({0, cols}, {HighFive::DataSpace::UNLIMITED, cols}));
    }

    inline void group(h5::HighFiveFile& file, std::string const& path)
    {
        if (!file.exist(path))
            file.file().createGroup(path);
    }

    // value appended to the 1D dataset at path (collective resize, rank 0 writes)
    template<typename T>
    void append_1d(h5::HighFiveFile& file, std::string const& path, T const value)
    {
        auto ds        = file.getDataSet(path);
        auto const old = ds.getDimensions()[0];
        ds.resize({old + 1});
        if (mpi::rank() == 0)
            ds.select({old}, {1}).write(value);
    }

    template<typename T>
    void attribute(auto node, std::string const& key, T const& value)
    {
        if (!node.hasAttribute(key))
            node.createAttribute(key, value);
    }
} // namespace detail


// the file's groups, datasets and attributes, every level up to max_levels (collective, as
// HDF5's creations; what is there already is kept)
template<typename Float = float, std::size_t dim>
void init(h5::HighFiveFile& file, VtkInit<dim> const& in)
{
    using detail::attribute;
    detail::group(file, std::string{base});
    auto root = file.file().getGroup(std::string{base});
    attribute(root, "Version", std::vector<int>{2, 2});
    attribute(root, "Origin", std::vector<float>{0, 0, 0});
    attribute(root, "Type", std::string{"OverlappingAMR"});
    attribute(root, "GridDescription", std::string{"XYZ"});

    if (!file.exist(std::string{base} + "/Steps/Values"))
        detail::resizable_1d<Float>(file, std::string{base} + "/Steps/Values");
    attribute(file.file().getGroup(std::string{base} + "/Steps"), "NSteps", 0);

    auto const chunk_rows
        = std::max<std::size_t>(1, in.chunk_bytes / (in.n_components * sizeof(Float)));
    for (int ln = 0; ln < in.max_levels; ++ln)
    {
        auto const level = detail::level_path(ln), steps = detail::step_level_path(ln);
        detail::group(file, level);
        std::vector<float> spacing(dim);
        for (std::size_t i = 0; i < dim; ++i)
            spacing[i] = static_cast<float>(in.spacing[i] / std::pow(2., ln));
        attribute(file.file().getGroup(level), "Spacing", spacing);
        if (!file.exist(level + "/AMRBox"))
            detail::resizable_2d<int>(file, level + "/AMRBox", 6, boxes_chunk);
        if (!file.exist(level + "/PointData/data"))
            detail::resizable_2d<Float>(file, level + "/PointData/data", in.n_components,
                                        chunk_rows);
        detail::group(file, level + "/CellData");
        detail::group(file, level + "/FieldData");
        for (auto const* offsets : {"/AMRBoxOffset", "/NumberOfAMRBox", "/PointDataOffset/data"})
            if (!file.exist(steps + offsets))
                detail::resizable_1d<std::int64_t>(file, steps + offsets);
        detail::group(file, steps + "/CellDataOffset");
        detail::group(file, steps + "/FieldDataOffset");
    }
}

// the same from VtkInit's members as arguments
template<typename Float = float, std::size_t dim>
void init(h5::HighFiveFile& file, std::size_t const n_components, int const max_levels,
          std::array<double, dim> const& spacing, auto&&... args)
{
    init<Float>(file, VtkInit<dim>{n_components, max_levels, spacing,
                                   std::forward<decltype(args)>(args)...});
}

// a step at time (collective: rank 0 writes)
template<typename Float = float>
void append_step(h5::HighFiveFile& file, double const time)
{
    detail::append_1d(file, std::string{base} + "/Steps/Values", static_cast<Float>(time));
    auto nsteps = file.file().getGroup(std::string{base} + "/Steps").getAttribute("NSteps");
    nsteps.write(nsteps.template read<int>() + 1);
}

// level ln at the last step: level's patches (every rank's), this rank's rows in data (its
// patches in level order, rows_of(box) each, n_components columns)
template<typename Float, std::size_t dim>
struct VtkLevel
{
    int ln = 0;
    amr::Level<dim> const& level;
    int rank                 = 0;
    std::size_t n_components = 1;
    std::span<Float const> data;
};

template<typename Float, std::size_t dim>
void append_level(h5::HighFiveFile& file, VtkLevel<Float, dim> const& in)
{
    auto const& [ln, level, rank, n_components, data] = in;
    auto const level_path = detail::level_path(ln), steps = detail::step_level_path(ln);

    // rank major: by owner, each owner's patches in level order
    std::vector<std::size_t> order(level.patches.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto const a, auto const b) {
        return level.patches[a].rank < level.patches[b].rank;
    });
    std::uint64_t rows_before = 0, my_rows = 0, total_rows = 0;
    for (auto const k : order)
    {
        auto const rows = rows_of(level.patches[k].box);
        if (level.patches[k].rank < rank)
            rows_before += rows;
        if (level.patches[k].rank == rank)
            my_rows += rows;
        total_rows += rows;
    }
    if (data.size() != my_rows * n_components)
        throw std::runtime_error("damn::hdf5::vtkhdf::append_level: " + std::to_string(data.size())
                                 + " values for " + std::to_string(my_rows) + " rows of "
                                 + std::to_string(n_components));

    // boxes
    auto boxes           = file.getDataSet(level_path + "/AMRBox");
    auto const old_boxes = boxes.getDimensions()[0];
    detail::append_1d(file, steps + "/NumberOfAMRBox", static_cast<std::int64_t>(order.size()));
    detail::append_1d(file, steps + "/AMRBoxOffset", static_cast<std::int64_t>(old_boxes));
    boxes.resize({old_boxes + order.size(), 6});
    if (rank == 0 and !order.empty())
    {
        std::vector<std::array<int, 6>> rows(order.size(), std::array<int, 6>{});
        for (std::size_t r = 0; r < order.size(); ++r)
            for (std::size_t i = 0; i < dim; ++i)
            {
                rows[r][2 * i]     = level.patches[order[r]].box.lower[i];
                rows[r][2 * i + 1] = level.patches[order[r]].box.upper[i];
            }
        boxes.select({old_boxes, 0}, {order.size(), 6}).write(rows);
    }

    // points
    auto points         = file.getDataSet(level_path + "/PointData/data");
    auto const old_rows = points.getDimensions()[0];
    detail::append_1d(file, steps + "/PointDataOffset/data", static_cast<std::int64_t>(old_rows));
    points.resize({old_rows + total_rows, n_components});
    if (my_rows > 0)
        points.select({old_rows + rows_before, 0}, {my_rows, n_components}).write_raw(data.data());
}

// the same from VtkLevel's members as arguments
template<typename Float, std::size_t dim>
void append_level(h5::HighFiveFile& file, int const ln, amr::Level<dim> const& level,
                  auto&&... args)
{
    append_level(file, VtkLevel<Float, dim>{ln, level, std::forward<decltype(args)>(args)...});
}

} // namespace damn::hdf5::vtkhdf

#endif // DAMN_HDF5_VTKHDF_HPP
