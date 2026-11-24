#ifndef DAMN_HDF5_LEVEL_TABLE_HPP
#define DAMN_HDF5_LEVEL_TABLE_HPP

// a level's patches' rows in one dataset, as VTKHDF does: the patches one after another in the
// level's patch order (every rank's), patch k's rows from offsets[k] to offsets[k + 1]
//
//   level_rows(level, rank, local_rows)       collective: LevelRows{level, rank, table}, the
//                                             table from the row count of each of this rank's
//                                             patches (in level order): every patch's offsets,
//                                             the same on every rank
//   write_level(file, group, rows)            the level's boxes, ids, owners and the offsets
//   read_level<dim>(file, group, number)      them back, on any rank (any number of ranks)
//   write_boxes(file, path, boxes)            a list of boxes the same on every rank (e.g. what
//   read_boxes<dim>(file, path)               a level was clustered to), and back
//   write_rows(file, path, rows, spans)       collective: the dataset is created at the total,
//                                             then each rank writes its patches' rows straight
//                                             from their memory, one span per local patch:
//                                             a StridedSpan (a member of an AoS, a column ...) or
//                                             a BoxRegion (a field's values on its domain, in C
//                                             order)
//   read_rows(file, path, offset, span)       span.size() rows from offset into memory, in place
//
// No staging copy: a StridedSpan's stride, or a BoxRegion's sub box, is the memory side's
// hyperslab. Rows and offsets
// are 64 bit (a 3d run's particles overflow 32 bits, as VTKHDF's offsets). Writes are
// independent (each rank its own rows); every rank creates every dataset, as HDF5 needs

#include "damn/amr/hierarchy.hpp"
#include "damn/core/def.hpp"
#include "mkn/kul/strided_span.hpp"
#include "damn/hdf5/h5_file.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "highfive/H5File.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>

namespace damn::core
{
template<typename T, std::size_t dim>
struct BoxRegion;
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace damn::hdf5
{
struct LevelTable
{
    std::vector<std::uint64_t> offsets{0}; // patches + 1, in level order

    MKN_KUL_FN_SIG_RET std::size_t patches() const { return offsets.size() - 1; }
    MKN_KUL_FN_SIG_RET std::uint64_t total() const { return offsets.back(); }
    MKN_KUL_FN_SIG_RET std::uint64_t offset(std::size_t const k) const { return offsets[k]; }
    MKN_KUL_FN_SIG_RET std::uint64_t rows(std::size_t const k) const
    {
        return offsets[k + 1] - offsets[k];
    }
};

namespace detail
{
    template<typename T>
    MKN_KUL_FN_SIG_RET hid_t native_type()
    {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, double>)
            return H5T_NATIVE_DOUBLE;
        else if constexpr (std::is_same_v<U, float>)
            return H5T_NATIVE_FLOAT;
        else if constexpr (std::is_same_v<U, int>)
            return H5T_NATIVE_INT;
        else if constexpr (std::is_same_v<U, std::uint32_t>)
            return H5T_NATIVE_UINT32;
        else if constexpr (std::is_same_v<U, std::int64_t>)
            return H5T_NATIVE_INT64;
        else if constexpr (std::is_same_v<U, std::uint64_t>)
            return H5T_NATIVE_UINT64;
        else
            static_assert(sizeof(U) == 0, "damn::hdf5: no native HDF5 type");
    }

    // the memory side of a span: n values a stride apart, in a space as long as they reach
    struct MemorySpace
    {
        hid_t id = -1;

        MemorySpace(std::size_t const n, std::size_t const stride)
        {
            hsize_t const extent = (n - 1) * stride + 1;
            id                   = H5Screate_simple(1, &extent, nullptr);
            hsize_t const start = 0, step = stride, count = n;
            H5Sselect_hyperslab(id, H5S_SELECT_SET, &start, &step, &count, nullptr);
        }

        // select, a sub box of a buffer over the points of buffer (C order)
        template<std::size_t dim>
        MemorySpace(core::Box<int, dim> const& buffer, core::Box<int, dim> const& select)
        {
            std::array<hsize_t, dim> extent{}, start{}, count{};
            for (std::size_t i = 0; i < dim; ++i)
            {
                extent[i] = static_cast<hsize_t>(buffer.upper[i] - buffer.lower[i] + 1);
                start[i]  = static_cast<hsize_t>(select.lower[i] - buffer.lower[i]);
                count[i]  = static_cast<hsize_t>(select.upper[i] - select.lower[i] + 1);
            }
            id = H5Screate_simple(dim, extent.data(), nullptr);
            H5Sselect_hyperslab(id, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);
        }
        ~MemorySpace() { H5Sclose(id); }
        MemorySpace(MemorySpace const&)            = delete;
        MemorySpace& operator=(MemorySpace const&) = delete;
    };

    // the file side: count rows from offset
    struct FileSpace
    {
        hid_t id = -1;

        FileSpace(hid_t const dataset, std::uint64_t const offset, std::size_t const count)
            : id{H5Dget_space(dataset)}
        {
            hsize_t const start = offset, n = count;
            H5Sselect_hyperslab(id, H5S_SELECT_SET, &start, nullptr, &n, nullptr);
        }
        ~FileSpace() { H5Sclose(id); }
        FileSpace(FileSpace const&)            = delete;
        FileSpace& operator=(FileSpace const&) = delete;
    };

    template<typename T>
    MKN_KUL_FN_SIG_RET MemorySpace memory_space(mkn::kul::StridedSpan<T> const& span)
    {
        return {span.size(), span.stride_values()};
    }
    template<typename T, std::size_t dim>
    MKN_KUL_FN_SIG_RET MemorySpace memory_space(core::BoxRegion<T, dim> const& region)
    {
        return {region.span.box, region.select};
    }
    template<typename T>
    MKN_KUL_FN_SIG_RET T* memory_data(mkn::kul::StridedSpan<T> const& span)
    {
        return span.data();
    }
    template<typename T, std::size_t dim>
    MKN_KUL_FN_SIG_RET T* memory_data(core::BoxRegion<T, dim> const& region)
    {
        return region.span.data();
    }

    template<typename Span>
    void write_span(HighFive::DataSet const& ds, std::uint64_t const offset, Span const& span)
    {
        using T = typename Span::value_type;
        if (span.size() == 0)
            return;
        auto const memory = memory_space(span);
        FileSpace const file{ds.getId(), offset, span.size()};
        if (H5Dwrite(ds.getId(), native_type<T>(), memory.id, file.id, H5P_DEFAULT,
                     memory_data(span))
            < 0)
            throw std::runtime_error("damn::hdf5::write_rows: H5Dwrite failed");
    }

    template<typename Span>
    void read_span(HighFive::DataSet const& ds, std::uint64_t const offset, Span const& span)
    {
        using T = typename Span::value_type;
        static_assert(!std::is_const_v<T>);
        if (span.size() == 0)
            return;
        auto const memory = memory_space(span);
        FileSpace const file{ds.getId(), offset, span.size()};
        if (H5Dread(ds.getId(), native_type<T>(), memory.id, file.id, H5P_DEFAULT,
                    memory_data(span))
            < 0)
            throw std::runtime_error("damn::hdf5::read_rows: H5Dread failed");
    }

    // a dataset every rank creates, rank 0 fills
    template<typename T>
    void write_on_rank_0(h5::HighFiveFile& file, std::string const& path,
                         std::vector<T> const& values)
    {
        auto const ds = file.create_data_set<T>(path, values.size());
        if (mpi::rank() == 0)
            write_span(ds, 0,
                       mkn::kul::StridedSpan<T const>{values.data(), values.size(), sizeof(T)});
    }

    template<typename T>
    MKN_KUL_FN_SIG_RET std::vector<T> read_all(h5::HighFiveFile& file, std::string const& path)
    {
        auto const ds = file.getDataSet(path);
        std::vector<T> values(ds.getElementCount());
        read_span(ds, 0, mkn::kul::StridedSpan<T>{values.data(), values.size(), sizeof(T)});
        return values;
    }
} // namespace detail


// a level, this rank and where every patch's rows are: what writing a level's datasets needs
template<std::size_t dim>
struct LevelRows
{
    amr::Level<dim> const& level;
    int rank = 0;
    LevelTable table;
};

// collective: every patch's offsets from the row counts of this rank's patches, in level order
template<std::size_t dim>
MKN_KUL_FN_SIG_RET LevelRows<dim> level_rows(amr::Level<dim> const& level, int const rank,
                                             std::vector<std::uint64_t> const& local_rows)
{
    auto const all = mpi::collect(local_rows); // per rank, in its level order
    std::vector<std::size_t> next(all.size(), 0);

    LevelTable table;
    table.offsets.reserve(level.patches.size() + 1);
    for (auto const& patch : level.patches)
    {
        auto const r = static_cast<std::size_t>(patch.rank);
        if (r >= all.size() or next[r] >= all[r].size())
            throw std::runtime_error("damn::hdf5::level_table: fewer rows than patches on rank "
                                     + std::to_string(patch.rank));
        table.offsets.push_back(table.offsets.back() + all[r][next[r]++]);
    }
    if (next[static_cast<std::size_t>(rank)] != local_rows.size())
        throw std::runtime_error("damn::hdf5::level_table: more rows than local patches");
    return {level, rank, std::move(table)};
}

namespace detail
{
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::vector<int> flat_boxes(std::vector<core::Box<int, dim>> const& boxes)
    {
        std::vector<int> flat;
        flat.reserve(2 * dim * boxes.size());
        for (auto const& b : boxes)
        {
            for (std::size_t i = 0; i < dim; ++i)
                flat.push_back(b.lower[i]);
            for (std::size_t i = 0; i < dim; ++i)
                flat.push_back(b.upper[i]);
        }
        return flat;
    }

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::vector<core::Box<int, dim>> boxes_of(std::vector<int> const& flat)
    {
        if (flat.size() % (2 * dim) != 0)
            throw std::runtime_error("damn::hdf5: not a list of boxes");
        std::vector<core::Box<int, dim>> boxes(flat.size() / (2 * dim));
        for (std::size_t k = 0; k < boxes.size(); ++k)
            for (std::size_t i = 0; i < dim; ++i)
            {
                boxes[k].lower[i] = flat[2 * dim * k + i];
                boxes[k].upper[i] = flat[2 * dim * k + dim + i];
            }
        return boxes;
    }
} // namespace detail

// boxes, the same on every rank (collective: every rank creates, rank 0 writes)
template<std::size_t dim>
void write_boxes(h5::HighFiveFile& file, std::string const& path,
                 std::vector<core::Box<int, dim>> const& boxes)
{
    detail::write_on_rank_0(file, path, detail::flat_boxes(boxes));
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::vector<core::Box<int, dim>> read_boxes(h5::HighFiveFile& file,
                                                               std::string const& path)
{
    return detail::boxes_of<dim>(detail::read_all<int>(file, path));
}

// group/boxes (patches x 2 dim: lower then upper), group/ids, group/ranks, group/offsets
template<std::size_t dim>
void write_level(h5::HighFiveFile& file, std::string const& group, LevelRows<dim> const& rows)
{
    auto const& [level, rank, table] = rows;
    if (table.patches() != level.patches.size())
        throw std::runtime_error("damn::hdf5::write_level: a table of another level");
    std::vector<core::Box<int, dim>> boxes;
    std::vector<int> ranks;
    std::vector<std::uint32_t> ids;
    for (auto const& p : level.patches)
    {
        boxes.push_back(p.box);
        ids.push_back(p.id);
        ranks.push_back(p.rank);
    }
    write_boxes(file, group + "/boxes", boxes);
    detail::write_on_rank_0(file, group + "/ids", ids);
    detail::write_on_rank_0(file, group + "/ranks", ranks);
    detail::write_on_rank_0(file, group + "/offsets", table.offsets);
}

template<std::size_t dim>
struct StoredLevel
{
    amr::Level<dim> level; // the writer's patches and owners
    LevelTable table;
};

template<std::size_t dim>
MKN_KUL_FN_SIG_RET StoredLevel<dim> read_level(h5::HighFiveFile& file, std::string const& group,
                                               int const number)
{
    auto const boxes = detail::read_all<int>(file, group + "/boxes");
    auto const ids   = detail::read_all<std::uint32_t>(file, group + "/ids");
    auto const ranks = detail::read_all<int>(file, group + "/ranks");

    StoredLevel<dim> stored;
    stored.table.offsets = detail::read_all<std::uint64_t>(file, group + "/offsets");
    if (boxes.size() != 2 * dim * ids.size() or ranks.size() != ids.size()
        or stored.table.patches() != ids.size())
        throw std::runtime_error("damn::hdf5::read_level: inconsistent level in " + group);

    stored.level.number = number;
    for (std::size_t k = 0; k < ids.size(); ++k)
    {
        amr::Patch<dim> p;
        for (std::size_t i = 0; i < dim; ++i)
        {
            p.box.lower[i] = boxes[2 * dim * k + i];
            p.box.upper[i] = boxes[2 * dim * k + dim + i];
        }
        p.id   = ids[k];
        p.rank = ranks[k];
        stored.level.patches.push_back(p);
    }
    return stored;
}

// collective: path holds rows.table.total() rows; each rank writes its patches' rows from
// spans, one per local patch in level order
template<typename Span, std::size_t dim>
void write_rows(h5::HighFiveFile& file, std::string const& path, LevelRows<dim> const& rows,
                std::vector<Span> const& spans)
{
    using T                          = std::remove_cv_t<typename Span::value_type>;
    auto const& [level, rank, table] = rows;
    auto const ds                    = file.create_data_set<T>(path, table.total());
    std::size_t local                = 0;
    for (std::size_t k = 0; k < level.patches.size(); ++k)
    {
        if (level.patches[k].rank != rank)
            continue;
        if (local >= spans.size())
            throw std::runtime_error("damn::hdf5::write_rows: fewer spans than local patches");
        auto const& span = spans[local++];
        if (span.size() != table.rows(k))
            throw std::runtime_error("damn::hdf5::write_rows: " + path + ": a span of "
                                     + std::to_string(span.size()) + " for "
                                     + std::to_string(table.rows(k)) + " rows");
        detail::write_span(ds, table.offset(k), span);
    }
    if (local != spans.size())
        throw std::runtime_error("damn::hdf5::write_rows: more spans than local patches");
}

// span.size() rows of path from offset, into the span's memory
template<typename Span>
void read_rows(h5::HighFiveFile& file, std::string const& path, std::uint64_t const offset,
               Span const& span)
{
    auto const ds = file.getDataSet(path);
    if (offset + span.size() > ds.getElementCount())
        throw std::runtime_error("damn::hdf5::read_rows: " + path + ": past its end");
    detail::read_span(ds, offset, span);
}

} // namespace damn::hdf5

#endif // DAMN_HDF5_LEVEL_TABLE_HPP
