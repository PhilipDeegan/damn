#ifndef DAMN_HDF5_CHECKPOINT_HPP
#define DAMN_HDF5_CHECKPOINT_HPP

// a level's resources, by name, in one HDF5 group, through level tables: doc/hdf5/checkpoint.md
// every rank calls every collective, patches or not: a rank without patches learns each
// resource's kind and columns from the others

#include "damn/amr/box_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/memory/box_span.hpp"
#include "mkn/kul/strided_span.hpp"
#include "damn/hdf5/level_table.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <concepts>

namespace damn::amr
{
template<std::size_t dim>
struct Level;
} // namespace damn::amr
namespace damn::hdf5::h5
{
class HighFiveFile;
} // namespace damn::hdf5::h5

namespace damn::hdf5
{
template<typename R, std::size_t dim>
concept HasBoxSpan = amr::BoxData<R, dim> and requires(R& r, core::Box<int, dim> const& box) {
    { box_span(r, box).box } -> std::convertible_to<core::Box<int, dim>>;
};

template<typename R, std::size_t dim>
concept HasCheckpointBox = requires(R const& r, core::Box<int, dim> const& box) {
    { checkpoint_box(r, box) } -> std::convertible_to<core::Box<int, dim>>;
};

// the points of a BoxData resource a checkpoint holds
template<std::size_t dim>
MKN_KUL_FN_SIG_RET auto stored_box(auto const& resource, core::Box<int, dim> const& cells)
{
    if constexpr (HasCheckpointBox<std::decay_t<decltype(resource)>, dim>)
        return core::Box<int, dim>{checkpoint_box(resource, cells)};
    else
        return core::Box<int, dim>{domain_box(resource, cells)};
}

template<typename R>
concept HasColumns = requires(R& r, std::size_t const n) {
    for_each_column(r, [](std::string_view, auto const&) {});
    resize_columns(r, n);
};

template<std::size_t dim, typename PatchData>
struct LevelCheckpoint
{
    h5::HighFiveFile& file;
    std::string group;
    amr::Level<dim> const& level;
    amr::LevelData<dim, PatchData>& data;
    std::vector<std::string> const& names;
};

namespace detail
{
    // a column's type, to tell a rank without patches what to create
    template<typename T>
    MKN_KUL_FN_SIG_RET char type_code()
    {
        using U = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<U, double>)
            return 'd';
        else if constexpr (std::is_same_v<U, float>)
            return 'f';
        else if constexpr (std::is_same_v<U, int>)
            return 'i';
        else if constexpr (std::is_same_v<U, std::uint32_t>)
            return 'u';
        else if constexpr (std::is_same_v<U, std::int64_t>)
            return 'l';
        else if constexpr (std::is_same_v<U, std::uint64_t>)
            return 'L';
        else
            static_assert(sizeof(U) == 0, "damn::hdf5::checkpoint: no type code");
    }

    // fn.template operator()<T>() for the type of code
    void with_type(char const code, auto&& fn)
    {
        switch (code)
        {
            case 'd': return fn.template operator()<double>();
            case 'f': return fn.template operator()<float>();
            case 'i': return fn.template operator()<int>();
            case 'u': return fn.template operator()<std::uint32_t>();
            case 'l': return fn.template operator()<std::int64_t>();
            case 'L': return fn.template operator()<std::uint64_t>();
            default:
                throw std::runtime_error(std::string{"damn::hdf5::checkpoint: no type for "}
                                         + code);
        }
    }

    // what a resource is written as: "b<type>" (BoxData values), or "c" then "<type><name>;"
    //  per column
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::string schema_of(auto& resource)
    {
        using R = std::decay_t<decltype(resource)>;
        if constexpr (HasColumns<R>)
        {
            std::string s = "c";
            for_each_column(resource, [&](std::string_view const name, auto const& span) {
                s += type_code<typename std::decay_t<decltype(span)>::value_type>();
                s += std::string{name} + ";";
            });
            return s;
        }
        else if constexpr (amr::BoxData<R, dim>)
            return std::string{"b"} + type_code<amr::box_data_value_t<R, dim>>();
        else
            static_assert(sizeof(R) == 0, "damn::hdf5::checkpoint: neither BoxData nor columns");
    }

    struct Column
    {
        char type;
        std::string name;
    };

    MKN_KUL_FN_SIG_RET inline std::vector<Column> columns_of(std::string const& schema)
    {
        std::vector<Column> out;
        for (std::size_t i = 1; i < schema.size();)
        {
            auto const end = schema.find(';', i);
            out.push_back({schema[i], schema.substr(i + 1, end - i - 1)});
            i = end + 1;
        }
        return out;
    }

    // the same schema on every rank that has the resource, for every rank
    MKN_KUL_FN_SIG_RET inline std::string agreed(std::string const& local, std::string const& name)
    {
        std::string out;
        for (auto const& s : mpi::collect(local))
        {
            if (s.empty())
                continue;
            if (!out.empty() and s != out)
                throw std::runtime_error("damn::hdf5::checkpoint: " + name
                                         + " differs between ranks");
            out = s;
        }
        if (out.empty())
            throw std::runtime_error("damn::hdf5::checkpoint: no rank has " + name);
        return out;
    }

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::uint64_t rows_of(auto& resource, core::Box<int, dim> const& cells)
    {
        using R = std::decay_t<decltype(resource)>;
        if constexpr (HasColumns<R>)
        {
            std::uint64_t n = 0;
            bool first      = true;
            for_each_column(resource, [&](std::string_view, auto const& span) {
                if (!first and span.size() != n)
                    throw std::runtime_error("damn::hdf5::checkpoint: columns of different sizes");
                n     = span.size();
                first = false;
            });
            return n;
        }
        else
            return stored_box(resource, cells).size();
    }

    // a resource's values on its domain, in C order, for one without a box_span
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET auto staged(auto const& resource, core::Box<int, dim> const& cells)
    {
        using T = amr::box_data_value_t<std::decay_t<decltype(resource)>, dim>;
        std::vector<T> out;
        out.reserve(stored_box(resource, cells).size());
        core::for_each_point(stored_box(resource, cells),
                             [&](auto const& p) { out.push_back(value(resource, cells, p)); });
        return out;
    }

    // fn(k, local index, patch, resource) for each local patch holding name
    template<std::size_t dim, typename PatchData>
    void for_local(LevelCheckpoint<dim, PatchData> const& c, std::string const& name, auto&& fn)
    {
        std::size_t local = 0;
        for (std::size_t k = 0; k < c.level.patches.size(); ++k)
        {
            if (c.level.patches[k].rank != mpi::rank())
                continue;
            auto const i = local++;
            if (i >= c.data.data.size())
                throw std::runtime_error("damn::hdf5::checkpoint: fewer patch datas than patches");
            if (!amr::with_resource(c.data.data[i], name, [&](auto& r) { fn(k, i, r); }))
                throw std::runtime_error("damn::hdf5::checkpoint: no resource " + name);
        }
    }
} // namespace detail


template<std::size_t dim, typename PatchData>
void write_checkpoint(LevelCheckpoint<dim, PatchData> const& c)
{
    auto const rank  = mpi::rank();
    auto const cells = [&](std::size_t const k) { return c.level.patches[k].box; };

    { // the level itself, its tables' rows: cells
        std::vector<std::uint64_t> rows;
        for (auto const& p : c.level.patches)
            if (p.rank == rank)
                rows.push_back(p.box.size());
        write_level(c.file, c.group + "/level", level_rows(c.level, rank, rows));
    }

    for (auto const& name : c.names)
    {
        std::string local_schema;
        std::vector<std::uint64_t> local_rows;
        detail::for_local(c, name, [&](auto const k, auto, auto& r) {
            local_schema = detail::schema_of<dim>(r);
            local_rows.push_back(detail::rows_of(r, cells(k)));
        });
        auto const schema = detail::agreed(local_schema, name);
        auto const rows   = level_rows(c.level, rank, local_rows);
        auto const path   = c.group + "/" + name;
        detail::write_on_rank_0(c.file, path + "/offsets", rows.table.offsets);

        if (schema[0] == 'b')
            detail::with_type(schema[1], [&]<typename T>() {
                std::vector<std::vector<T>> stage; // for resources without a box_span
                std::vector<core::BoxRegion<T const, dim>> regions;
                detail::for_local(c, name, [&](auto const k, auto, auto& r) {
                    using R = std::decay_t<decltype(r)>;
                    if constexpr (amr::BoxData<R, dim> and not HasColumns<R>)
                        if constexpr (std::is_same_v<amr::box_data_value_t<R, dim>, T>)
                        {
                            auto const domain = stored_box(r, cells(k));
                            if constexpr (HasBoxSpan<R, dim>)
                            {
                                core::BoxSpan<T const, dim> const span = box_span(r, cells(k));
                                regions.push_back(core::region(span, domain));
                            }
                            else
                            {
                                auto& s = stage.emplace_back(detail::staged(r, cells(k)));
                                regions.push_back({{s.data(), domain}, domain});
                            }
                        }
                });
                write_rows(c.file, path + "/values", rows, regions);
            });
        else
            for (auto const& column : detail::columns_of(schema))
                detail::with_type(column.type, [&]<typename T>() {
                    std::vector<mkn::kul::StridedSpan<T const>> spans;
                    detail::for_local(c, name, [&](auto, auto, auto& r) {
                        if constexpr (HasColumns<std::decay_t<decltype(r)>>)
                            for_each_column(r, [&](std::string_view const n, auto const& span) {
                                using S = std::decay_t<decltype(span)>;
                                if constexpr (std::is_same_v<
                                                  std::remove_cv_t<typename S::value_type>, T>)
                                    if (n == column.name)
                                        spans.push_back(span);
                            });
                    });
                    write_rows(c.file, path + "/" + column.name, rows, spans);
                });
    }
}

// the same from LevelCheckpoint's members as arguments
void write_checkpoint(h5::HighFiveFile& file, auto&&... args)
{
    write_checkpoint(LevelCheckpoint{file, std::forward<decltype(args)>(args)...});
}

// into c.data, allocated on the same patches as written (same ranks): BoxData on their domain,
// columns resized to their rows first
template<std::size_t dim, typename PatchData>
void read_checkpoint(LevelCheckpoint<dim, PatchData> const& c)
{
    auto const stored = read_level<dim>(c.file, c.group + "/level", c.level.number);
    if (stored.level.patches != c.level.patches)
        throw std::runtime_error("damn::hdf5::read_checkpoint: " + c.group
                                 + " was written for other patches");
    auto const cells = [&](std::size_t const k) { return c.level.patches[k].box; };

    for (auto const& name : c.names)
    {
        auto const path = c.group + "/" + name;
        LevelTable table;
        table.offsets = detail::read_all<std::uint64_t>(c.file, path + "/offsets");
        if (table.patches() != c.level.patches.size())
            throw std::runtime_error("damn::hdf5::read_checkpoint: " + path + ": other patches");

        detail::for_local(c, name, [&](auto const k, auto, auto& r) {
            using R = std::decay_t<decltype(r)>;
            if constexpr (HasColumns<R>)
            {
                resize_columns(r, table.rows(k));
                for_each_column(r, [&](std::string_view const n, auto const& span) {
                    read_rows(c.file, path + "/" + std::string{n}, table.offset(k), span);
                });
            }
            else if constexpr (amr::BoxData<R, dim>)
            {
                auto const domain = stored_box(r, cells(k));
                if (domain.size() != table.rows(k))
                    throw std::runtime_error("damn::hdf5::read_checkpoint: " + path
                                             + ": a domain of another size");
                if constexpr (HasBoxSpan<R, dim>)
                    read_rows(c.file, path + "/values", table.offset(k),
                              core::region(box_span(r, cells(k)), domain));
                else
                {
                    using T = amr::box_data_value_t<R, dim>;
                    std::vector<T> s(domain.size());
                    read_rows(c.file, path + "/values", table.offset(k),
                              mkn::kul::strided_span(s.data(), s.size(), sizeof(T)));
                    std::size_t i = 0;
                    core::for_each_point(domain,
                                         [&](auto const& p) { value(r, cells(k), p) = s[i++]; });
                }
            }
        });
    }
}

// the same from LevelCheckpoint's members as arguments
void read_checkpoint(h5::HighFiveFile& file, auto&&... args)
{
    read_checkpoint(LevelCheckpoint{file, std::forward<decltype(args)>(args)...});
}

} // namespace damn::hdf5

#endif // DAMN_HDF5_CHECKPOINT_HPP
