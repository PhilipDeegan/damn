#ifndef DAMN_TST_AMR_TOY_REGRID_HPP
#define DAMN_TST_AMR_TOY_REGRID_HPP

// regrid tests' helpers, serial and MPI: toy level data filled with values known everywhere
// (fields) and one particle per cell, and checks of a regridded level against them

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "tst/damn/amr/toy.hpp"

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace damn::amr
{
namespace
{
    // field name's value at p: every field differs
    double g_of(Hierarchy<2> const& h, std::string_view const name, Point2 const& p)
    {
        return f_of(h, p) + 1e4 * static_cast<double>(name[0]) + 1e6 * name.size();
    }

    template<typename Fn>
    void for_each_field(toy::PatchData& data, Fn&& fn)
    {
        for_each_resource(data, [&](std::string_view const name, auto& r) {
            if constexpr (std::is_same_v<std::decay_t<decltype(r)>, toy::Field>)
                fn(name, r);
        });
    }

    // domain values from g_of (ghosts stay NaN), a domain particle per cell, one patch ghost
    // particle per patch
    void fill(Hierarchy<2> const& h, LevelData_t& ld)
    {
        for_each_patch(ld, [&](Patch<2> const& patch, toy::PatchData& data) {
            for_each_field(data, [&](std::string_view const name, toy::Field& f) {
                core::for_each_point(domain_box(f, patch.box), [&](auto const& p) {
                    value(f, patch.box, p) = g_of(h, name, p);
                });
            });
            core::for_each_point(patch.box, [&](auto const& p) {
                data.domain.items.push_back({{p[0], p[1]}, {p[0], p[1]}, patch.id});
            });
            auto const ghost = patch.box.lower - Point2{std::array{1, 1}};
            data.patch_ghost.items.push_back(
                {{ghost[0], ghost[1]}, {ghost[0], ghost[1]}, patch.id});
        });
    }

    // mismatches of ld's domains with fill's values: fields, and one domain particle per cell
    // (where it started)
    std::size_t check(Hierarchy<2> const& h, LevelData_t& ld)
    {
        std::size_t bad = 0;
        for_each_patch(ld, [&](Patch<2> const& patch, toy::PatchData& data) {
            for_each_field(data, [&](std::string_view const name, toy::Field& f) {
                core::for_each_point(domain_box(f, patch.box), [&](auto const& p) {
                    bad += value(f, patch.box, p) != g_of(h, name, p);
                });
            });
            bad += data.domain.items.size() != static_cast<std::size_t>(patch.box.size());
            for (auto const& item : data.domain.items)
                bad += !core::isIn(Point2{item.cell}, patch.box) or item.cell != item.origin;
        });
        return bad;
    }

    std::size_t domain_items(LevelData_t const& ld)
    {
        std::size_t n = 0;
        for (auto const& data : ld.data)
            n += data.domain.items.size();
        return n;
    }
} // namespace
} // namespace damn::amr

#endif // DAMN_TST_AMR_TOY_REGRID_HPP
