#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <string_view>
#include <vector>

namespace damn::amr::kelvin
{
namespace
{
    template<std::size_t dim>
    void faces_are_a_kelvin_cells()
    {
        std::size_t diagonal = 0;
        std::set<std::array<int, dim>> seen;
        for (std::size_t k = 0; k < n_faces<dim>; ++k)
        {
            auto const d = face_delta<dim>(k);
            EXPECT_TRUE(seen.insert(d).second) << k;
            EXPECT_EQ(mirror<dim>(mirror<dim>(k)), k);
            auto const m = face_delta<dim>(mirror<dim>(k));
            int nonzero  = 0;
            for (std::size_t i = 0; i < dim; ++i)
            {
                EXPECT_EQ(m[i], -d[i]) << k;
                nonzero += d[i] != 0;
                EXPECT_EQ(std::abs(d[i]), d[i] == 0 ? 0 : is_diagonal<dim>(k) ? 1 : 2) << k;
            }
            EXPECT_EQ(nonzero, is_diagonal<dim>(k) ? static_cast<int>(dim) : 1) << k;
            diagonal += is_diagonal<dim>(k);
        }
        EXPECT_EQ(diagonal, std::size_t{1} << dim);
    }

    template<std::size_t dim>
    void neighbours_are_a_cell_away()
    {
        core::Point<int, dim> const cell{};
        for (auto const site : {Site::corner, Site::body})
            for (std::size_t k = 0; k < n_faces<dim>; ++k)
            {
                auto const [c, s] = neighbour(cell, site, k);
                EXPECT_EQ(s == site, !is_diagonal<dim>(k)) << k;
                for (std::size_t i = 0; i < dim; ++i)
                    EXPECT_LE(std::abs(c[i]), 1) << k;
                auto const [back, back_site] = neighbour(c, s, mirror<dim>(k));
                EXPECT_EQ(back, cell);
                EXPECT_EQ(back_site, site);
            }
    }
} // namespace

TEST(Kelvin, faces)
{
    faces_are_a_kelvin_cells<2>();
    faces_are_a_kelvin_cells<3>();
    static_assert(n_faces<3> == 14 and n_faces<2> == 8);
    // 3D order: hexagons by sign, lexicographic, then squares -x +x -y +y -z +z
    static_assert(face_delta<3>(0) == std::array{-1, -1, -1});
    static_assert(face_delta<3>(1) == std::array{-1, -1, 1});
    static_assert(face_delta<3>(7) == std::array{1, 1, 1});
    static_assert(face_delta<3>(8) == std::array{-2, 0, 0});
    static_assert(face_delta<3>(13) == std::array{0, 0, 2});
    static_assert(mirror<3>(0) == 7 and mirror<3>(8) == 9 and mirror<3>(13) == 12);
}

TEST(Kelvin, neighbours)
{
    neighbours_are_a_cell_away<2>();
    neighbours_are_a_cell_away<3>();
}

TEST(Kelvin, keysCellsSites)
{
    for (int x = -3; x <= 3; ++x)
        for (int y = -3; y <= 3; ++y)
            for (auto const site : {Site::corner, Site::body})
            {
                core::Point<int, 2> const c{std::array{x, y}};
                auto const key = key_of(c, site);
                EXPECT_EQ(cell_of(key), c);
                EXPECT_EQ(site_of(key), site);
                EXPECT_EQ(key[0] & 1, key[1] & 1);
            }
}

TEST(Kelvin, keyBoxes)
{
    // keys -N..N: odd N, the lowest cells hold body sites only; even N, the uppermost corner only
    for (auto const N : {std::array{7, 6, 5}, std::array{4, 4, 4}})
    {
        KeyBox<3> const keys{Key<3>{std::array{-N[0], -N[1], -N[2]}}, Key<3>{N}, {}};
        EXPECT_NO_THROW(keys.check());
        std::size_t sites = 0, expected = 1, evens = 1, odds = 1;
        core::for_each_point(keys.cells(), [&](auto const& c) {
            for (auto const site : {Site::corner, Site::body})
                sites += keys.has(c, site);
        });
        for (auto const n : N)
        {
            evens *= static_cast<std::size_t>(2 * (n / 2) + 1);
            odds *= static_cast<std::size_t>(2 * ((n + 1) / 2));
        }
        expected = evens + odds;
        EXPECT_EQ(sites, expected);
        if (N == std::array{7, 6, 5})
        {
            EXPECT_EQ(sites, 533u); // 7 7 5 corners, 8 6 6 bodies
        }
    }

    KeyBox<3> const periodic{
        Key<3>{std::array{-8, -6, -4}}, Key<3>{std::array{7, 5, 3}}, {true, true, true}};
    EXPECT_NO_THROW(periodic.check());
    EXPECT_EQ(periodic.cells(), (core::Box<int, 3>{std::array{-4, -3, -2}, std::array{3, 2, 1}}));
    EXPECT_EQ(periodic.wrap(Key<3>{std::array{8, -7, 5}}), (Key<3>{std::array{-8, 5, -3}}));
    EXPECT_THROW(
        (KeyBox<3>{Key<3>{std::array{-7, 0, 0}}, Key<3>{std::array{7, 1, 1}}, {true, false, false}}
             .check()),
        std::runtime_error);
}


namespace
{
    using Box3   = core::Box<int, 3>;
    using Point3 = core::Point<int, 3>;
    using Faces  = std::array<double, n_faces<3>>;

    struct PatchData
    {
        SiteField<double, 3> u;
        SiteField<Faces, 3> phi;
    };

    void for_each_resource(PatchData& d, auto&& fn)
    {
        for_each_site_resource("u", d.u, fn);
        for_each_site_resource("phi", d.phi, fn);
    }

    double f(Key<3> const& k)
    {
        return 100. * k[0] + 10. * k[1] + k[2];
    }

    Hierarchy<3, Kelvin> hierarchy(KeyBox<3> const& keys, std::array<int, 3> const& patch_cells)
    {
        Hierarchy<3, Kelvin> h;
        h.domain   = keys.cells();
        h.dx       = {1., 1., 1.};
        h.periodic = keys.periodic;
        h.ratio    = 3;
        set_level(h, uniform_level_0(h, patch_cells, 1));
        return h;
    }

    // domain sites f(key), phi f(key) + k; ghosts NaN; fill_ghosts: every ghost that is a site
    //  holds its owner's values, every other ghost NaN
    void ghosts_filled(KeyBox<3> const& keys, std::array<int, 3> const& patch_cells)
    {
        auto const h   = hierarchy(keys, patch_cells);
        double const n = std::numeric_limits<double>::quiet_NaN();
        auto ld        = allocate(h.levels[0], 0, [&](Patch<3> const& p) {
            PatchData d;
            d.u.allocate(p.box, n);
            Faces none;
            none.fill(n);
            d.phi.allocate(p.box, none);
            core::for_each_point(p.box, [&](auto const& c) {
                for (auto const site : {Site::corner, Site::body})
                {
                    if (!keys.has(c, site))
                        continue;
                    auto const k               = key_of(c, site);
                    value(d.u[site], p.box, c) = f(k);
                    for (std::size_t i = 0; i < n_faces<3>; ++i)
                        value(d.phi[site], p.box, c)[i] = f(k) + static_cast<double>(i);
                }
            });
            return d;
        });
        auto nb        = make_neighbours(h, 0, 0, single_node(), 1);
        copy(fill_ghosts, nb, site_names({"u", "phi"}), ld);

        std::size_t filled = 0;
        for (std::size_t i = 0; i < size(ld); ++i)
        {
            auto const& box = ld.patches[i].box;
            core::for_each_point(core::grow(box, 1), [&](auto const& c) {
                if (core::isIn(c, box))
                    return;
                for (auto const site : {Site::corner, Site::body})
                {
                    auto const k  = key_of(c, site);
                    auto const u  = value(ld.data[i].u[site], box, c);
                    auto const& p = value(ld.data[i].phi[site], box, c);
                    if (keys.has(k))
                    {
                        auto const w = keys.wrap(k);
                        ASSERT_EQ(u, f(w)) << c;
                        ASSERT_EQ(p[13], f(w) + 13.) << c;
                        ++filled;
                    }
                    else
                        ASSERT_TRUE(std::isnan(u) and std::isnan(p[0])) << c;
                }
            });
        }
        // one patch of a bounded domain: every ghost past the domain, no site
        bool const neighbours = size(ld) > 1 or keys.periodic[0];
        EXPECT_EQ(filled > 0, neighbours) << filled;
    }
} // namespace

TEST(Kelvin, siteFieldsGhostsFilledPeriodic)
{
    KeyBox<3> const keys{
        Key<3>{std::array{-8, -6, -4}}, Key<3>{std::array{7, 5, 3}}, {true, true, true}};
    for (auto const patch_cells : {std::array{8, 6, 4}, std::array{3, 2, 4}, std::array{1, 1, 1}})
        ghosts_filled(keys, patch_cells);
}

TEST(Kelvin, siteFieldsGhostsFilledBounded)
{
    for (auto const N : {std::array{7, 6, 5}, std::array{4, 4, 4}})
        for (auto const patch_cells : {std::array{8, 8, 8}, std::array{3, 2, 4}})
            ghosts_filled(KeyBox<3>{Key<3>{std::array{-N[0], -N[1], -N[2]}}, Key<3>{N}, {}},
                          patch_cells);
}

} // namespace damn::amr::kelvin


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
