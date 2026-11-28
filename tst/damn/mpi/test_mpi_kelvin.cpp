// any number of ranks

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"

#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/topology.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace damn::amr::kelvin
{
namespace
{
    using Faces = std::array<double, n_faces<3>>;

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

    void ghosts_filled(KeyBox<3> const& keys, std::array<int, 3> const& patch_cells)
    {
        int const rank = mpi::rank();
        Hierarchy<3, Kelvin> h;
        h.domain   = keys.cells();
        h.dx       = {1., 1., 1.};
        h.periodic = keys.periodic;
        h.ratio    = 3;
        set_level(h, uniform_level_0(h, patch_cells, mpi::size()));

        double const n = std::numeric_limits<double>::quiet_NaN();
        auto ld        = allocate(h.levels[0], rank, [&](Patch<3> const& p) {
            PatchData d;
            Faces none;
            none.fill(n);
            d.u.allocate(p.box, n);
            d.phi.allocate(p.box, none);
            core::for_each_point(p.box, [&](auto const& c) {
                for (auto const site : {Site::corner, Site::body})
                    if (keys.has(c, site))
                    {
                        auto const k               = key_of(c, site);
                        value(d.u[site], p.box, c) = f(k);
                        for (std::size_t i = 0; i < n_faces<3>; ++i)
                            value(d.phi[site], p.box, c)[i] = f(k) + static_cast<double>(i);
                    }
            });
            return d;
        });
        auto nb        = make_neighbours(h, 0, rank, mpi::topology(), 1);
        mpi::copy(fill_ghosts, nb, site_names({"u", "phi"}), ld);

        for (std::size_t i = 0; i < size(ld); ++i)
        {
            auto const& box = ld.patches[i].box;
            core::for_each_point(core::grow(box, 1), [&](auto const& c) {
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
                    }
                    else
                    {
                        ASSERT_TRUE(std::isnan(u) and std::isnan(p[0])) << c;
                    }
                }
            });
        }
    }
} // namespace

TEST(MPIKelvin, siteFieldsGhostsFilledOverTheRanks)
{
    ghosts_filled(
        KeyBox<3>{Key<3>{std::array{-8, -6, -4}}, Key<3>{std::array{7, 5, 3}}, {true, true, true}},
        {3, 2, 4});
    ghosts_filled(KeyBox<3>{Key<3>{std::array{-7, -6, -5}}, Key<3>{std::array{7, 6, 5}}, {}},
                  {3, 2, 4});
    ghosts_filled(KeyBox<3>{Key<3>{std::array{-4, -4, -4}}, Key<3>{std::array{4, 4, 4}}, {}},
                  {1, 1, 1});
}

} // namespace damn::amr::kelvin


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
