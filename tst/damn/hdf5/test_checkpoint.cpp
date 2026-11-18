// damn::hdf5 checkpoints: a level's resources by name, written then read back into fresh patch
// data on the same patches: a C ordered field through its box_span (in place) with its ghosts
// (its checkpoint_box), an x fastest one through value() (staged) on its domain only, items
// through their columns (resized, then in place); ranks without patches take part. Any number
// of ranks

#include "damn/hdf5/checkpoint.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/memory/box_span.hpp"
#include "mkn/kul/strided_span.hpp"
#include "damn/hdf5/h5_file.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include "tst/damn/amr/toy.hpp"

#include "gtest/gtest.h"
#include "mpi.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace toy
{
// a field in C order (the last direction fastest), as damn's NdArray: in place through box_span
struct CField
{
    int gw = 2;
    std::vector<double> v{};
};
inline Box2 domain_box(CField const&, Box2 const& cells)
{
    return cells;
}
inline Box2 ghost_box(CField const& f, Box2 const& cells)
{
    return damn::core::grow(cells, f.gw);
}
inline Box2 checkpoint_box(CField const& f, Box2 const& cells) // its ghosts too
{
    return ghost_box(f, cells);
}
inline damn::core::BoxSpan<double, 2> box_span(CField& f, Box2 const& cells)
{
    return {f.v.data(), ghost_box(f, cells)};
}
inline double& value(CField& f, Box2 const& cells, Point2 const& p)
{
    return box_span(f, cells)(p);
}
inline double const& value(CField const& f, Box2 const& cells, Point2 const& p)
{
    return damn::core::BoxSpan<double const, 2>{f.v.data(), ghost_box(f, cells)}(p);
}

// toy particles as columns: an Item's members, one row per item
template<typename P>
void columns_of(P& p, auto&& fn)
{
    auto const n   = p.items.size();
    auto* const it = p.items.data();
    auto const of  = [&](auto* const first) {
        return mkn::kul::StridedSpan<std::remove_reference_t<decltype(*first)>>{n ? first : nullptr,
                                                                                n, sizeof(Item)};
    };
    fn("cell_x", of(n ? &it->cell[0] : nullptr));
    fn("cell_y", of(n ? &it->cell[1] : nullptr));
    fn("origin_x", of(n ? &it->origin[0] : nullptr));
    fn("from_patch", of(n ? &it->from_patch : nullptr));
}
inline void for_each_column(Particles& p, auto&& fn)
{
    columns_of(p, fn);
}
inline void for_each_column(Particles const& p, auto&& fn)
{
    columns_of(p, fn);
}
inline void resize_columns(Particles& p, std::size_t const n)
{
    p.items.resize(n);
}

struct CheckpointData
{
    CField E;
    Field B{{true, false}}; // x fastest: staged
    Particles parts;
};
inline void for_each_resource(CheckpointData& d, auto&& fn)
{
    fn("E", d.E);
    fn("B", d.B);
    fn("parts", d.parts);
}
} // namespace toy

namespace damn::hdf5
{
namespace
{
    using Box2         = core::Box<int, 2>;
    auto constexpr nan = std::numeric_limits<double>::quiet_NaN();

    toy::CheckpointData allocate_one(amr::Patch<2> const& p)
    {
        toy::CheckpointData d;
        d.E.v.assign(ghost_box(d.E, p.box).size(), nan);
        d.B.v.assign(ghost_box(d.B, p.box).size(), nan);
        return d;
    }

    double e_of(toy::Point2 const& p)
    {
        return 1000. * p[0] + p[1];
    }
    double b_of(toy::Point2 const& p)
    {
        return -1000. * p[1] + p[0];
    }

    void fill(toy::CheckpointData& d, amr::Patch<2> const& p)
    {
        core::for_each_point(ghost_box(d.E, p.box), // ghosts too: E's checkpoint box
                             [&](auto const& pt) { value(d.E, p.box, pt) = e_of(pt); });
        core::for_each_point(domain_box(d.B, p.box),
                             [&](auto const& pt) { value(d.B, p.box, pt) = b_of(pt); });
        for (std::uint32_t i = 0; i < 2 * p.id + 1; ++i)
            d.parts.items.push_back(
                {{p.box.lower[0] + int(i), p.box.lower[1]}, {int(i), -int(i)}, p.id});
    }

    void round_trip(int const n_patches)
    {
        int const rank = mpi::rank(), size = mpi::size();
        std::string const name = "test_damn_checkpoint.h5";

        std::vector<Box2> boxes;
        std::vector<int> owners;
        for (int k = 0; k < n_patches; ++k)
        {
            boxes.push_back({std::array{10 * k, 0}, std::array{10 * k + 9, 5 + k}});
            owners.push_back((n_patches - 1 - k) % size);
        }
        auto const level = amr::make_level(1, boxes, owners);
        std::vector<std::string> const names{"E", "B", "parts"};

        {
            auto ld = amr::allocate(level, rank, allocate_one);
            for (std::size_t i = 0; i < ld.data.size(); ++i)
                fill(ld.data[i], ld.patches[i]);
            h5::HighFiveFile file{name, h5::HiFile::Create | h5::HiFile::Truncate};
            write_checkpoint(file, "/t", level, ld, names);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto ld = amr::allocate(level, rank, allocate_one); // fresh: NaN, no items
        {
            h5::HighFiveFile file{name, h5::HiFile::ReadOnly};
            read_checkpoint(LevelCheckpoint{file, "/t", level, ld, names});
        }
        for (std::size_t i = 0; i < ld.data.size(); ++i)
        {
            auto const& p = ld.patches[i];
            auto& d       = ld.data[i];
            core::for_each_point(ghost_box(d.E, p.box), [&](auto const& pt) {
                EXPECT_EQ(value(d.E, p.box, pt), e_of(pt)); // its checkpoint box: the ghosts
            });
            core::for_each_point(ghost_box(d.B, p.box), [&](auto const& pt) {
                auto const v = value(d.B, p.box, pt);
                if (core::isIn(pt, domain_box(d.B, p.box)))
                    EXPECT_EQ(v, b_of(pt));
                else
                    EXPECT_TRUE(std::isnan(v)); // domain only: no ghosts
            });
            auto expect = allocate_one(p);
            fill(expect, p);
            ASSERT_EQ(d.parts.items.size(), expect.parts.items.size());
            for (std::size_t j = 0; j < d.parts.items.size(); ++j)
            {
                EXPECT_EQ(d.parts.items[j].cell, expect.parts.items[j].cell);
                EXPECT_EQ(d.parts.items[j].origin[0], expect.parts.items[j].origin[0]);
                EXPECT_EQ(d.parts.items[j].origin[1], 0); // not a column: not read
                EXPECT_EQ(d.parts.items[j].from_patch, p.id);
            }
        }

        // other patches than written: refused
        auto const other = amr::make_level(1, std::vector<Box2>{boxes[0]}, std::vector<int>{0});
        auto other_ld    = amr::allocate(other, rank, allocate_one);
        {
            h5::HighFiveFile file{name, h5::HiFile::ReadOnly};
            EXPECT_THROW(read_checkpoint(file, "/t", other, other_ld, names), std::runtime_error);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            std::filesystem::remove(name);
    }
} // namespace

TEST(Checkpoint, fieldsAndColumnsRoundTrip)
{
    round_trip(7);
}

TEST(Checkpoint, ranksWithoutPatchesTakePart)
{
    round_trip(2);
}

} // namespace damn::hdf5


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    auto const ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}
