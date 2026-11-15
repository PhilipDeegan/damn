// damn::hdf5 level tables: a level's patches' rows (members of an array of structs, through
// strided spans, and a contiguous column) written by their owners straight from memory, the
// level (boxes, ids, owners, offsets) with them, then read back by any rank, its own patches or
// another's, into memory in place; run with any number of ranks, e.g. mpirun -n 3

#include "damn/hdf5/level_table.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/memory/strided_span.hpp"
#include "damn/hdf5/h5_file.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include "gtest/gtest.h"
#include "mpi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace damn::hdf5
{
namespace
{
    struct Item // as a particle: mixed members, padding
    {
        double w;
        int c;
        std::array<double, 2> d;
    };

    std::size_t rows_of(std::uint32_t const id)
    {
        return 3 * id + 1;
    }

    Item item(std::uint32_t const id, std::size_t const i)
    {
        return {id + 0.5 * i, int(100 * id + i), {-1. * i, 1000. * id + i}};
    }

    std::vector<Item> items_of(std::uint32_t const id)
    {
        std::vector<Item> out(rows_of(id));
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = item(id, i);
        return out;
    }

    template<typename T>
    auto member(std::vector<Item>& v, T Item::* const m)
    {
        return core::strided_span(&(v[0].*m), v.size(), sizeof(Item));
    }
} // namespace

TEST(LevelTable, ownersWriteFromMemoryAnyRankReadsIntoMemory)
{
    int const rank = mpi::rank(), size = mpi::size();
    std::string const name = "test_damn_level_table.h5";

    // 7 patches round robin, patch k's id k, its rows 3k + 1
    std::vector<core::Box<int, 1>> boxes;
    std::vector<int> owners;
    for (int k = 0; k < 7; ++k)
    {
        boxes.push_back({std::array{10 * k}, std::array{10 * k + 9}});
        owners.push_back(k % size);
    }
    auto const level = amr::make_level(2, boxes, owners);

    std::vector<std::vector<Item>> local; // this rank's, in level order
    std::vector<std::uint64_t> local_rows;
    for (auto const& p : level.patches)
        if (p.rank == rank)
        {
            local.push_back(items_of(p.id));
            local_rows.push_back(rows_of(p.id));
        }

    {
        auto const rows   = level_rows(level, rank, local_rows);
        auto const& table = rows.table;
        ASSERT_EQ(table.patches(), level.patches.size());
        for (std::size_t k = 0; k < level.patches.size(); ++k)
            EXPECT_EQ(table.rows(k), rows_of(level.patches[k].id));

        h5::HighFiveFile file{name, h5::HiFile::Create | h5::HiFile::Truncate};
        write_level(file, "/level2", rows);

        std::vector<core::StridedSpan<double const>> w, dy;
        std::vector<core::StridedSpan<int const>> c;
        for (auto& v : local)
        {
            w.push_back(member(v, &Item::w));
            c.push_back(member(v, &Item::c));
            dy.push_back(core::strided_span<double const>(&v[0].d[1], v.size(), sizeof(Item)));
        }
        write_rows(file, "/level2/w", rows, w);
        write_rows(file, "/level2/c", rows, c);
        write_rows(file, "/level2/d_y", rows, dy);

        std::vector<core::StridedSpan<double const>> columns; // contiguous
        std::vector<std::vector<double>> per_patch;
        for (auto const n : local_rows)
            per_patch.emplace_back(n, 10. * rank);
        for (auto const& v : per_patch)
            columns.push_back(core::strided_span(v.data(), v.size(), sizeof(double)));
        write_rows(file, "/level2/column", rows, columns);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    {
        h5::HighFiveFile file{name, h5::HiFile::ReadOnly};
        auto const stored = read_level<1>(file, "/level2", 2);
        EXPECT_EQ(stored.level.number, 2);
        ASSERT_EQ(stored.level.patches, level.patches);

        for (std::size_t k = 0; k < stored.level.patches.size(); ++k) // every patch, any owner
        {
            auto const id = stored.level.patches[k].id;
            std::vector<Item> back(stored.table.rows(k), Item{-9, -9, {-9, -9}});
            read_rows(file, "/level2/w", stored.table.offset(k), member(back, &Item::w));
            read_rows(file, "/level2/c", stored.table.offset(k), member(back, &Item::c));
            read_rows(file, "/level2/d_y", stored.table.offset(k),
                      core::strided_span(&back[0].d[1], back.size(), sizeof(Item)));
            for (std::size_t i = 0; i < back.size(); ++i)
            {
                auto const expect = item(id, i);
                EXPECT_EQ(back[i].w, expect.w);
                EXPECT_EQ(back[i].c, expect.c);
                EXPECT_EQ(back[i].d[1], expect.d[1]);
                EXPECT_EQ(back[i].d[0], -9.); // not read: untouched
            }

            std::vector<double> column(stored.table.rows(k));
            read_rows(file, "/level2/column", stored.table.offset(k),
                      core::strided_span(column.data(), column.size(), sizeof(double)));
            for (auto const v : column)
                EXPECT_EQ(v, 10. * stored.level.patches[k].rank);
        }

        std::vector<double> past(1);
        EXPECT_THROW(read_rows(file, "/level2/w", stored.table.total(),
                               core::strided_span(past.data(), 1, sizeof(double))),
                     std::runtime_error);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::filesystem::remove(name);
}

TEST(LevelTable, rowsMustMatchThePatches)
{
    int const rank = mpi::rank(), size = mpi::size();
    std::vector<core::Box<int, 1>> boxes{{std::array{0}, std::array{9}}};
    auto const level = amr::make_level(0, boxes, std::vector<int>{0});
    std::vector<std::uint64_t> const none, two{1, 2};
    if (size == 1)
    {
        EXPECT_THROW(std::ignore = level_rows(level, rank, none), std::runtime_error);
        EXPECT_THROW(std::ignore = level_rows(level, rank, two), std::runtime_error);
    }
    auto const rows = level_rows(level, rank, rank == 0 ? std::vector<std::uint64_t>{4} : none);
    EXPECT_EQ(rows.table.total(), 4u);
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
