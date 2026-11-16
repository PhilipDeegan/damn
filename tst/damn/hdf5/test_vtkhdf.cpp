// damn::hdf5::vtkhdf: an OverlappingAMR time series written by every rank, read back: steps,
// every level's Steps entries (a level that never has patches included), the boxes rank major,
// each rank's rows where the offsets say, a level whose boxes change between steps (a regrid);
// any number of ranks, e.g. mpirun -n 3

#include "damn/hdf5/vtkhdf.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/hdf5/h5_file.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include "gtest/gtest.h"
#include "mpi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace damn::hdf5::vtkhdf
{
namespace
{
    using Box2                       = core::Box<int, 2>;
    std::size_t constexpr components = 3;

    // row r of patch id's block, component c
    float value(std::uint32_t const id, std::size_t const r, std::size_t const c)
    {
        return 1000.f * id + 3.f * static_cast<float>(r) + static_cast<float>(c);
    }

    // this rank's rows, its patches in level order
    std::vector<float> rows_for(amr::Level<2> const& level, int const rank)
    {
        std::vector<float> out;
        for (auto const& p : level.patches)
            if (p.rank == rank)
                for (std::size_t r = 0; r < rows_of(p.box); ++r)
                    for (std::size_t c = 0; c < components; ++c)
                        out.push_back(value(p.id, r, c));
        return out;
    }

    template<typename T>
    std::vector<T> read_1d(h5::HighFiveFile& f, std::string const& path)
    {
        return f.getDataSet(path).read<std::vector<T>>();
    }
} // namespace

TEST(VtkHdf, aTimeSeriesOfLevelsReadBack)
{
    int const rank = mpi::rank(), size = mpi::size();
    std::string const name = "test_damn_vtkhdf.vtkhdf";

    // level 0: 4 patches over the ranks; level 1: on the last rank only, other boxes at step 2;
    //  level 2: never
    auto const level0 = amr::make_level(0,
                                        std::vector<Box2>{{std::array{0, 0}, std::array{7, 3}},
                                                          {std::array{8, 0}, std::array{15, 3}},
                                                          {std::array{0, 4}, std::array{7, 7}},
                                                          {std::array{8, 4}, std::array{15, 7}}},
                                        std::vector<int>{3 % size, 1 % size, 2 % size, 0});
    std::array const level1
        = {amr::make_level(1, std::vector<Box2>{{std::array{4, 4}, std::array{11, 7}}},
                           std::vector<int>{size - 1}),
           amr::make_level(1,
                           std::vector<Box2>{{std::array{0, 0}, std::array{3, 3}},
                                             {std::array{20, 2}, std::array{23, 9}}},
                           std::vector<int>{size - 1, size - 1})};
    amr::Level<2> const none{2, {}};

    {
        h5::HighFiveFile file{name, h5::HiFile::Create | h5::HiFile::Truncate};
        init(file, components, 3, std::array{0.5, 0.25});
        for (std::size_t step = 0; step < 2; ++step)
        {
            append_step(file, 0.1 * static_cast<double>(step));
            for (auto const* level : {&level0, &level1[step], &none})
            {
                auto const rows = rows_for(*level, rank);
                append_level<float>(file, level->number, *level, rank, components,
                                    std::span<float const>{rows});
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    h5::HighFiveFile file{name, h5::HiFile::ReadOnly, false};
    EXPECT_EQ(file.read_attribute<int>(std::string{base} + "/Steps", "NSteps"), 2);
    EXPECT_EQ(read_1d<float>(file, std::string{base} + "/Steps/Values"),
              (std::vector<float>{0.f, 0.1f}));
    EXPECT_EQ(file.read_attribute<std::string>(std::string{base}, "Type"), "OverlappingAMR");

    for (int ln = 0; ln < 3; ++ln)
    {
        auto const steps  = std::string{base} + "/Steps/Level" + std::to_string(ln);
        auto const count  = read_1d<std::int64_t>(file, steps + "/NumberOfAMRBox");
        auto const box_at = read_1d<std::int64_t>(file, steps + "/AMRBoxOffset");
        auto const row_at = read_1d<std::int64_t>(file, steps + "/PointDataOffset/data");
        ASSERT_EQ(count.size(), 2u) << ln; // every level, every step
        ASSERT_EQ(box_at.size(), 2u);
        ASSERT_EQ(row_at.size(), 2u);

        auto const boxes
            = file.getDataSet(std::string{base} + "/Level" + std::to_string(ln) + "/AMRBox")
                  .read<std::vector<std::array<int, 6>>>();
        auto const data
            = file.getDataSet(std::string{base} + "/Level" + std::to_string(ln) + "/PointData/data")
                  .read<std::vector<std::vector<float>>>();
        for (std::size_t step = 0; step < 2; ++step)
        {
            auto const& level = ln == 0 ? level0 : ln == 1 ? level1[step] : none;
            ASSERT_EQ(count[step], static_cast<std::int64_t>(level.patches.size()));

            // rank major: by owner, then level order
            std::vector<amr::Patch<2>> expect;
            for (int r = 0; r < size; ++r)
                for (auto const& p : level.patches)
                    if (p.rank == r)
                        expect.push_back(p);
            auto row = static_cast<std::size_t>(row_at[step]);
            for (std::size_t b = 0; b < expect.size(); ++b)
            {
                auto const& box = boxes[static_cast<std::size_t>(box_at[step]) + b];
                auto const& p   = expect[b];
                EXPECT_EQ(box, (std::array{p.box.lower[0], p.box.upper[0], p.box.lower[1],
                                           p.box.upper[1], 0, 0}));
                for (std::size_t r = 0; r < rows_of(p.box); ++r, ++row)
                    for (std::size_t c = 0; c < components; ++c)
                        ASSERT_EQ(data[row][c], value(p.id, r, c)) << ln << " " << step;
            }
            if (step == 0 and ln < 2)
            {
                EXPECT_EQ(static_cast<std::int64_t>(row), row_at[1]); // the next step's start
            }
        }
    }
    EXPECT_EQ(file.file()
                  .getGroup(std::string{base} + "/Level1")
                  .getAttribute("Spacing")
                  .read<std::vector<float>>(),
              (std::vector<float>{0.25f, 0.125f}));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::filesystem::remove(name);
}

TEST(VtkHdf, rowsAndPoints)
{
    Box2 const cells{std::array{2, 0}, std::array{4, 0}}; // 3 x 1 cells: 4 x 2 nodes
    EXPECT_EQ(points_of(cells), 8u);
    EXPECT_EQ(rows_of(cells), 16u); // 2D: twice, a 3D view
    EXPECT_EQ(rows_of(core::Box<int, 1>{std::array{0}, std::array{9}}), 44u); // 11 nodes x 4
}

} // namespace damn::hdf5::vtkhdf


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    auto const ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}
