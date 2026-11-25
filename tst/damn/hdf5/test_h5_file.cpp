// damn::hdf5::h5::HighFiveFile: datasets and attributes written then read back, serially and
// across ranks; run with any number of ranks, e.g. mpirun -n 3 bin/test_damn_hdf5/test/...

#include "damn/hdf5/damn_hdf5.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>
#include <numeric>

static_assert(DAMN_HAS_HIGHFIVE);

using namespace damn;
using damn::hdf5::h5::HiFile;
using damn::hdf5::h5::HighFiveFile;

namespace
{
// per number of ranks: runs on 1 and 2 ranks may be concurrent (tst/.phlop.exec.yaml)
std::string file_name()
{
    return "test_damn_h5_file_n" + std::to_string(mpi::size()) + ".h5";
}

std::vector<double> values_for(int const rank) // rank r writes r + 1 values
{
    std::vector<double> v(rank + 1);
    std::iota(v.begin(), v.end(), rank * 100.);
    return v;
}

std::string path_for(int const rank)
{
    return "/ranks/r" + std::to_string(rank);
}
} // namespace


TEST(HighFiveFile, serialRoundTripOfFlatAndNestedDatasets)
{
    auto const name
        = "serial_n" + std::to_string(mpi::size()) + "_r" + std::to_string(mpi::rank()) + ".h5";
    std::vector<double> const flat{1.5, 2.5, 3.5, 4.5};
    std::vector<std::vector<int>> const nested{{1, 2, 3}, {4, 5, 6}};
    {
        HighFiveFile h5{name, HiFile::Truncate, /*para=*/false};
        h5.create_data_set<double>("/a/flat", flat.size());
        h5.write_data_set_flat("/a/flat", flat.data());
        h5.create_data_set<int>("/a/nested", std::vector<std::size_t>{2, 3});
        h5.write_data_set<2>("/a/nested", nested);
        h5.write_attribute("/a", "answer", 42);
        h5.write_attribute("/a", "name", std::string{"damn"});
        h5.write_attribute("/a", "vec", std::vector<double>{1., 2.});
    }
    {
        HighFiveFile h5{name, HiFile::ReadOnly, /*para=*/false};
        EXPECT_TRUE(h5.exist("/a/flat"));
        EXPECT_FALSE(h5.exist("/a/nothing"));
        EXPECT_EQ(h5.read_data_set_flat<double>("/a/flat"), flat);
        EXPECT_EQ((h5.read_data_set<int, 2>("/a/nested")), nested);
        EXPECT_EQ(h5.read_attribute<int>("/a", "answer"), 42);
        EXPECT_EQ(h5.read_attribute("/a", "name"), "damn");
        EXPECT_EQ(h5.read_attribute<std::vector<double>>("/a", "vec"),
                  (std::vector<double>{1., 2.}));
        EXPECT_THROW(h5.getDataSet("/a/nothing"), std::runtime_error);
    }
    std::remove(name.c_str());
}

TEST(HighFiveFile, eachRankWritesItsOwnDatasetAndAttribute)
{
    auto const rank = mpi::rank();
    {
        HighFiveFile h5{file_name(), HiFile::Truncate};
        auto const values = values_for(rank);
        // every rank creates every rank's dataset, then writes its own
        h5.create_data_set_per_mpi<double>(path_for(rank) + "/v", values.size());
        h5.write_data_set_flat(path_for(rank) + "/v", values.data());
        h5.write_attributes_per_mpi(path_for(rank), "rank", rank);
        h5.write_attribute("/", "n_ranks", mpi::size());
    }
    mpi::barrier();
    {
        HighFiveFile h5{file_name(), HiFile::ReadOnly};
        EXPECT_EQ(h5.read_attribute<int>("/", "n_ranks"), mpi::size());
        for (int r = 0; r < mpi::size(); ++r)
        {
            EXPECT_EQ(h5.read_data_set_flat<double>(path_for(r) + "/v"), values_for(r));
            EXPECT_EQ(h5.read_attribute<int>(path_for(r), "rank"), r);
        }
    }
    mpi::barrier();
    if (rank == 0)
        std::remove(file_name().c_str());
}

TEST(HighFiveFile, resizableDatasetsStartEmpty)
{
    auto const name
        = "resizable_n" + std::to_string(mpi::size()) + "_r" + std::to_string(mpi::rank()) + ".h5";
    {
        HighFiveFile h5{name, HiFile::Truncate, /*para=*/false};
        auto one = h5.create_resizable_1d_data_set<double>("/g/one", 8);
        auto two = h5.create_resizable_2d_data_set<double, 3>("/g/two", 8, 3);
        EXPECT_EQ(one.getElementCount(), 0u);
        EXPECT_EQ(two.getDimensions(), (std::vector<std::size_t>{0, 3}));

        one.resize({4});
        one.write(std::vector<double>{1, 2, 3, 4});
    }
    {
        HighFiveFile h5{name, HiFile::ReadOnly, /*para=*/false};
        EXPECT_EQ(h5.read_data_set_flat<double>("/g/one"), (std::vector<double>{1, 2, 3, 4}));
    }
    std::remove(name.c_str());
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}
