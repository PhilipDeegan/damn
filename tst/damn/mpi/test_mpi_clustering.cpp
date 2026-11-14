// damn::mpi::cluster: each rank's tagged cells, the same boxes on every rank as the serial
// clustering of all of them. Any number of ranks:
//   mpirun -n 3 ./bin/test_damn_mpi/test_mpi_clustering

#include "damn/amr/clustering.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/mpi/clustering.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <vector>

namespace damn::amr
{
namespace
{
    using Box2   = core::Box<int, 2>;
    using Point2 = core::Point<int, 2>;

    // rank r tags cells r, 2 r + 5, ... spread over the domain
    std::vector<Point2> tags_of(int const rank)
    {
        std::vector<Point2> tags;
        for (int k = 0; k < 3; ++k)
            tags.push_back(Point2{std::array{(11 * rank + 7 * k) % 64, (5 * rank + 13 * k) % 64}});
        return tags;
    }
} // namespace

TEST(MpiClustering, everyRankGetsTheClusteringOfEveryRanksTags)
{
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    Hierarchy<2> h;
    h.domain     = Box2{std::array{0, 0}, std::array{63, 63}};
    h.dx         = {.1, .1};
    h.periodic   = {true, false};
    h.max_levels = 2;
    set_level(h, uniform_level_0(h, {16, 16}, ranks));

    ClusteringOptions<2> const options;
    auto const boxes = mpi::cluster(h, 0, tags_of(rank), options);

    std::vector<Point2> all;
    for (int r = 0; r < ranks; ++r)
        for (auto const& t : tags_of(r))
            all.push_back(t);
    EXPECT_EQ(boxes, cluster(h, 0, tagged_tiles(h, 0, all, options), options));
    EXPECT_FALSE(boxes.empty());
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
