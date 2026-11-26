#include "damn/core/bon.hpp"
#include "damn/core/def/bootstrap.hpp"

#include "gtest/gtest.h"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace damn
{
namespace
{
    constexpr std::array<std::pair<std::string_view, std::string_view>, 5> table{{
        {"settings", "res/mkn/mpi"},
        {"mpi/exchange_node/ok", "1"},
        {"mpi/exchange_node/latency_s", "1.5e-07"},
        {"mpi/version/library", "Open MPI v5.0.7, package: Debian"},
        {"hdf5/parallel/ok", "0"},
    }};
} // namespace

TEST(bon, tree_nests_by_path)
{
    auto const t = bon::tree(table);
    ASSERT_TRUE(t.is_map());
    EXPECT_EQ(t.map().size(), 3u); // settings, mpi, hdf5: in input order
    EXPECT_EQ(t.map()[1].key, "mpi");
    EXPECT_EQ(t["mpi"]["exchange_node"].map().size(), 2u);
    EXPECT_EQ(t["settings"].str(), "res/mkn/mpi");
}

TEST(bon, find_by_path)
{
    auto const t = bon::tree(table);
    ASSERT_NE(bon::find(t, "mpi/exchange_node"), nullptr);
    EXPECT_EQ(bon::find(t, "mpi/exchange_node/latency_s")->str(), "1.5e-07");
    EXPECT_EQ(bon::find(t, "mpi/nope"), nullptr);
    EXPECT_EQ(bon::find(t, "settings/deeper"), nullptr); // a leaf has no children
    EXPECT_EQ(bon::find(t, ""), &t);
}

TEST(bon, get_as_T_or_fallback)
{
    auto const t = bon::tree(table);
    EXPECT_DOUBLE_EQ(bon::get(t, "mpi/exchange_node/latency_s", 1.), 1.5e-07);
    EXPECT_EQ(bon::get(t, "mpi/exchange_node/ok", 0), 1);
    EXPECT_DOUBLE_EQ(bon::get(t, "mpi/missing", 2.), 2.);
    EXPECT_DOUBLE_EQ(bon::get(t, "mpi/exchange_node", 3.), 3.);   // a map, not a leaf
    EXPECT_DOUBLE_EQ(bon::get(t, "mpi/version/library", 4.), 4.); // not a number
    EXPECT_EQ(bon::get(t, "mpi/version/library", std::string{}),
              "Open MPI v5.0.7, package: Debian"); // a string whole, spaces included
}

TEST(bon, get_over_the_flat_table)
{
    EXPECT_DOUBLE_EQ(bon::get(table, "mpi/exchange_node/latency_s", 1.), 1.5e-07);
    EXPECT_EQ(bon::get(table, "hdf5/parallel/ok", 1), 0);
    EXPECT_DOUBLE_EQ(bon::get(table, "mpi/missing", 2.), 2.);
}

TEST(bon, bootstrap_facts)
{
    // whatever this build's entries hold, the flat and tree lookups agree, and a missing fact is
    // its fallback
    for (auto const& [path, value] : bootstrap::entries)
    {
        auto const* const node = bon::find(bootstrap::tree(), path);
        if (node and node->is_string())
        {
            EXPECT_EQ(node->str(), std::string{value}) << path;
        }
    }
    EXPECT_DOUBLE_EQ(bootstrap::get("no/such/fact", 7.), 7.);
    EXPECT_EQ(bootstrap::Toolchain::gpu.empty(), false);
    if constexpr (!DAMN_HAVE_BOOTSTRAP)
    {
        EXPECT_TRUE(bootstrap::entries.empty());
    }
}

} // namespace damn

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
