// a toy patch shaped like a hybrid one: fixed fields plus a runtime list of populations

#include "damn/amr/level_data.hpp"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace toy
{
struct Population
{
    std::string name;
    std::vector<double> density;
    std::vector<int> particles;
};

struct PatchData
{
    std::vector<double> Bx, By;
    std::vector<Population> populations;
};

// found by ADL from damn::amr
void for_each_resource(PatchData& data, auto&& fn)
{
    fn("Bx", data.Bx);
    fn("By", data.By);
    for (auto& pop : data.populations)
    {
        fn(pop.name + "_density", pop.density);
        fn(pop.name + "_particles", pop.particles);
    }
}

struct NoResources
{
};
} // namespace toy


namespace damn::amr
{
static_assert(HasResources<toy::PatchData>);
static_assert(not HasResources<toy::NoResources>);

namespace
{
    // 10 x 8 cells, 6 patches of 4 x 4 (at most), ranks 0 0 0 1 1 1 (given, not partitioned)
    auto level_0()
    {
        auto const domain = core::Box<int, 2>{std::array{0, 0}, std::array{9, 7}};
        return make_level(0, chop(domain, std::array{4, 4}), {0, 0, 0, 1, 1, 1});
    }

    // one value per cell, two populations named from the input
    auto make_patch_data(Patch<2> const& patch)
    {
        auto const n = patch.box.size();
        toy::PatchData data{std::vector<double>(n, patch.id), std::vector<double>(n), {}};
        for (std::string const name : {"protons", "alpha"})
            data.populations.push_back({name, std::vector<double>(n), {}});
        return data;
    }
} // namespace


TEST(LevelData, allocatesOnePatchDataPerLocalPatch)
{
    auto const level = level_0();
    auto const ld    = allocate(level, 1, make_patch_data);

    EXPECT_EQ(ld.level, 0);
    ASSERT_EQ(size(ld), 3u);
    ASSERT_EQ(ld.data.size(), 3u);
    for (std::size_t i = 0; i < size(ld); ++i)
    {
        EXPECT_EQ(ld.patches[i].rank, 1);
        EXPECT_EQ(ld.data[i].Bx.size(), ld.patches[i].box.size());
        EXPECT_EQ(ld.data[i].Bx[0], ld.patches[i].id);
    }
}

TEST(LevelData, findsPatchDataById)
{
    auto ld = allocate(level_0(), 1, make_patch_data);

    EXPECT_EQ(at(ld, 4).Bx[0], 4.);
    EXPECT_FALSE(index_of(ld, 0)); // rank 0's
    EXPECT_THROW((void)at(ld, 0), std::runtime_error);

    at(ld, 5).By[0] = 2.;
    EXPECT_EQ(ld.data[*index_of(ld, 5)].By[0], 2.);
}

TEST(LevelData, forEachPatchVisitsLocalPatchesInOrder)
{
    auto ld = allocate(level_0(), 0, make_patch_data);

    std::vector<std::uint32_t> ids;
    for_each_patch(ld, [&](auto const& patch, auto& data) {
        ids.push_back(patch.id);
        data.By.assign(data.By.size(), 1.);
    });
    EXPECT_EQ(ids, (std::vector<std::uint32_t>{0, 1, 2}));

    auto const& cld = ld;
    for_each_patch(cld, [](auto const&, auto const& data) { EXPECT_EQ(data.By[0], 1.); });
}

TEST(LevelData, resourcesIncludeRuntimePopulations)
{
    auto ld = allocate(level_0(), 0, make_patch_data);

    EXPECT_EQ(resource_names(ld.data[0]),
              (std::vector<std::string>{"Bx", "By", "protons_density", "protons_particles",
                                        "alpha_density", "alpha_particles"}));

    bool const found = with_resource(ld.data[0], "alpha_density", [](auto& resource) {
        if constexpr (std::is_same_v<std::decay_t<decltype(resource)>, std::vector<double>>)
            resource[0] = 7.;
    });
    EXPECT_TRUE(found);
    EXPECT_EQ(ld.data[0].populations[1].density[0], 7.);

    EXPECT_FALSE(with_resource(ld.data[0], "electrons_density", [](auto&) {}));
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
