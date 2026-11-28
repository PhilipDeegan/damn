#include "damn/core/geometry/box.hpp"

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/time_refinement.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace damn::amr
{
namespace
{
    using Box1 = core::Box<int, 1>;

    Hierarchy<1> levels(int const n)
    {
        Hierarchy<1> h;
        h.domain     = Box1{std::array{0}, std::array{15}};
        h.dx         = {.1};
        h.max_levels = n;
        set_level(h, make_level(0, std::vector{h.domain}));
        if (n > 1)
            set_level(h, make_level(1, std::vector{Box1{std::array{8}, std::array{15}}}));
        if (n > 2)
            set_level(h, make_level(2, std::vector{Box1{std::array{20}, std::array{27}}}));
        return h;
    }

    struct Record
    {
        std::vector<std::string> calls;

        void log(auto const&... args)
        {
            std::stringstream ss;
            ((ss << args << " "), ...);
            auto s = ss.str();
            s.pop_back();
            calls.push_back(s);
        }
        void advance(int ln, double t, double dt) { log("advance", ln, t, dt); }
        void prepare_step(int ln, double t) { log("prepare", ln, t); }
        void first_step(int ln, double t, double ct, double cdt) { log("first", ln, t, ct, cdt); }
        void last_step(int ln) { log("last", ln); }
        void synchronize(int ln, double t) { log("sync", ln, t); }
    };

    // advance and after_step only
    struct AfterSteps
    {
        std::vector<std::string> calls;
        void advance(int ln, double t, double) { calls.push_back("advance " + label(ln, t)); }
        void after_step(int ln, double t, bool last)
        {
            calls.push_back("after " + label(ln, t) + (last ? " last" : ""));
        }
        static std::string label(int ln, double t)
        {
            std::stringstream ss;
            ss << ln << " " << t;
            return ss.str();
        }
    };

    // advance only
    struct Minimal
    {
        int steps = 0;
        void advance(int, double, double) { ++steps; }
    };
} // namespace


TEST(TimeRefinement, oneLevelIsOneStep)
{
    Record r;
    advance(levels(1), r, 1., .5);
    EXPECT_EQ(r.calls, (std::vector<std::string>{"prepare 0 1", "advance 0 1 0.5"}));
}

TEST(TimeRefinement, finerLevelsSubcycleAfterTheCoarserThenSynchronize)
{
    Record r;
    advance(levels(3), r, 0., 1.);
    EXPECT_EQ(r.calls, (std::vector<std::string>{
                           "prepare 0 0",
                           "advance 0 0 1",
                           "first 1 0 0 1",
                           "prepare 1 0",
                           "advance 1 0 0.5", //
                           "first 2 0 0 0.5",
                           "prepare 2 0",
                           "advance 2 0 0.25",
                           "prepare 2 0.25",
                           "advance 2 0.25 0.25",
                           "last 2",
                           "sync 2 0.5", //
                           "prepare 1 0.5",
                           "advance 1 0.5 0.5",
                           "last 1", //
                           "first 2 0.5 0.5 0.5",
                           "prepare 2 0.5",
                           "advance 2 0.5 0.25",
                           "prepare 2 0.75",
                           "advance 2 0.75 0.25",
                           "last 2",
                           "sync 2 1", //
                           "sync 1 1",
                       }));
}

// after a step of a level and of its finer levels with it, the finest level's too: each
//  substep's last flag, level 0's always set
TEST(TimeRefinement, afterStepFollowsTheFinerLevelsCatchingUp)
{
    AfterSteps r;
    advance(levels(3), r, 0., 1.);
    EXPECT_EQ(r.calls, (std::vector<std::string>{
                           "advance 0 0",
                           "advance 1 0",
                           "advance 2 0",
                           "after 2 0.25",
                           "advance 2 0.25",
                           "after 2 0.5 last",
                           "after 1 0.5",
                           "advance 1 0.5",
                           "advance 2 0.5",
                           "after 2 0.75",
                           "advance 2 0.75",
                           "after 2 1 last",
                           "after 1 1 last",
                           "after 0 1 last",
                       }));
}

TEST(TimeRefinement, aTimeRatioOtherThanTheRefinementRatio)
{
    Minimal m;
    advance(levels(3), m, 0., 1., 4); // dt ~ dx^2
    EXPECT_EQ(m.steps, 1 + 4 + 16);

    Record r;
    advance(levels(2), r, 0., 1., 4);
    EXPECT_EQ(r.calls.size(), 2u + 4 * 2 + 2 + 1); // level 0, 4 substeps, first/last, sync
    EXPECT_EQ(r.calls[4], "advance 1 0 0.25");
    EXPECT_EQ(r.calls.back(), "sync 1 1");
}

TEST(TimeRefinement, hooksBeyondAdvanceAreOptional)
{
    Minimal m;
    advance(levels(3), m, 0., 1.);
    EXPECT_EQ(m.steps, 1 + 2 + 4);
}

// ---------------------------------------------------------------- substeps (SAMRAI's split)

namespace
{
    // each level's steps as Steps; next_dt(ln, step): bound(ln, step) (none: the default)
    template<typename Bound = std::nullptr_t>
    struct Steps
    {
        Bound bound{};
        std::vector<std::vector<Step>> steps = std::vector<std::vector<Step>>(3);

        void advance(int ln, Step const& step)
        {
            steps[static_cast<std::size_t>(ln)].push_back(step);
        }
        double next_dt(int ln, Step const& step)
            requires(!std::is_same_v<Bound, std::nullptr_t>)
        {
            return bound(ln, step);
        }
    };
    template<typename Bound>
    Steps(Bound) -> Steps<Bound>;

    // a level's substep start times in one coarser step, and the end of the last
    std::vector<double> times(std::vector<Step> const& steps)
    {
        std::vector<double> t;
        for (auto const& s : steps)
            t.push_back(s.t);
        if (!steps.empty())
            t.push_back(steps.back().t + steps.back().dt);
        return t;
    }
} // namespace

TEST(TimeRefinementSubsteps, theTimeRatioWithoutNextDt)
{
    Steps s;
    TimeRefinement<1> stepping{{.time_ratio = 4}};
    stepping.advance(levels(2), s, 0., 1.);
    ASSERT_EQ(s.steps[1].size(), 4u);
    EXPECT_EQ(times(s.steps[1]), (std::vector{0., .25, .5, .75, 1.}));
    for (std::size_t k = 0; k < 4; ++k)
    {
        auto const& step = s.steps[1][k];
        EXPECT_EQ(step.coarse_t, 0.);
        EXPECT_EQ(step.coarse_dt, 1.);
        EXPECT_EQ(step.substep, static_cast<int>(k));
        EXPECT_EQ(step.first, k == 0);
        EXPECT_EQ(step.last, k == 3);
    }
    EXPECT_TRUE(s.steps[0][0].first and s.steps[0][0].last);
}

TEST(TimeRefinementSubsteps, aBoundSetsTheCount)
{
    // level 1 (not refinable: max_levels 2) asks for a third of its coarser step
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : step.coarse_dt / 3.; }};
    TimeRefinement<1> stepping{{.time_ratio = 4}};
    stepping.advance(levels(2), s, 0., 1.); // new: its first substep bounded by 1 / 4
    stepping.advance(levels(2), s, 1., 1.); // not new: the bound, a third
    ASSERT_EQ(s.steps[1].size(), 4u + 3u);
    std::vector<Step> const second(s.steps[1].begin() + 4, s.steps[1].end());
    auto const t = times(second);
    ASSERT_EQ(t.size(), 4u);
    EXPECT_EQ(t.front(), 1.);
    EXPECT_EQ(t.back(), 2.); // the last substep ends at the coarser level's end exactly
    for (auto const& step : second)
        EXPECT_NEAR(step.dt, 1. / 3., 1e-15);
}

TEST(TimeRefinementSubsteps, aRefinableLevelsCountIsAMultipleOfTheRegridInterval)
{
    // level 1 can be refined (max_levels 3): 3 substeps asked, 4 taken (regrid interval 2)
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : step.coarse_dt / 3.; }};
    auto h = levels(3);
    TimeRefinement<1> stepping{{.time_ratio = 4, .regrid_interval = 2}};
    stepping.advance(h, s, 0., 1.);
    stepping.advance(h, s, 1., 1.);
    EXPECT_EQ(s.steps[1].size(), 4u + 4u);
}

TEST(TimeRefinementSubsteps, aLargeBoundIsOneSubstepOnceTheLevelIsNotNew)
{
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : 10 * step.coarse_dt; }};
    TimeRefinement<1> stepping{{.time_ratio = 4}};
    stepping.advance(levels(2), s, 0., 1.);
    // new: its first substep coarse dt / time_ratio, then its bound: the rest at once
    EXPECT_EQ(times(s.steps[1]), (std::vector{0., .25, 1.}));
    stepping.advance(levels(2), s, 1., 1.);
    EXPECT_EQ(s.steps[1].size(), 2u + 1u);
    EXPECT_EQ(s.steps[1].back().dt, 1.);
}

TEST(TimeRefinementSubsteps, growDtLimitsHowFastTheBoundGrows)
{
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : 10 * step.coarse_dt; }};
    TimeRefinement<1> stepping{{.time_ratio = 4, .grow_dt = 1.}};
    stepping.advance(levels(2), s, 0., 1.);
    stepping.advance(levels(2), s, 1., 1.);
    EXPECT_EQ(s.steps[1].size(), 4u + 4u); // dt_max kept at 1 / 4
}

TEST(TimeRefinementSubsteps, aBoundChangingMidStepSplitsWhatIsLeft)
{
    // after its first substep the level can take the whole rest
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : 1.; }};
    TimeRefinement<1> stepping{{.time_ratio = 4}};
    stepping.advance(levels(2), s, 0., 1.);
    EXPECT_EQ(times(s.steps[1]), (std::vector{0., .25, 1.}));
}

TEST(TimeRefinementSubsteps, aLevelAddedBetweenAdvancesIsNew)
{
    Steps s{[](int ln, Step const& step) { return ln == 0 ? step.dt : 10 * step.coarse_dt; }};
    auto h            = levels(3);
    auto const level2 = h.levels[2];
    h.levels.pop_back(); // two levels, max_levels 3
    TimeRefinement<1> stepping{{.time_ratio = 4}};
    stepping.advance(h, s, 0., 1.);
    h.levels.push_back(level2); // a regrid made level 2
    stepping.advance(h, s, 1., 1.);
    EXPECT_EQ(s.steps[1].size(), 2u + 1u); // level 1: not new, one substep
    // level 2: new, its first substep coarse dt / time_ratio, then the rest
    EXPECT_EQ(times(s.steps[2]), (std::vector{1., 1.25, 2.}));
}

TEST(TimeRefinementSubsteps, aRestoredStateContinuesAsTheOriginal)
{
    auto const bound
        = [](int ln, Step const& step) { return ln == 0 ? step.dt : 10 * step.coarse_dt; };
    TimeRefinement<1> original{{.time_ratio = 4}};
    Steps first{bound};
    original.advance(levels(2), first, 0., 1.); // level 1 new: 2 substeps
    auto const saved = original.state();

    Steps continued{bound}, restored{bound}, fresh{bound};
    original.advance(levels(2), continued, 1., 1.);
    TimeRefinement<1> from_restart{{.time_ratio = 4}};
    from_restart.restore(saved);
    from_restart.advance(levels(2), restored, 1., 1.);
    TimeRefinement<1>{{.time_ratio = 4}}.advance(levels(2), fresh, 1., 1.);

    EXPECT_EQ(times(restored.steps[1]), times(continued.steps[1])); // one substep, not new
    EXPECT_EQ(times(continued.steps[1]), (std::vector{1., 2.}));
    EXPECT_EQ(times(fresh.steps[1]), (std::vector{1., 1.25, 2.})); // a fresh state: new again
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
