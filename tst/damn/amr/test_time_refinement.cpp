// damn::amr::advance: the order and times of the hooks with time refinement

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/time_refinement.hpp"
#include "damn/core/geometry/box.hpp"

#include "gtest/gtest.h"

#include <array>
#include <sstream>
#include <string>
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

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
