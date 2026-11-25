#ifndef DAMN_TST_AMR_TOY_HPP
#define DAMN_TST_AMR_TOY_HPP

// a toy field (primal or dual per direction, gw ghosts), toy particles and patch data for
// damn::amr's copy tests, serial and MPI, and their brute force helpers

#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace toy
{
using Box2   = damn::core::Box<int, 2>;
using Point2 = damn::core::Point<int, 2>;

// values on the nodes of a patch: primal nodes on cell corners (one more than cells),
// dual on cell centres, gw ghost nodes each side
struct Field
{
    std::array<bool, 2> primal{};
    int gw = 2;
    std::vector<double> v{};
};

inline Box2 domain_box(Field const& f, Box2 cells)
{
    for (std::size_t i = 0; i < 2; ++i)
        cells.upper[i] += f.primal[i];
    return cells;
}

inline Box2 ghost_box(Field const& f, Box2 const& cells)
{
    return damn::core::grow(domain_box(f, cells), f.gw);
}

inline std::size_t flat(Field const& f, Box2 const& cells, Point2 const& p)
{
    auto const gb = ghost_box(f, cells);
    auto const nx = gb.upper[0] - gb.lower[0] + 1;
    assert(damn::core::isIn(p, gb));
    return (p[1] - gb.lower[1]) * nx + (p[0] - gb.lower[0]);
}

inline double& value(Field& f, Box2 const& cells, Point2 const& p)
{
    return f.v[flat(f, cells, p)];
}
inline double const& value(Field const& f, Box2 const& cells, Point2 const& p)
{
    return f.v[flat(f, cells, p)];
}

struct Population
{
    std::string name;
    Field density;
    std::vector<int> particles; // neither BoxData nor CellItemData
};

// a particle: its cell and where it started, for checks
struct Item
{
    std::array<int, 2> cell;
    std::array<int, 2> origin;
    std::uint32_t from_patch;
};

struct Particles
{
    using value_type = Item;
    int gw           = 1; // ghost cells a patch may hold particles in
    std::vector<Item> items{};
};

inline Box2 domain_box(Particles const&, Box2 const& cells)
{
    return cells;
}
inline Box2 ghost_box(Particles const& p, Box2 const& cells)
{
    return damn::core::grow(cells, p.gw);
}
inline void for_each_item(Particles const& p, auto&& fn)
{
    for (auto const& item : p.items)
        fn(item);
}
inline Point2 cell(Item const& item)
{
    return Point2{item.cell};
}
inline void append(Particles& p, Item item, Point2 const& shift)
{
    for (std::size_t i = 0; i < 2; ++i)
        item.cell[i] += shift[i];
    p.items.push_back(item);
}

// counts the PatchData alive (not moved from), and the most at once: regrid's memory tests
struct Tracker
{
    static inline long live = 0, peak = 0;
    static void reset_peak() { peak = live; }

    Tracker() { up(); }
    Tracker(Tracker const& that)
        : alive{that.alive}
    {
        if (alive)
            up();
    }
    Tracker(Tracker&& that) noexcept
        : alive{std::exchange(that.alive, false)}
    {
    }
    Tracker& operator=(Tracker const& that)
    {
        if (alive != that.alive)
            that.alive ? up() : down();
        alive = that.alive;
        return *this;
    }
    Tracker& operator=(Tracker&& that) noexcept
    {
        if (alive)
            down();
        alive = std::exchange(that.alive, false);
        return *this;
    }
    ~Tracker()
    {
        if (alive)
            down();
    }

private:
    void up()
    {
        if (++live > peak)
            peak = live;
    }
    void down() { --live; }
    bool alive = true;
};

struct PatchData
{
    Tracker tracker;
    Field Bx{{true, false}}, rho{{true, true}}, Ez{{false, false}};
    std::vector<Population> populations;
    Particles domain, patch_ghost;
};

inline void for_each_resource(PatchData& data, auto&& fn)
{
    fn("Bx", data.Bx);
    fn("rho", data.rho);
    fn("Ez", data.Ez);
    for (auto& pop : data.populations)
    {
        fn(pop.name + "_density", pop.density);
        fn(pop.name + "_particles", pop.particles);
    }
    fn("domain", data.domain);
    fn("patch_ghost", data.patch_ghost);
}

// LevelData holding patch data by unique_ptr (move only), as plamn's
inline void for_each_resource(std::unique_ptr<PatchData>& data, auto&& fn)
{
    for_each_resource(*data, fn);
}
} // namespace toy


namespace damn::amr
{
static_assert(BoxData<toy::Field, 2>);
static_assert(not BoxData<std::vector<int>, 2>);
static_assert(CellItemData<toy::Particles, 2>);
static_assert(not CellItemData<std::vector<int>, 2>);
static_assert(not BoxData<toy::Particles, 2>);

namespace
{
    using toy::Box2;
    using toy::Point2;
    using LevelData_t = LevelData<2, toy::PatchData>;

    double const nan = std::numeric_limits<double>::quiet_NaN();

    // 8 x 6 cells, patches of at most 3 x 4: 3 x 2 of them, uneven
    [[maybe_unused]] Hierarchy<2> make_hierarchy(std::array<bool, 2> const periodic,
                                                 int const n_ranks = 1)
    {
        Hierarchy<2> h;
        h.domain   = Box2{std::array{0, 0}, std::array{7, 5}};
        h.dx       = {.1, .1};
        h.periodic = periodic;
        set_level(h, uniform_level_0(h, {3, 4}, n_ranks));
        return h;
    }

    int constexpr reach = 3; // toy fields: 2 ghosts, plus one for primal nodes

    [[maybe_unused]] Neighbours<2> neighbours_of(Hierarchy<2> const& h, int const rank = 0,
                                                 Topology const& t = single_node())
    {
        return make_neighbours(h, 0, rank, t, reach);
    }

    [[maybe_unused]] toy::PatchData make_patch_data(Patch<2> const& patch)
    {
        toy::PatchData data;
        data.populations = {{"protons", {{true, true}}, {}}, {"alpha", {{true, true}}, {}}};
        auto const init  = [&](toy::Field& f) { f.v.assign(ghost_box(f, patch.box).size(), nan); };
        init(data.Bx);
        init(data.rho);
        init(data.Ez);
        for (auto& pop : data.populations)
            init(pop.density);
        return data;
    }

    [[maybe_unused]] toy::Field& field(toy::PatchData& d, std::string const& name)
    {
        toy::Field* f = nullptr;
        for_each_resource(d, [&](std::string_view n, auto& r) {
            if constexpr (std::is_same_v<std::decay_t<decltype(r)>, toy::Field>)
                if (n == name)
                    f = &r;
        });
        assert(f);
        return *f;
    }

    int mod(int a, int n)
    {
        return ((a % n) + n) % n;
    }

    // a function of the physical point: periodic where the hierarchy is
    [[maybe_unused]] double f_of(Hierarchy<2> const& h, Point2 p)
    {
        auto const shape = level_domain(h, 0).shape();
        for (std::size_t i = 0; i < 2; ++i)
            if (h.periodic[i])
                p[i] = mod(p[i], shape[i]);
        return 100. * p[0] + p[1];
    }

    // every (patch, shift) image of a field's box, for brute force
    template<typename BoxFn>
    std::vector<std::pair<Patch<2>, Box2>> images(Hierarchy<2> const& h, BoxFn&& box_of)
    {
        std::vector<Point2> shifts{Point2{}};
        for (auto const& s : periodic_shifts(h, 0))
            shifts.push_back(s);
        std::vector<std::pair<Patch<2>, Box2>> out;
        for (auto const& p : h.levels[0].patches)
            for (auto const& s : shifts)
                out.emplace_back(p, core::shift(box_of(p), s));
        return out;
    }
} // namespace
} // namespace damn::amr

#endif // DAMN_TST_AMR_TOY_HPP
