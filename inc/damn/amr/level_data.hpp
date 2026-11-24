#ifndef DAMN_AMR_LEVEL_DATA_HPP
#define DAMN_AMR_LEVEL_DATA_HPP

// LevelData: one rank's data on one level, one PatchData per local patch
//
// damn does not know what a PatchData holds: the user's struct owns everything a patch
// needs (fields, particles, populations...). Generic code reaches individual resources
// only through the user's
//
//   void for_each_resource(PatchData&, fn)   // fn(std::string_view name, resource&)
//
// which may also walk runtime sized members (e.g. a vector of populations)

#include "damn/core/def.hpp"
#include "damn/amr/hierarchy.hpp"

#include <memory>
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace damn::amr
{
template<std::size_t dim, typename PatchData>
struct LevelData
{
    auto static constexpr dimension = dim;
    using patch_data_type           = PatchData;

    int level = 0;
    std::vector<Patch<dim>> patches{}; // this rank's patches of the level
    std::vector<PatchData> data{};     // data[i] belongs to patches[i]

    // a rank without a patch of the level: a PatchData made on a one cell patch, its resources'
    //  types and geometry (centring, ghosts), e.g. for inter_level's plans between two patch
    //  data types. Null while there are patches (data[0] serves)
    std::shared_ptr<PatchData> prototype{};
};


// ld.prototype made (make(patch)) if ld has no patch, dropped otherwise
template<std::size_t dim, typename PatchData>
void set_prototype(LevelData<dim, PatchData>& ld, auto&& make)
{
    if (!ld.patches.empty())
        ld.prototype.reset();
    else if (!ld.prototype)
        ld.prototype = std::make_shared<PatchData>(make(Patch<dim>{}));
}


// data[0], else the prototype: one PatchData to read resource types and geometry from, null if
//  there is neither
template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET PatchData* any_patch_data(LevelData<dim, PatchData>& ld)
{
    return ld.data.empty() ? ld.prototype.get() : &ld.data[0];
}

template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET PatchData const* any_patch_data(LevelData<dim, PatchData> const& ld)
{
    return ld.data.empty() ? ld.prototype.get() : &ld.data[0];
}


// the data of rank's patches of level, data[i] = make(patches[i])
template<std::size_t dim>
MKN_KUL_FN_SIG_RET auto allocate(Level<dim> const& level, int const rank, auto&& make)
{
    using PatchData = std::decay_t<decltype(make(level.patches[0]))>;

    LevelData<dim, PatchData> ld{level.number, local_patches(level, rank), {}};
    set_prototype(ld, make);
    ld.data.reserve(ld.patches.size());
    for (auto const& patch : ld.patches)
        ld.data.emplace_back(make(patch));
    return ld;
}


template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET std::size_t size(LevelData<dim, PatchData> const& ld)
{
    return ld.patches.size();
}

template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET std::optional<std::size_t> index_of(LevelData<dim, PatchData> const& ld,
                                                       std::uint32_t const id)
{
    for (std::size_t i = 0; i < ld.patches.size(); ++i)
        if (ld.patches[i].id == id)
            return i;
    return std::nullopt;
}

// index of the local patch with id, throws if there is none
template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET std::size_t local_index(LevelData<dim, PatchData> const& ld,
                                           std::uint32_t const id)
{
    if (auto const i = index_of(ld, id))
        return *i;
    throw std::runtime_error("damn::amr: no local patch with id " + std::to_string(id)
                             + " on level " + std::to_string(ld.level));
}

// the data of the local patch with id
template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET PatchData& at(LevelData<dim, PatchData>& ld, std::uint32_t const id)
{
    return ld.data[local_index(ld, id)];
}

template<std::size_t dim, typename PatchData>
MKN_KUL_FN_SIG_RET PatchData const& at(LevelData<dim, PatchData> const& ld, std::uint32_t const id)
{
    return ld.data[local_index(ld, id)];
}

// fn(patch, data) for each local patch
template<std::size_t dim, typename PatchData>
void for_each_patch(LevelData<dim, PatchData>& ld, auto&& fn)
{
    for (std::size_t i = 0; i < ld.patches.size(); ++i)
        fn(ld.patches[i], ld.data[i]);
}

template<std::size_t dim, typename PatchData>
void for_each_patch(LevelData<dim, PatchData> const& ld, auto&& fn)
{
    for (std::size_t i = 0; i < ld.patches.size(); ++i)
        fn(ld.patches[i], ld.data[i]);
}


// ---------------------------------------------------------------- resources

namespace detail
{
    struct AnyResource // what for_each_resource is checked with
    {
        void operator()(std::string_view, auto&) const {}
    };
} // namespace detail

template<typename PatchData>
concept HasResources = requires(PatchData& data) {
    for_each_resource(data, detail::AnyResource{});
};

// names of data's resources, in for_each_resource's order
template<HasResources PatchData>
MKN_KUL_FN_SIG_RET std::vector<std::string> resource_names(PatchData& data)
{
    std::vector<std::string> names;
    for_each_resource(data, [&](std::string_view name, auto&) { names.emplace_back(name); });
    return names;
}

// fn(resource) for data's resource called name; false if there is none
template<HasResources PatchData>
bool with_resource(PatchData& data, std::string_view const name, auto&& fn)
{
    bool found = false;
    for_each_resource(data, [&](std::string_view n, auto& resource) {
        if (n == name)
        {
            found = true;
            fn(resource);
        }
    });
    return found;
}

} // namespace damn::amr

#endif // DAMN_AMR_LEVEL_DATA_HPP
