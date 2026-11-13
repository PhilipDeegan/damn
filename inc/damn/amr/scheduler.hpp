#ifndef DAMN_AMR_SCHEDULER_HPP
#define DAMN_AMR_SCHEDULER_HPP

// Scheduler: named schedules, as in PHAREHUB/PHARE#1079
//
//   add(scheduler, key, kind, names)  resources registered under the same key and kind are
//                                     exchanged together, by one schedule
//   fill<kind>(scheduler, key, ...)   runs it on a level; the schedule is (re)built when the
//                                     level is new or its patches changed
//
// the call site says what moves (key) and how (kind); there is no per quantity function

#include "damn/core/def.hpp"
#include "damn/amr/schedule.hpp"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include <cstddef>
#include <stdexcept>

namespace damn::amr
{
template<std::size_t dim>
struct Scheduler
{
    struct Entry
    {
        std::vector<std::string> names{};
        std::map<int, Schedule<dim>> per_level{}; // level number -> schedule
    };

    std::map<std::pair<std::string, Transfer>, Entry> entries{};
};


// registers resources `names` under key for kind; adding to a key extends it, and drops
// its schedules
template<std::size_t dim>
void add(Scheduler<dim>& scheduler, std::string const& key, Transfer const kind,
         std::vector<std::string> const& names)
{
    auto& entry = scheduler.entries[{key, kind}];
    for (auto const& name : names)
        if (std::find(entry.names.begin(), entry.names.end(), name) == entry.names.end())
            entry.names.push_back(name);
    entry.per_level.clear();
}

template<std::size_t dim>
NO_DISCARD bool has(Scheduler<dim> const& scheduler, std::string const& key, Transfer const kind)
{
    return scheduler.entries.count({key, kind}) > 0;
}

template<std::size_t dim>
NO_DISCARD std::vector<std::string> const& names(Scheduler<dim> const& scheduler,
                                                 std::string const& key, Transfer const kind)
{
    if (!has(scheduler, key, kind))
        throw std::runtime_error("damn::amr::Scheduler: nothing registered for " + key);
    return scheduler.entries.at({key, kind}).names;
}

// the schedule for key on level ln, built if missing or stale
template<Transfer kind, std::size_t dim, typename PatchData>
Schedule<dim> const& schedule_for(Scheduler<dim>& scheduler, std::string const& key,
                                  Hierarchy<dim> const& h, LevelData<dim, PatchData>& ld,
                                  int const rank)
{
    auto const it = scheduler.entries.find({key, kind});
    if (it == scheduler.entries.end())
        throw std::runtime_error("damn::amr::Scheduler: nothing registered for " + key);

    auto& entry   = it->second;
    auto const ln = ld.level;
    auto found    = entry.per_level.find(ln);
    if (found == entry.per_level.end() or found->second.rank != rank
        or !is_current(found->second, h.levels.at(ln)))
        found = entry.per_level
                    .insert_or_assign(ln, make_schedule(kind, h, ln, ld, rank, entry.names))
                    .first;
    return found->second;
}

// exchanges key's resources on ld's level
template<Transfer kind, std::size_t dim, typename PatchData>
void fill(Scheduler<dim>& scheduler, std::string const& key, Hierarchy<dim> const& h,
          LevelData<dim, PatchData>& ld, int const rank)
{
    run(schedule_for<kind>(scheduler, key, h, ld, rank), ld);
}

// forget every schedule built on level ln and finer, e.g. after a regrid
template<std::size_t dim>
void drop_schedules(Scheduler<dim>& scheduler, int const ln)
{
    for (auto& [_, entry] : scheduler.entries)
        entry.per_level.erase(entry.per_level.lower_bound(ln), entry.per_level.end());
}

} // namespace damn::amr

#endif // DAMN_AMR_SCHEDULER_HPP
