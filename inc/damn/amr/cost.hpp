#ifndef DAMN_AMR_COST_HPP
#define DAMN_AMR_COST_HPP

// what a patch costs to advance, for partitioning: doc/amr/cost.md
// counts are global (summed over the ranks by the caller): every rank gets the same costs

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace damn::amr
{
// per level (its number) and box, a weight: what partitioning balances
template<std::size_t dim>
using Weigh = std::function<std::vector<double>(int, std::vector<core::Box<int, dim>> const&)>;

struct CostModel
{
    double item       = 1.; // per counted item
    double cell       = 0.; // per cell
    double ghost_cell = 0.; // per cell within reach of the box, outside it
    double patch      = 0.; // per patch
    int reach         = 0;  // ghost width, in cells

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET double operator()(core::Box<int, dim> const& box, double const count) const
    {
        auto const cells = static_cast<double>(box.size());
        double ghosts    = 0.;
        if (ghost_cell != 0. and reach > 0)
            ghosts = static_cast<double>(core::grow(box, reach).size()) - cells;
        // fused multiply adds, explicitly: each rounded once (IEEE), so the same doubles whatever
        //  the target (-march) or compiler, the same decomposition everywhere (python's
        //  math.fma too); left to the compiler, a*b + c*d fuses either way round, or not
        return std::fma(item, count, std::fma(cell, cells, std::fma(ghost_cell, ghosts, patch)));
    }
};

// a hybrid step's seconds per term, measured on one rank (3D, interpolation order 1, ghost
// reach 4): fitted to the busy time per patch of a decomposition log
MKN_KUL_FN_SIG_RET inline CostModel measured_hybrid()
{
    return CostModel{1.259e-7, 2.472e-7, 1.703e-7, 2.087e-4, 4};
}

template<std::size_t dim>
struct CostMap
{
    core::Box<int, dim> box{};       // the cells counted
    std::array<int, dim> block{};    // cells per block and direction
    std::vector<double> counts{};    // per block, x fastest

    // blocks per direction
    MKN_KUL_FN_SIG_RET std::array<int, dim> shape() const
    {
        std::array<int, dim> n{};
        auto const cells = box.shape();
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (block[d] < 1)
                throw std::runtime_error("damn::amr::CostMap: block < 1");
            n[d] = (cells[d] + block[d] - 1) / block[d];
        }
        return n;
    }

    // the count in b: each block's, times the share of its cells inside b
    MKN_KUL_FN_SIG_RET double operator()(core::Box<int, dim> const& b) const
    {
        auto const n       = shape();
        std::size_t blocks = 1;
        for (auto const x : n)
            blocks *= static_cast<std::size_t>(x);
        if (counts.size() != blocks)
            throw std::runtime_error("damn::amr::CostMap: one count per block");
        auto const overlap = b * box;
        if (!overlap)
            return 0.;
        std::array<int, dim> first{}, last{};
        for (std::size_t d = 0; d < dim; ++d)
        {
            first[d] = (overlap->lower[d] - box.lower[d]) / block[d];
            last[d]  = (overlap->upper[d] - box.lower[d]) / block[d];
        }
        double sum             = 0.;
        std::array<int, dim> k = first;
        while (true)
        {
            core::Box<int, dim> cell_block{};
            std::size_t flat = 0, stride = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                cell_block.lower[d] = box.lower[d] + k[d] * block[d];
                cell_block.upper[d] = std::min(cell_block.lower[d] + block[d] - 1, box.upper[d]);
                flat += static_cast<std::size_t>(k[d]) * stride;
                stride *= static_cast<std::size_t>(n[d]);
            }
            if (auto const in = *overlap * cell_block)
                sum += counts[flat] * static_cast<double>(in->size())
                       / static_cast<double>(cell_block.size());
            std::size_t d = 0;
            for (; d < dim; ++d)
            {
                if (++k[d] <= last[d])
                    break;
                k[d] = first[d];
            }
            if (d == dim)
                break;
        }
        return sum;
    }
};

// a Weigh: the model's cost of each box from counts(ln, boxes), one count per box
template<std::size_t dim>
MKN_KUL_FN_SIG_RET Weigh<dim> weigh_by(CostModel const model, Weigh<dim> counts)
{
    return [model, counts = std::move(counts)](int const ln,
                                               std::vector<core::Box<int, dim>> const& boxes) {
        auto const n = counts(ln, boxes);
        if (n.size() != boxes.size())
            throw std::runtime_error("damn::amr::weigh_by: one count per box");
        std::vector<double> out(boxes.size());
        for (std::size_t i = 0; i < boxes.size(); ++i)
            out[i] = model(boxes[i], n[i]);
        return out;
    };
}

// per level its CostMap's counts: a Weigh of counts (weigh_by's input)
template<std::size_t dim>
MKN_KUL_FN_SIG_RET Weigh<dim> counts_of(std::vector<CostMap<dim>> maps)
{
    return [maps = std::move(maps)](int const ln, std::vector<core::Box<int, dim>> const& boxes) {
        if (ln < 0 or static_cast<std::size_t>(ln) >= maps.size())
            throw std::runtime_error("damn::amr::counts_of: no CostMap for level "
                                     + std::to_string(ln));
        std::vector<double> out;
        out.reserve(boxes.size());
        for (auto const& b : boxes)
            out.push_back(maps[static_cast<std::size_t>(ln)](b));
        return out;
    };
}

} // namespace damn::amr

#endif // DAMN_AMR_COST_HPP
