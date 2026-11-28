#ifndef DAMN_CORE_DATA_NDARRAY_NDARRAY_VIEW_HPP
#define DAMN_CORE_DATA_NDARRAY_NDARRAY_VIEW_HPP


#include "damn/core/data/ndarray/ndarray_base.hpp"
#include "damn/core/data/ndarray/ndarray_mask.hpp"
#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include "mkn/kul/float.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/math.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::core
{


template<std::size_t dim, typename DataType = double, bool c_ordering = true>
class NdArrayView
{
    using viewer = NdArrayViewer<dim, c_ordering>;

public:
    std::size_t static const dimension = dim;
    static constexpr bool is_c_ordered = c_ordering;
    using type                         = DataType;
    using value_type                   = DataType;
    using pointer_type                 = DataType*;


    NdArrayView() DAMN_ALL = default;

    NdArrayView(pointer_type ptr, std::array<std::uint32_t, dim> const nCells) DAMN_ALL
        : ptr_{ptr},
          size_{mkn::kul::math::product(nCells)},
          nCells_{nCells}
    {
    }


    NdArrayView(NdArrayView const&)            = default;
    NdArrayView& operator=(NdArrayView const&) = default;


    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto& operator[](std::array<Index, dim> const& indexes) DAMN_ALL
    {
        return viewer::at(ptr_, nCells_, indexes);
    }
    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto& operator[](std::array<Index, dim> const& indexes) const DAMN_ALL
    {
        return viewer::at(ptr_, nCells_, indexes);
    }

    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto const&
    operator()(std::array<Index, dim> const& indexes) const DAMN_ALL
    {
        return viewer::at(ptr_, nCells_, indexes);
    }

    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto& operator()(std::array<Index, dim> const& indexes) DAMN_ALL
    {
        return const_cast<DataType&>(static_cast<NdArrayView const&>(*this)(indexes));
    }


    MKN_KUL_FN_SIG_RET inline auto const& operator()(auto const... indexes) const DAMN_ALL
    {
        return viewer::at(ptr_, nCells_, indexes...);
    }

    inline auto& operator()(auto const... indexes) DAMN_ALL
    {
        return viewer::at(ptr_, nCells_, indexes...);
    }


    MKN_KUL_FN_SIG_RET auto& data() const DAMN_ALL { return ptr_; }
    MKN_KUL_FN_SIG_RET auto& data() DAMN_ALL { return ptr_; }

    MKN_KUL_FN_SIG_RET auto& size() const DAMN_ALL { return size_; }
    MKN_KUL_FN_SIG_RET auto& shape() const DAMN_ALL { return nCells_; }

    MKN_KUL_FN_SIG_RET auto begin() const DAMN_ALL { return ptr_; }
    MKN_KUL_FN_SIG_RET auto begin() DAMN_ALL { return ptr_; }

    MKN_KUL_FN_SIG_RET auto end() const DAMN_ALL { return ptr_ + size_; }
    MKN_KUL_FN_SIG_RET auto end() DAMN_ALL { return ptr_ + size_; }


    void zero() DAMN_ALL { fill(0); }
    auto zeros() const
    {
        return mkn::kul::math::sum_from(*this, [](auto const e) { return e == 0 ? 1 : 0; });
    }
    // void check() const
    // {
    //     DAMN_DEBUG_DO({
    //         for (auto const& e : *this)
    //         {
    //             if (std::isnan(e))
    //                 throw std::runtime_error("NAN");
    //             if (std::isinf(e))
    //                 throw std::runtime_error("INF");
    //         }
    //     })
    // }

    // void notZero() const
    // {
    //     for (std::size_t i = 0; i < size(); ++i)
    //         if (std::abs(data()[i]) < 1e-15)
    //             throw std::runtime_error("ZERO");
    // }

    auto& fill(DataType const& v) DAMN_ALL
    {
        std::fill(begin(), end(), v);
        return *this;
    }

    bool isclose(NdArrayView const& that, double diff = 1e-13) const
    {
        if (this->size() != that.size())
            return false;
        for (std::size_t i = 0; i < this->size(); ++i)
            if (!mkn::kul::float_equals(this->data()[i], that.data()[i], diff))
                return false;
        return true;
    }


    bool operator==(NdArrayView const& that) const
    {
        if (this->size() != that.size())
            return false;
        for (std::size_t i = 0; i < this->size(); ++i)
            if (this->data()[i] != that.data()[i])
                return false;
        return true;
    }

    template<typename View>
    void reset(View& view) DAMN_ALL
    {
        this->ptr_    = view.data();
        this->size_   = view.size();
        this->nCells_ = view.nCells_;
    }
    template<typename Vec>
    void reset(Vec& vec, std::array<std::uint32_t, dim> const& nCells) DAMN_ALL
    {
        this->ptr_    = vec.data();
        this->size_   = vec.size();
        this->nCells_ = nCells;
    }

    void fill_from(NdArrayView const& that)
    {
        if (mkn::kul::for_N_any<dim>([&](auto i) { return shape()[i] != that.shape()[i]; }))
            throw std::runtime_error("ArrayView::fill_from: Incompatible input shape");
        std::copy(that.data(), that.data() + size(), data());
    }

    void setBuffer(pointer_type ptr) DAMN_ALL { ptr_ = ptr; }
    void setShape(std::array<std::uint32_t, dim> const nCells) DAMN_ALL
    {
        nCells_ = nCells;
        size_   = mkn::kul::math::product(nCells);
    }

    auto& reshape(auto const& shape)
    {
        setShape(shape);
        return *this;
    }


    auto size_address() DAMN_ALL { return &size_; }


    MKN_KUL_FN_SIG_RET auto operator[](NdArrayMask&& mask)
    {
        return MaskedView{*this, std::forward<NdArrayMask>(mask)};
    }
    MKN_KUL_FN_SIG_RET auto operator[](NdArrayMask const& mask) { return MaskedView{*this, mask}; }

private:
    pointer_type ptr_  = nullptr;
    std::size_t size_  = 0;
    std::array<std::uint32_t, dim> nCells_ = {};
};


// Like NdArrayView but steps by sizeof(T) rather than sizeof(V) when indexing.
// V must be at offset 0 within T (i.e. V is the first base of T).
template<std::size_t dim, typename V, typename T, bool c_ordering = true>
class NdArrayViewSpan
{
    using viewer = NdArrayViewer<dim, c_ordering>;

    template<typename V2, typename T2>
    static V2& hax(V2* data, std::uint32_t const i) DAMN_ALL
    {
        return *reinterpret_cast<V2*>(reinterpret_cast<T2*>(data) + i);
    }

public:
    std::size_t static const dimension = dim;
    using type       = V;
    using value_type = V;

    NdArrayViewSpan() DAMN_ALL = default;

    NdArrayViewSpan(V* ptr, std::array<std::uint32_t, dim> const& nCells) DAMN_ALL
        : ptr_{ptr},
          size_{mkn::kul::math::product(nCells)},
          nCells_{nCells}
    {
    }

    NdArrayViewSpan(NdArrayViewSpan const&) DAMN_ALL            = default;
    NdArrayViewSpan& operator=(NdArrayViewSpan const&) DAMN_ALL = default;

    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto& operator()(std::array<Index, dim> const& indexes) DAMN_ALL
    {
        return hax<V, T>(ptr_, viewer::idx(nCells_, indexes));
    }
    template<typename Index>
    MKN_KUL_FN_SIG_RET inline auto const&
    operator()(std::array<Index, dim> const& indexes) const DAMN_ALL
    {
        return hax<V const, T const>(ptr_, viewer::idx(nCells_, indexes));
    }
    MKN_KUL_FN_SIG_RET inline auto& operator()(auto const... indexes) DAMN_ALL
    {
        return hax<V, T>(ptr_, viewer::idx(nCells_, indexes...));
    }
    MKN_KUL_FN_SIG_RET inline auto const& operator()(auto const... indexes) const DAMN_ALL
    {
        return hax<V const, T const>(ptr_, viewer::idx(nCells_, indexes...));
    }

    MKN_KUL_FN_SIG_RET auto data() DAMN_ALL { return ptr_; }
    MKN_KUL_FN_SIG_RET auto data() const DAMN_ALL { return ptr_; }
    MKN_KUL_FN_SIG_RET auto& size() const DAMN_ALL { return size_; }
    MKN_KUL_FN_SIG_RET auto& shape() const DAMN_ALL { return nCells_; }

    // Iteration yields T& (full derived type) — correct step size, full CPU interface.
    MKN_KUL_FN_SIG_RET auto begin() DAMN_ALL { return reinterpret_cast<T*>(ptr_); }
    MKN_KUL_FN_SIG_RET auto begin() const DAMN_ALL { return reinterpret_cast<T const*>(ptr_); }
    MKN_KUL_FN_SIG_RET auto end() DAMN_ALL { return reinterpret_cast<T*>(ptr_) + size_; }
    MKN_KUL_FN_SIG_RET auto end() const DAMN_ALL
    {
        return reinterpret_cast<T const*>(ptr_) + size_;
    }

private:
    V* ptr_                              = nullptr;
    std::size_t size_                    = 0;
    std::array<std::uint32_t, dim> nCells_ = {};
};


template<bool c_ordering = true, typename DataType, std::size_t dim>
auto make_array_view(DataType* data, std::array<std::uint32_t, dim> const shape) DAMN_ALL
{
    return NdArrayView<dim, DataType, c_ordering>{data, shape};
}

template<bool c_ordering = true, typename DataType, std::size_t dim>
auto make_array_view(DataType const* const data,
                     std::array<std::uint32_t, dim> const shape) DAMN_ALL
{
    return NdArrayView<dim, DataType const, c_ordering>{data, shape};
}

template<typename DataType, std::size_t dim>
auto make_array_view(std::vector<DataType>& vec, std::array<std::uint32_t, dim> const shape)
{
    return NdArrayView<dim, DataType>{vec.data(), shape};
}


template<typename DataType, std::size_t dim>
auto make_array_view(std::vector<DataType> const& vec, std::array<std::uint32_t, dim> const shape)
{
    return NdArrayView<dim, DataType const>{vec.data(), shape};
}


// true for NdArrayView and anything deriving from it (NdArrayVector, Field, Grid, ...)
template<typename T>
concept is_ndarray_c = requires(std::remove_cvref_t<T>* p) {
    []<std::size_t dim, typename D, bool c_ordering>(NdArrayView<dim, D, c_ordering> const*) {}(p);
};

template<typename T>
inline constexpr bool is_ndarray_v = is_ndarray_c<T>;


} // namespace damn::core

#endif // DAMN_CORE_DATA_NDARRAY_NDARRAY_VIEW_HPP
