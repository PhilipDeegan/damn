#ifndef DAMN_CORE_GEOMETRY_RANGE_HPP
#define DAMN_CORE_GEOMETRY_RANGE_HPP

#include "damn/core/def.hpp"

#include <concepts>
#include <cstddef>
#include <iterator>
#include <utility>

namespace damn
{
namespace core
{
    template<typename Iterator>
    struct Range
    {
        using iterator          = Iterator;
        using iterator_category = Iterator::iterator_category;
        using value_type        = Iterator::value_type;
        using difference_type   = Iterator::difference_type;
        using reference         = Iterator::reference;
        using pointer           = Iterator::pointer;

        Range()             = default;
        Range(Range&&)      = default;
        Range(Range const&) = default;

        template<class Container>
        explicit Range(Container const& c)
            : first_{std::begin(c)}
            , end_{std::end(c)}
        {
        }

        Range(Iterator begin, Iterator end)
            : first_{begin}
            , end_{end}
        {
        }
        MKN_KUL_FN_SIG_RET Iterator begin() const { return first_; }
        MKN_KUL_FN_SIG_RET Iterator end() const { return end_; }
        MKN_KUL_FN_SIG_RET std::size_t size() const { return std::distance(first_, end_); }

    private:
        Iterator first_;
        Iterator end_;
    };

    template<typename Array, std::integral Index = std::size_t>
    class IndexRange
    {
    public:
        using array_t = Array;

        IndexRange(Array& arr, Index first, Index end)
            : first_{first}
            , end_{end}
            , array_{&arr}
        {
        }

        IndexRange(Array& arr)
            : first_{0}
            , end_{arr.size()}
            , array_{&arr}
        {
        }

        IndexRange& operator=(IndexRange const& from) = default;
        IndexRange& operator=(IndexRange&& from)      = default;
        IndexRange(IndexRange&& from)                 = default;
        IndexRange(IndexRange const& from)            = default;

        MKN_KUL_FN_SIG_RET auto size() const { return end_ - first_; }
        MKN_KUL_FN_SIG_RET auto ibegin() const { return first_; }
        MKN_KUL_FN_SIG_RET auto iend() const { return end_; }
        MKN_KUL_FN_SIG_RET auto begin() const { return std::begin(*array_) + first_; }
        MKN_KUL_FN_SIG_RET auto end() const { return std::begin(*array_) + end_; }
        MKN_KUL_FN_SIG_RET auto& array() { return *array_; }
        MKN_KUL_FN_SIG_RET auto const& array() const { return *array_; }

        // these ones are not used for now... they give access to the array element
        // via a range index (i.e. not an array index, but relative to the array index
        // at the start of the range).
        // it is commented out since it may be used now by mistake by a function
        // that'd take an Array or a Range as template arg... resulting in wrong results
        // more thinking is needed if they become useful...
        // auto& operator[](std::size_t idx) { return (*array_)[first_ + idx]; }
        // auto const& operator[](std::size_t idx) const { return (*array_)[first_ + idx]; }


        auto& operator[](std::size_t i) { return *(first_ + i); }
        auto& operator[](std::size_t i) const { return *(first_ + i); }

    private:
        Index first_;
        Index end_;
        Array* array_;
    };

    template<typename Array>
    MKN_KUL_FN_SIG_RET auto makeRange(Array& arr, std::size_t begin, std::size_t end)
    {
        return IndexRange{arr, begin, end};
    }

    template<typename Iterator>
    MKN_KUL_FN_SIG_RET Range<Iterator> makeRange(Iterator&& begin, Iterator&& end)
    {
        return Range{std::forward<Iterator>(begin), std::forward<Iterator>(end)};
    }


    template<typename Array, typename Index>
    MKN_KUL_FN_SIG_RET auto makeRange(IndexRange<Array, Index> irange)
    {
        auto& arr  = irange.array();
        auto begin = std::begin(arr) + irange.ibegin();
        auto end   = std::begin(arr) + irange.iend();
        return makeRange(begin, end);
    }

    template<typename Container>
    MKN_KUL_FN_SIG_RET auto makeIndexRange(Container& container)
    {
        return IndexRange<Container>{container, 0, container.size()};
    }

    template<typename Container>
    MKN_KUL_FN_SIG_RET auto makeRange(Container& container)
    {
        return makeRange(std::begin(container), std::end(container));
    }

} // namespace core
} // namespace damn

namespace damn::core
{
template<typename Box, typename Iterator>
struct BoxRange
{
    using iterator = Iterator;


    BoxRange& operator=(BoxRange const&) = default;
    BoxRange& operator=(BoxRange&&)      = default;
    BoxRange(BoxRange&&)                 = default;
    BoxRange(BoxRange const&)            = default;

    BoxRange(Box box, Iterator begin, Iterator end)
        : box_{box}
        , first_{begin}
        , end_{end}
    {
    }

    auto& begin() const { return first_; }
    auto& end() const { return end_; }
    auto& box() const { return box_; }
    auto size() const { return std::distance(first_, end_); }

private:
    Box box_;
    Iterator first_;
    Iterator end_;
};
} // namespace damn::core


#endif // DAMN_CORE_GEOMETRY_RANGE_HPP
