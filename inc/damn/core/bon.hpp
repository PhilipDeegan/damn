#ifndef DAMN_CORE_BON_HPP
#define DAMN_CORE_BON_HPP

// mkn.kul's bon::Node as a tree of paths: "a/b/c" pairs (a generated table, e.g. bootstrap's
// entries) made into nested maps, a node found by its path, a leaf read as T (kul's from_string)
//
//   auto const t = damn::bon::tree(entries);              // {a: {b: {c: v}}}
//   damn::bon::find(t, "a/b")                             // the {c: v} node, null if none
//   damn::bon::get<double>(t, "a/b/c", 1.)                // v as a double, 1. if absent or not one
//   damn::bon::get<double>(entries, "a/b/c", 1.)          // the same over the flat table

#include "mkn/kul/bon.hpp" // IWYU pragma: export
#include "mkn/kul/lang/cpp/compiler/def.hpp"
#include "mkn/kul/string.hpp" // IWYU pragma: keep

#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace damn::bon
{
using Node = mkn::kul::bon::Node;

namespace detail
{
    // the child of a map node keyed key, made (an empty map) if not there
    inline Node& child(Node& node, std::string_view const key)
    {
        if (!node.is_map())
            node.value = Node::Map{};
        auto& map = std::get<Node::Map>(node.value);
        for (auto& [k, n] : map)
            if (k == key)
                return n;
        map.push_back({std::string{key}, Node{Node::Map{}}});
        return map.back().node;
    }

    // path's parts, each up to the next sep (no empty parts: "a//b" is a/b)
    template<typename Fn>
    void each_part(std::string_view path, char const sep, Fn&& fn)
    {
        while (!path.empty())
        {
            auto const end  = path.find(sep);
            auto const part = path.substr(0, end);
            if (!part.empty())
                fn(part);
            if (end == std::string_view::npos)
                break;
            path.remove_prefix(end + 1);
        }
    }

    // a leaf as T (operator>>), fallback if it is not one; a string as it is (all of it)
    template<typename T>
    T parse(std::string const& value, T const& fallback)
    {
        try
        {
            return mkn::kul::from_string<T>(value);
        }
        catch (std::exception const&)
        {
            return fallback;
        }
    }

    inline std::string parse(std::string const& value, std::string const& /*fallback*/)
    {
        return value;
    }
} // namespace detail

// pairs of (path, value), paths sep separated, as one tree of maps with string leaves: a path
// that is also another's prefix keeps the longer one's map (a leaf and children can't share a
// node: tables keep leaves at the ends, "a/b/ok" beside "a/b/x")
template<typename Pairs>
MKN_KUL_FN_SIG_RET Node tree(Pairs const& pairs, char const sep = '/')
{
    Node root{Node::Map{}};
    for (auto const& [path, value] : pairs)
    {
        Node* node = &root;
        detail::each_part(std::string_view{path}, sep,
                          [&](auto const part) { node = &detail::child(*node, part); });
        if (node != &root and (!node->is_map() or node->map().empty()))
            node->value = std::string{value};
    }
    return root;
}

// the node at path (sep separated), null if a part is missing; "" is root
MKN_KUL_FN_SIG_RET inline Node const* find(Node const& root, std::string_view const path,
                                           char const sep = '/')
{
    Node const* node = &root;
    detail::each_part(path, sep, [&](auto const part) {
        if (node)
            node = node->find(std::string{part});
    });
    return node;
}

// the leaf at path as T (from_string), fallback if missing, not a leaf, or not a T
template<typename T>
MKN_KUL_FN_SIG_RET T get(Node const& root, std::string_view const path, T const& fallback,
                         char const sep = '/')
{
    auto const* const node = find(root, path, sep);
    return node and node->is_string() ? detail::parse(node->str(), fallback) : fallback;
}

// the same over a flat table of (path, value) pairs: no tree made
template<typename T, typename Pairs>
    requires(!std::is_same_v<Pairs, Node>)
MKN_KUL_FN_SIG_RET T get(Pairs const& pairs, std::string_view const path, T const& fallback)
{
    for (auto const& [k, v] : pairs)
        if (std::string_view{k} == path)
            return detail::parse(std::string{v}, fallback);
    return fallback;
}

} // namespace damn::bon

#endif // DAMN_CORE_BON_HPP
