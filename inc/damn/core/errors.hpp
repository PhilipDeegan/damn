#ifndef DAMN_CORE_ERRORS_HPP
#define DAMN_CORE_ERRORS_HPP

#include "damn/core/def.hpp"

#include <string>
#include <iostream>
#include <unordered_map>
#include <cstddef>


namespace damn::core
{
class Errors
{
public:
    NO_DISCARD static Errors& instance()
    {
        static Errors i;
        return i;
    }

    NO_DISCARD bool any() { return errors.size() > 0; }


    void log(std::string const key, std::string const val)
    {
        if (!errors.count(key))
        {
            std::cerr << key << " :" << val << std::endl;
            errors.emplace(key, val);
            error_count.emplace(key, 0);
        }

        error_count[key] = error_count[key] + 1;
    }


private:
    std::unordered_map<std::string, std::string> errors;
    std::unordered_map<std::string, std::size_t> error_count;
};

} // namespace damn::core


#endif /* DAMN_CORE_ERRORS_H */
