#include "damn/initializer/data_provider.hpp"


namespace damn::initializer
{
DAMNDictHandler& DAMNDictHandler::INSTANCE()
{
    static DAMNDictHandler handler;
    return handler;
}
} // namespace damn::initializer
