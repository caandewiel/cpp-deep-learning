#ifndef __VERTEX_H__
#define __VERTEX_H__

#include <memory>
#include <stdexcept>
#include <vector>

#include "Traversable.hpp"

namespace pdl
{
struct Vertex
{
    std::vector<std::shared_ptr<Vertex>> incoming;
    std::vector<std::shared_ptr<Vertex>> outgoing;
    Traversable *value;

    template <typename T> T &getValue()
    {
        return static_cast<T>(*value);
    }
};
} // namespace pdl

#endif // __VERTEX_H__