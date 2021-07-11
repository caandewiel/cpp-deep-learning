#ifndef __VERTEX_H__
#define __VERTEX_H__

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Traversable.hpp"

namespace pdl
{
struct Vertex
{
    ~Vertex()
    {
	    std::cout << "Vertex deleted\n";
    }

    std::vector<std::shared_ptr<Vertex>> incoming;
    std::vector<std::shared_ptr<Vertex>> outgoing;
    std::unique_ptr<Traversable> value;

    template <typename T> T &getValue()
    {
        return static_cast<T>(*value);
    }
};
} // namespace pdl

#endif // __VERTEX_H__