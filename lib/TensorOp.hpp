#ifndef __TENSOROP_H__
#define __TENSOROP_H__

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

#include "Tensor.hpp"
#include "xtensor/xbuilder.hpp"

namespace pdl
{
class Module
{
};

template <typename T> class ModuleBinary
{
public:
    virtual Tensor<T> forward(const Tensor<T> &a, const Tensor<T> &b) = 0;
};

template <typename T>
static std::function<void(Tensor<T> &x)> gradientMultiplication = [](Tensor<T> &x) {
    const auto &parent = **x.getVertex().outgoing.begin();

    //     static_assert(parent.incoming.size() == 2, "Invalid graph");

    for (const auto &child : parent.incoming)
    {
        if (child.get() != &x.getVertex())
        {
            x.gradient() = child->template getValue<Tensor<T> &>().data();
        }
    }
};

template <typename T>
static std::function<void(Tensor<T> &x)> gradientAddition =
    [](Tensor<T> &x) { x.gradient() = xt::ones_like(x.data()); };

template <typename T> class ModuleMultiplication : protected ModuleBinary<T>
{
public:
    Tensor<T> forward(const Tensor<T> &a, const Tensor<T> &b) override
    {
        return {a.data() * b.data(), a, b};
    }
};

template <typename T> class ModuleAddition : protected ModuleBinary<T>
{
public:
    Tensor<T> forward(const Tensor<T> &a, const Tensor<T> &b) override
    {
        return {a.data() + b.data(), a, b};
    }
};
} // namespace pdl

#endif // __TENSOROP_H__