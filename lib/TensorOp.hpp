#ifndef __TENSOROP_H__
#define __TENSOROP_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor_forward.hpp>

#include "Tensor.hpp"
#include "xtensor/xmath.hpp"

namespace pdl::ops
{
namespace multiply
{
template <typename T>
const static std::function<xt::xarray<T>(Tensor<T> &x)> backward = [](Tensor<T> &x) {
    std::cout << "multiply\n";
    const auto &parent = **x.m_outgoing.begin();

    for (const auto &child : parent.m_incoming)
    {
        if (child.get() != &x)
        {
            return child->data();
        }
    }

    // This happens when a tensor is multiplied by itself.
    return static_cast<xt::xarray<T>>(2 * x.data());
};

template <TensorData T> static std::shared_ptr<Tensor<T>> forward(Tensor<T> &a, Tensor<T> &b)
{
    auto result = std::shared_ptr<Tensor<T>>(new Tensor<T>(a.data() * b.data(), a, b));

    a.setGradientFunction(&backward<T>);
    b.setGradientFunction(&backward<T>);
    a.m_outgoing.emplace_back(result);
    b.m_outgoing.emplace_back(result);

    return result;
}
} // namespace multiply

namespace add
{
template <TensorData T>
const static std::function<xt::xarray<T>(Tensor<T> &x)> backward = [](Tensor<T> &x) {
    std::cout << "add\n";
    return xt::ones_like(x.data());
};

template <TensorData T> static std::shared_ptr<Tensor<T>> forward(Tensor<T> &a, Tensor<T> &b)
{
    a.setGradientFunction(&backward<T>);
    b.setGradientFunction(&backward<T>);
    auto result = std::shared_ptr<Tensor<T>>(new Tensor<T>(a.data() + b.data(), a, b));

    return result;
}
} // namespace add

namespace exp
{
template <TensorData T>
const static std::function<xt::xarray<T>(Tensor<T> &x)> backward = [](Tensor<T> &x) { return xt::ones_like(x.data()); };

template <TensorData T> static std::shared_ptr<Tensor<T>> forward(Tensor<T> &a)
{
    a.setGradientFunction(&backward<T>);
    auto result = std::shared_ptr<Tensor<T>>(new Tensor<T>(xt::exp(a.data()), a));

    return result;
}
} // namespace exp

namespace log
{
template <TensorData T>
const static std::function<xt::xarray<T>(Tensor<T> &x)> backward = [](Tensor<T> &x) { return 1 / x.data(); };

template <TensorData T> static std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> a)
{
    auto result = std::shared_ptr<Tensor<T>>(new Tensor<T>(xt::log(a->data()), *a));
    a->setGradientFunction(&backward<T>);

    return result;
}
} // namespace log

} // namespace pdl::ops

#endif // __TENSOROP_H__