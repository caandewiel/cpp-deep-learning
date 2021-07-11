#include "Tensor.hpp"
#include <memory>
#include <utility>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor_forward.hpp>

#include <xtensor-blas/xlinalg.hpp>

#include "TensorOp.hpp"

namespace pdl
{
template <TensorData T> Tensor<T>::Tensor(xt::xarray<T> data) : m_data(std::move(data))
{
    m_isLeaf = true;
}

template <TensorData T> Tensor<T>::Tensor(xt::xarray<T> data, Tensor<T> &a) : m_data(std::move(data))
{
    m_isLeaf = false;
    m_incoming.emplace_back(a.getSharedPointer());
}

template <TensorData T> Tensor<T>::Tensor(xt::xarray<T> data, Tensor<T> &a, Tensor<T> &b) : m_data(std::move(data))
{
    m_isLeaf = false;
    m_incoming.emplace_back(a.getSharedPointer());
    m_incoming.emplace_back(b.getSharedPointer());
}

template <TensorData T> const std::string Tensor<T>::toString() const
{
    std::stringstream result;
    result << m_data << m_gradient;

    return result.str();
}

template <TensorData T> const xt::xarray<T> &Tensor<T>::data() const
{
    return m_data;
}

template <TensorData T> const xt::xarray<T> &Tensor<T>::gradient() const
{
    return m_gradient;
}

template <TensorData T> xt::xarray<T> &Tensor<T>::gradient()
{
    return m_gradient;
}

template <TensorData T> const std::function<xt::xarray<T>(Tensor<T> &)> *Tensor<T>::gradientFunction() const
{
    return m_gradientFunction;
}

template <TensorData T>
void Tensor<T>::setGradientFunction(std::function<xt::xarray<T>(Tensor<T> &)> const *gradientFunction)
{
    m_gradientFunction = gradientFunction;
}

template <TensorData T> std::shared_ptr<Tensor<T>> Tensor<T>::getSharedPointer()
{
    return std::enable_shared_from_this<Tensor<T>>::shared_from_this();
}

template <TensorData T> void Tensor<T>::backward()
{
    xt::xarray<T> currentGradient;

    if (m_gradientFunction != nullptr)
    {
        currentGradient = (*m_gradientFunction)(*this);
    }
    else
    {
        currentGradient = xt::ones_like(m_data);
    }

    if (m_isLeaf)
    {
        m_gradient = currentGradient;
    }

    for (auto &tensor : m_incoming)
    {
        tensor->backward(currentGradient);
    }
}

template <TensorData T> void Tensor<T>::backward(const xt::xarray<T> &gradientAccumulated)
{
    xt::xarray<T> currentGradient;

    if (m_gradientFunction != nullptr)
    {
        currentGradient = (*m_gradientFunction)(*this) * gradientAccumulated;
    }
    else
    {
        currentGradient = xt::ones_like(m_data) * gradientAccumulated;
    }

    if (m_isLeaf)
    {
        m_gradient = currentGradient;
    }

    for (auto &tensor : m_incoming)
    {
        tensor->backward(currentGradient);
    }
}

template <TensorData T> std::shared_ptr<Tensor<T>> operator*(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
{
    return pdl::ops::multiply::forward(*a, *b);
}

template <TensorData T> std::shared_ptr<Tensor<T>> operator+(std::shared_ptr<Tensor<T>> a, std::shared_ptr<Tensor<T>> b)
{
    return pdl::ops::add::forward(*a, *b);
}

template class Tensor<float>;
template std::shared_ptr<Tensor<float>> operator+(std::shared_ptr<Tensor<float>> a, std::shared_ptr<Tensor<float>> b);
template std::shared_ptr<Tensor<float>> operator*(std::shared_ptr<Tensor<float>> a, std::shared_ptr<Tensor<float>> b);
} // namespace pdl