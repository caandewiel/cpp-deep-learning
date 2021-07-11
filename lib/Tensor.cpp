#include "Tensor.hpp"
#include <memory>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor_forward.hpp>

#include <xtensor-blas/xlinalg.hpp>

#include "TensorOp.hpp"

namespace pdl
{
template <TensorData T> Tensor<T>::Tensor(xt::xarray<T> data) : m_data(std::move(data))
{
    m_vertex = std::make_shared<Vertex>();
    m_vertex->value = this;
    m_gradient = xt::zeros_like(data);
}

template <TensorData T> Tensor<T>::Tensor(xt::xarray<T> data, std::shared_ptr<Vertex> vertexA) : m_data(std::move(data))
{
    m_vertex = std::make_shared<Vertex>();
    m_vertex->value = this;
    m_vertex->incoming.emplace_back(vertexA);

    vertexA->outgoing.emplace_back(m_vertex);
}

template <TensorData T>
Tensor<T>::Tensor(xt::xarray<T> data, const Tensor<T> &a, const Tensor<T> &b) : m_data(std::move(data))
{
    m_vertex = std::make_shared<Vertex>();
    m_vertex->value = this;
    m_vertex->incoming.emplace_back(a.m_vertex);
    m_vertex->incoming.emplace_back(b.m_vertex);

    a.m_vertex->outgoing.emplace_back(m_vertex);
    b.m_vertex->outgoing.emplace_back(m_vertex);
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

template <TensorData T> void Tensor<T>::backward()
{
    if (m_function != nullptr)
    {
        (*m_function)(*this);
    }
    else
    {
        m_gradient = xt::ones_like(m_data);
    }

    for (auto &vertex : m_vertex->incoming)
    {
        vertex->getValue<Tensor<T> &>().backward(m_gradient);
    }
}

template <TensorData T> void Tensor<T>::backward(const xt::xarray<T> &gradientAccumulated)
{
    if (m_function != nullptr)
    {
        (*m_function)(*this);
    }
    else
    {
        m_gradient = xt::ones_like(m_data);
    }

    m_gradient *= gradientAccumulated;

    for (auto &vertex : m_vertex->incoming)
    {
        vertex->getValue<Tensor<T> &>().backward(m_gradient);
    }
}

template <TensorData T> Vertex &Tensor<T>::getVertex() const
{
    return *m_vertex;
}

template <TensorData T> Tensor<T> operator*(Tensor<T> &a, Tensor<T> &b)
{
    ModuleMultiplication<T> operation;

    a.m_function = &gradientMultiplication<T>;
    b.m_function = &gradientMultiplication<T>;

    return operation.forward(a, b);
}

template <TensorData T> Tensor<T> operator+(Tensor<T> &a, Tensor<T> &b)
{
    ModuleAddition<T> operation;

    a.m_function = &gradientAddition<T>;
    b.m_function = &gradientAddition<T>;

    return operation.forward(a, b);
}

template class Tensor<float>;
template Tensor<float> operator+(Tensor<float> &a, Tensor<float> &b);
template Tensor<float> operator*(Tensor<float> &a, Tensor<float> &b);
} // namespace pdl