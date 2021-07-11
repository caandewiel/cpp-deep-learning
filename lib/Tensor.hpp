#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <functional>
#include <memory>
#include <sstream>
#include <type_traits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

#include "xtensor/xbuilder.hpp"

#include "Traversable.hpp"
#include "Vertex.hpp"
#include "xtensor/xtensor_forward.hpp"

namespace pdl
{
template <typename T>
concept TensorData = std::is_arithmetic<T>::value;

class Module;

template <TensorData T> class Tensor : public Traversable
{
public:
    Tensor(xt::xarray<T> data);
    Tensor(xt::xarray<T> data, std::shared_ptr<Vertex> vertexA);
    Tensor(xt::xarray<T> data, const Tensor<T> &a, const Tensor<T> &b);

    [[nodiscard]] const std::string toString() const;
    [[nodiscard]] const xt::xarray<T> &data() const;
    [[nodiscard]] const xt::xarray<T> &gradient() const;
    [[nodiscard]] xt::xarray<T> &gradient();
    [[nodiscard]] Vertex &getVertex() const;

    void backward();

    friend std::ostream &operator<<(std::ostream &stream, const Tensor<T> &a)
    {
        return stream << a.toString();
    }

    template <TensorData U>
    friend Tensor<U> operator+ (Tensor<U> &a, Tensor<U> &b);

    template <TensorData U>
    friend Tensor<U> operator* (Tensor<U> &a, Tensor<U> &b);

private:
    xt::xarray<T> m_data;
    xt::xarray<T> m_gradient;
    std::shared_ptr<Vertex> m_vertex;
    std::function<void(Tensor<T> &)> *m_function = nullptr;

    void backward(const xt::xarray<T> &gradientAccumulated);
};
} // namespace pdl

#endif // __TENSOR_H__