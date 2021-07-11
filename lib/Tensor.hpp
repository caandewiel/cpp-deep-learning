#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <functional>
#include <memory>
#include <sstream>
#include <type_traits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include "xtensor/xtensor_forward.hpp"

#include "xtensor/xbuilder.hpp"

#include "Traversable.hpp"
#include "Vertex.hpp"

namespace pdl
{
template <typename T>
concept TensorData = std::is_arithmetic<T>::value;

template <TensorData T> class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
public:
    Tensor(xt::xarray<T> data);
    Tensor(xt::xarray<T> data, Tensor<T> &a);
    Tensor(xt::xarray<T> data, Tensor<T> &a, Tensor<T> &b);

    ~Tensor()
    {
        std::cout << "Tensor deleted\n";
    }

    [[nodiscard]] const std::string toString() const;
    [[nodiscard]] const xt::xarray<T> &data() const;
    [[nodiscard]] const xt::xarray<T> &gradient() const;
    [[nodiscard]] xt::xarray<T> &gradient();
    [[nodiscard]] const std::function<xt::xarray<T>(Tensor<T> &)> *gradientFunction() const;
    [[nodiscard]] std::shared_ptr<Tensor<T>> getSharedPointer();

    void setGradientFunction(std::function<xt::xarray<T>(Tensor<T> &)> const *gradientFunction);

    void backward();

    friend std::ostream &operator<<(std::ostream &stream, const Tensor<T> &a)
    {
        return stream << a.toString();
    }

    template <TensorData U>
    friend std::shared_ptr<Tensor<U>> operator+(std::shared_ptr<Tensor<U>> a, std::shared_ptr<Tensor<U>> b);

    template <TensorData U>
    friend std::shared_ptr<Tensor<U>> operator*(std::shared_ptr<Tensor<U>> a, std::shared_ptr<Tensor<U>> b);

    std::vector<std::shared_ptr<Tensor<T>>> m_incoming;
    std::vector<std::shared_ptr<Tensor<T>>> m_outgoing;

private:
    xt::xarray<T> m_data;
    xt::xarray<T> m_gradient;

    std::function<xt::xarray<T>(Tensor<T> &)> const *m_gradientFunction = nullptr;

    bool m_isLeaf;

    void backward(const xt::xarray<T> &gradientAccumulated);
};

template <TensorData T> static std::shared_ptr<Tensor<T>> tensor(xt::xarray<T> data)
{
    return std::shared_ptr<Tensor<T>>(new Tensor<T>(std::move(data)));
}
} // namespace pdl

#endif // __TENSOR_H__