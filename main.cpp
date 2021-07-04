#include <array>
#include <iostream>
#include <vector>

#include "tensor.hpp"

template <class From, class To>
concept convertible_to = std::is_convertible_v<From, To> &&
    requires(std::add_rvalue_reference_t<From> (&f)()) {
  static_cast<To>(f());
};

template <class T, uint8_t... Dims> struct Tensor;

template <class T, uint8_t Dim> struct Tensor<T, Dim> {
  static constexpr size_t extent = 1;
  static constexpr size_t capacity = Dim;
};

template <class T, uint8_t Dim, uint8_t... Dims>
struct Tensor<T, Dim, Dims...> {
private:
  static constexpr size_t
  calculateCapacity(uint8_t first, convertible_to<uint8_t> auto... rest) {
    if constexpr (sizeof...(rest) == 0) {
      return first;
    } else {
      return first * calculateCapacity(rest...);
    }
  }

  static constexpr uint8_t
  calculateIndex(convertible_to<uint8_t> auto... indices) {
    std::array<uint8_t, sizeof...(Dims) + 1> allDimensions = {{Dim, Dims...}};
    std::array<uint8_t, sizeof...(indices)> allIndices = {
        {static_cast<unsigned char>(indices)...}};

    auto result = 0;

    for (size_t i = 0; i < allIndices.size() - 1; i++) {
      result += allIndices[i] * allDimensions[i + 1];
    }

    return result + allIndices.back();
  }

public:
  static constexpr size_t extent = 1 + sizeof...(Dims);
  static constexpr size_t capacity = calculateCapacity(Dim, Dims...);

private:
  std::array<T, capacity> m_data{};

public:
  Tensor<T, Dim, Dims...>() = default;
  Tensor<T, Dim, Dims...>(std::array<T, capacity> data) requires(
      data.size() == calculateCapacity(Dim, Dims...))
      : m_data(std::move(data)) {}

  constexpr decltype(auto)
  operator()(const convertible_to<uint8_t> auto... indices) {
    return m_data.at(calculateIndex(indices...));
  }
};

int main() {
  Tensor<float, 2, 3> tensor{{1, 2, 3, 4, 5, 6}};
  std::cout << tensor.extent << " - " << tensor.capacity << " - "
            << tensor(1, 2) << "\n";
  //   std::cout << tensor[1][2] << " - " << tensor.extent
  //             << "\n";

  return 0;
}
