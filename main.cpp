#include "lib/Tensor.hpp"

int main() {
  pdl::Tensor<float> a = {{2, 3}};
  pdl::Tensor<float> b = {{3, 4}};
  pdl::Tensor<float> c = {{4, 5}};
  pdl::Tensor<float> d = {{5, 6}};

  auto e = a * b;
  auto f = e * c;
  auto g = f + d;

  g.backward();

  std::cout << a.gradient() << "\n";
  std::cout << b.gradient() << "\n";
  std::cout << c.gradient() << "\n";
  std::cout << d.gradient() << "\n";
  std::cout << e.gradient() << "\n";
  std::cout << f.gradient() << "\n";
  std::cout << g.gradient() << "\n";

  std::cout << g.data() << "\n";

  return 0;
}
