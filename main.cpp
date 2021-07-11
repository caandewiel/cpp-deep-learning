#include <memory>
#include "lib/Tensor.hpp"
#include "lib/TensorOp.hpp"

int main() {
  auto a = pdl::tensor<float>({{2, 3}, {2, 3}});
  auto b = pdl::tensor<float>({{3, 4}, {3, 4}});
  auto c = pdl::tensor<float>({{4, 5}, {4, 5}});
  auto d = pdl::tensor<float>({{5, 6}, {5, 6}});

  auto e = a * b;
  auto f = e * c;
  auto g = pdl::ops::log::forward(f + d);

  g->backward();

  std::cout << a->gradient() << "\n";
  std::cout << b->gradient() << "\n";
  std::cout << c->gradient() << "\n";
  std::cout << d->gradient() << "\n";
  std::cout << e->gradient() << "\n";
  std::cout << f->gradient() << "\n";
  std::cout << g->gradient() << "\n";

  std::cout << g->data() << "\n";

  return 0;
}
