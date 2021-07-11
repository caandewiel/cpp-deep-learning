set(CMAKE_CXX_STANDARD 20)

add_library(pdl Tensor.hpp Tensor.cpp TensorOp.hpp TensorOp.cpp Vertex.hpp Vertex.cpp Traversable.hpp)
target_link_libraries(pdl xtensor xtensor-blas ${BLAS_LIBRARIES})