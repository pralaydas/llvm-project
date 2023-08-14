#ifndef MYSTANDALONEPASSES_H_
#define MYSTANDALONEPASSES_H_

#include <memory>

namespace mlir {
class Pass;
namespace standalone {
void createLowerStandaloneToArithPass();
}
} // namespace mlir

#endif // MYSTANDALONEPASSES_H_