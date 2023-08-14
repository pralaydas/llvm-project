#ifndef CONVERTSTANDALONETOLOOPS_H_
#define CONVERTSTANDALONETOLOOPS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class MLIRContext;
class ModuleOp;
template <typename T> class OpPassBase;

void populateArithToStandaloneConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns);
} // namespace mlir

#endif // CONVERTSTANDALONETOLOOPS_H_
