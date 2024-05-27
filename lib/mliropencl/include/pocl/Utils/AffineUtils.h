//===- AffineUtils.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POCL_AFFINEUTILS_H_
#define POCL_AFFINEUTILS_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>

#include <isl/aff.h>
#include <isl/set.h>

namespace mlir
{

mlir::affine::AffineValueMap getAVM (mlir::Operation *op);

class IslAnalysis
{
public:
  std::optional<llvm::SmallVector<isl_aff *> >
  getAffExprs (mlir::Operation *op, mlir::affine::AffineValueMap avm);

  std::optional<llvm::SmallVector<isl_aff *> >
  getAffExprs (mlir::Operation *op);

  isl_map *getAccessMap (mlir::Operation *op);

  isl_set *getDomain (Operation *op);

  isl_set *getMemrefShape (MemRefType ty);

  ~IslAnalysis ();
  IslAnalysis ();

  isl_ctx *
  getCtx ()
  {
    return ctx;
  }

private:
  isl_ctx *ctx;
};

template <typename T> class IslScopeFree
{
public:
  T obj;
  IslScopeFree (T obj) : obj (obj) {}
  ~IslScopeFree () { isl_set_free (obj); }
};

void populateAffineExprSimplificationPatterns (IslAnalysis &islAnalysis,
                                               RewritePatternSet &patterns);

} // namespace mlir

#endif // POCL_AFFINEUTILS_H_
