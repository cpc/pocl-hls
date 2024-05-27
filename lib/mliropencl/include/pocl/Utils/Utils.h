//===- Utils.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POCL_MLIR_UTILS_H
#define POCL_MLIR_UTILS_H

// From polygeist Passes/Utils.h

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/IntegerSet.h>

static inline mlir::scf::IfOp
cloneWithResults (mlir::scf::IfOp op,
                  mlir::OpBuilder &rewriter,
                  mlir::IRMapping mapping = {})
{
  using namespace mlir;
  return scf::IfOp::create (rewriter, op.getLoc (), op.getResultTypes (),
                            mapping.lookupOrDefault (op.getCondition ()),
                            true);
}
static inline mlir::affine::AffineIfOp
cloneWithResults (mlir::affine::AffineIfOp op,
                  mlir::OpBuilder &rewriter,
                  mlir::IRMapping mapping = {})
{
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands ())
    lower.push_back (mapping.lookupOrDefault (o));
  return mlir::affine::AffineIfOp::create (rewriter, op.getLoc (),
                                           op.getResultTypes (),
                                           op.getIntegerSet (), lower, true);
}

static inline mlir::scf::IfOp
cloneWithoutResults (mlir::scf::IfOp op,
                     mlir::OpBuilder &rewriter,
                     mlir::IRMapping mapping = {},
                     mlir::TypeRange types = {})
{
  using namespace mlir;
  return scf::IfOp::create (rewriter, op.getLoc (), types,
                            mapping.lookupOrDefault (op.getCondition ()),
                            true);
}
static inline mlir::affine::AffineIfOp
cloneWithoutResults (mlir::affine::AffineIfOp op,
                     mlir::OpBuilder &rewriter,
                     mlir::IRMapping mapping = {},
                     mlir::TypeRange types = {})
{
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands ())
    lower.push_back (mapping.lookupOrDefault (o));
  return mlir::affine::AffineIfOp::create (rewriter, op.getLoc (), types,
                                           op.getIntegerSet (), lower, true);
}

static inline mlir::scf::ForOp
cloneWithoutResults (mlir::scf::ForOp op,
                     mlir::PatternRewriter &rewriter,
                     mlir::IRMapping mapping = {})
{
  using namespace mlir;
  return scf::ForOp::create (rewriter, op.getLoc (),
                             mapping.lookupOrDefault (op.getLowerBound ()),
                             mapping.lookupOrDefault (op.getUpperBound ()),
                             mapping.lookupOrDefault (op.getStep ()));
}
static inline mlir::affine::AffineForOp
cloneWithoutResults (mlir::affine::AffineForOp op,
                     mlir::PatternRewriter &rewriter,
                     mlir::IRMapping mapping = {})
{
  using namespace mlir;
  SmallVector<Value> lower;
  for (auto o : op.getLowerBoundOperands ())
    lower.push_back (mapping.lookupOrDefault (o));
  SmallVector<Value> upper;
  for (auto o : op.getUpperBoundOperands ())
    upper.push_back (mapping.lookupOrDefault (o));
  return mlir::affine::AffineForOp::create (
    rewriter, op.getLoc (), lower, op.getLowerBoundMap (), upper,
    op.getUpperBoundMap (), op.getStepAsInt ());
}

static inline void
clearBlock (mlir::Block *block, mlir::PatternRewriter &rewriter)
{
  for (auto &op : llvm::make_early_inc_range (llvm::reverse (*block)))
    {
      assert (op.use_empty () && "expected 'op' to have no uses");
      rewriter.eraseOp (&op);
    }
}

static inline mlir::Block *
getThenBlock (mlir::scf::IfOp op)
{
  return op.thenBlock ();
}
static inline mlir::Block *
getThenBlock (mlir::affine::AffineIfOp op)
{
  return op.getThenBlock ();
}
static inline mlir::Block *
getElseBlock (mlir::scf::IfOp op)
{
  return op.elseBlock ();
}
static inline mlir::Block *
getElseBlock (mlir::affine::AffineIfOp op)
{
  if (op.hasElse ())
    return op.getElseBlock ();
  else
    return nullptr;
}

static inline mlir::Region &
getThenRegion (mlir::scf::IfOp op)
{
  return op.getThenRegion ();
}
static inline mlir::Region &
getThenRegion (mlir::affine::AffineIfOp op)
{
  return op.getThenRegion ();
}
static inline mlir::Region &
getElseRegion (mlir::scf::IfOp op)
{
  return op.getElseRegion ();
}
static inline mlir::Region &
getElseRegion (mlir::affine::AffineIfOp op)
{
  return op.getElseRegion ();
}

static inline mlir::scf::YieldOp
getThenYield (mlir::scf::IfOp op)
{
  return op.thenYield ();
}
static inline mlir::affine::AffineYieldOp
getThenYield (mlir::affine::AffineIfOp op)
{
  return llvm::cast<mlir::affine::AffineYieldOp> (
    op.getThenBlock ()->getTerminator ());
}
static inline mlir::scf::YieldOp
getElseYield (mlir::scf::IfOp op)
{
  return op.elseYield ();
}
static inline mlir::affine::AffineYieldOp
getElseYield (mlir::affine::AffineIfOp op)
{
  return llvm::cast<mlir::affine::AffineYieldOp> (
    op.getElseBlock ()->getTerminator ());
}

static inline bool
inBound (mlir::scf::IfOp op, mlir::Value v)
{
  return op.getCondition () == v;
}
static inline bool
inBound (mlir::affine::AffineIfOp op, mlir::Value v)
{
  return llvm::any_of (op.getOperands (),
                       [&] (mlir::Value e) { return e == v; });
}
static inline bool
inBound (mlir::scf::ForOp op, mlir::Value v)
{
  return op.getUpperBound () == v;
}
static inline bool
inBound (mlir::affine::AffineForOp op, mlir::Value v)
{
  return llvm::any_of (op.getUpperBoundOperands (),
                       [&] (mlir::Value e) { return e == v; });
}
static inline bool
hasElse (mlir::scf::IfOp op)
{
  return op.getElseRegion ().getBlocks ().size () > 0;
}
static inline bool
hasElse (mlir::affine::AffineIfOp op)
{
  return op.getElseRegion ().getBlocks ().size () > 0;
}

/// Swap side of predicate
static arith::CmpIPredicate
swapPredicate (arith::CmpIPredicate pred)
{
  switch (pred)
    {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      return pred;
    case arith::CmpIPredicate::slt:
      return arith::CmpIPredicate::sgt;
    case arith::CmpIPredicate::sle:
      return arith::CmpIPredicate::sge;
    case arith::CmpIPredicate::sgt:
      return arith::CmpIPredicate::slt;
    case arith::CmpIPredicate::sge:
      return arith::CmpIPredicate::sle;
    case arith::CmpIPredicate::ult:
      return arith::CmpIPredicate::ugt;
    case arith::CmpIPredicate::ule:
      return arith::CmpIPredicate::uge;
    case arith::CmpIPredicate::ugt:
      return arith::CmpIPredicate::ult;
    case arith::CmpIPredicate::uge:
      return arith::CmpIPredicate::ule;
    }
  llvm_unreachable ("unknown cmpi predicate kind");
}

#endif
