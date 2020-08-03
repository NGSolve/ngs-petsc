#ifndef FILE_NGSPETSC_MATVEC_HPP
#define FILE_NGSPETSC_MATVEC_HPP

/**
   Converting Matrices and Vectors.
 **/

#include <comp.hpp>
#include "petsc.h"
#include <python_ngstd.hpp> 

namespace ngs_petsc_interface
{

  /** Object Space, NGSolve- or PETSc-side **/
  enum OBS = { NGS, PETSC };

  /** What is a "vector" in an object space **/
  template<OBS T> using TVEC = void;
  template<> using TVEC<NGS> = shared_ptr<ngs::BaseVector>;
  template<> using TVEC<PETSC> = PETScVec;

  /** An operator takes a vector in Space A as input and a vector in space B as output **/
  enum OP_TYPE = { N2N, N2P, P2N, P2P };
  template<OBS A, OBS B> class Operator
  {
  public:
    using OBS_IN  = A;
    using OBS_OUT = B;
    constexpr OP_TYPE Type () const
    {
      if (A == NGS   && B == NGS)   return N2N;
      if (A == NGS   && B == PETSC) return N2P;
      if (A == PETSC && B == NGS)   return P2N;
      if (A == PETSC && B == PETSC) return P2P;
    };
    virtual void Apply (obs_traits<OBS_IN>::TVEC input, obs_traits<OBS_OUT>::TVEC output) = 0;
  };

  /** This is just a wrapper around an NGSolve-BaseMatrix! **/
  class NgsMatWrapper : public Operator<NGS, NGS>
  {
    virtual void Apply (shared_ptr<ngs::BaseVector> input, shared_ptr<ngs::BaseVector> output) override
    {
      ngs_mat->Mult(input, output);
    }
  protected:
    shared_ptr<ngs::BaseMatrix> ngs_mat;
  };

  /** **/
  template<> class Operator<PETSC, PETSC>
  {
    
  };

  template<OBS A, OBS B, OBS C> class MultOp : public Operator<A,C>
  {
    virtual void Apply (obs_traits<A>::TVEC input, obs_traits<C>::TVEC output) override
    {
      left->Apply(input, temp_vec);
      right->Apply(temp_vec, output);
    };
  protected:
    shared_ptr<Operator<A,B>> left;
    shared_ptr<Operator<B,C>> right;
  };

  template<OBS A, OBS B> class AddOp : public Operator<A,B>
  {
    virtual void Apply (obs_traits<A>::TVEC input, obs_traits<C>::TVEC output) override
    {
      for (auto op : ops)
	op->Add(input, temp_vec);
    };
  protected:
    Array<shared_ptr<Operator<A,B>>> ops;
  };
  
  
  class NGs2PETScVecMap;
  
  enum PETScMatType = {...};
  class PETScMat;


  class PETScKSP : public BaseMatrix
  {
    PETScKSP (shared_ptr<PETScMat> mat, Array<string> opts);
    PETScKSP (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<BitArray> freedofs, Array<string> opts);
    void SetNearNullSpace (ngs::Array<ngs::BaseVector> vecs);
    void SetPC (PETScPC pc);
    void Finalize();
    void Solve(rhs, sol);
    py::dict Results();

    CreateRowVector ();
    CreateColVector ();
    CreateVector ();

    Mult ();

  protected:
    KSP ksp;
    PETScOptions opts; // or: string petsc_id (prefix for specific options)
    shared_ptr<Ngs2PETScVecMap> row_map, col_map;
    shared_ptr<BaseMatrix> ngs_mat;
  };
  
  // Maps between ngsolve- and PETSc vector
  class Ngs2PETScVecMap
  {
    shared_ptr<ngs::ParallelDofs> pardofs;
    size_t low, high, loc, glob;
    Array<PetscInt> loc_inds, glob_nums;
    Array<PetscScalar> buf;
  public:
    /**
       Have [low .. high) of PETSc vec. These map to set DOFs of take_dofs.
       _glob .. global numset  (!= pardofs->NGofGlob()!)
    **/
    Ngs2PETScVecMap (shared_ptr<ngs::ParallelDofs> _pardofs,
		     shared_ptr<ngs::BitArray> take_dofs,
		     size_t _low, size_t _high, size_t _glob);

    shared_ptr<ngs::ParallelDofs> GetParallelDofs () const { return pardofs; }

    PETScVec CreatePETScVec () const;

    void Ngs2PETSc (shared_ptr<ngs::BaseVector> ngs_vec, ::Vec petsc_vec)
    {
      ngs_vec->Cumulate();
      auto fv = ngs_vec->FVDouble();
      for (auto k : Range(loc))
  	buf[k] = fv(loc_inds[k]);
      VecSetValues(petsc_vec, loc, &glob_nums[0], &buf[0], INSERT_VALUES);
      VecAssemblyBegin(petsc_vec);
      VecAssemblyEnd(petsc_vec);
    }
    // petsc -> ngsolve
    void PETSc2Ngs (shared_ptr<ngs::BaseVector> ngs_vec, ::Vec petsc_vec)
    {
      ngs_vec->Distribute();
      // SZ
      // This is odd, and rarely used from PETSc. Also, off-processor retrieval is disabled
      // Can you use VecGetArrayRead/VecRestoreArrayRead semantics?
      // In alternative, if Ngs2PETSc and PETSc2NGs needs off-process data, you can use
      // the VecScatter object
      VecGetValues(petsc_vec, loc, &glob_nums[0], &buf[0]);
      auto fv = ngs_vec->FVDouble();
      fv = 0.0;
      for (auto k : Range(loc))
  	fv(loc_inds[k]) = buf[k];
    }
  };
  
} // namespace ngs_petsc_interface

#endif
