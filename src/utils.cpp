#include "petsc_interface.hpp"

#include <python_ngstd.hpp> 

namespace ngs_petsc_interface
{

  void InitializePETSc (string name, FlatArray<string> options)
  {
    int argc = options.Size()+1;
    const char* progname = name.c_str();
    typedef const char * pchar;
    Array<pchar> ptrs(argc+1);
    ptrs[0] = progname;
    for (auto k : Range(argc-1))
      ptrs[k+1] = options[k].c_str();
    ptrs.Last() = NULL;
    pchar * pptr = &ptrs[0];
    char** cpptr = (char**)pptr;
    PetscInitialize(&argc, &cpptr, NULL, NULL);
  }


  void FinalizePETSc () { PetscFinalize(); }


  string GetDefaultId ()
  {
    static int cnt = 0;
    return string("NgsPETScObject_"+to_string(cnt++));
  }


  void SetOptions (FlatArray<string> opts_vals, string prefix, PetscOptions petsc_options)
  {
    for (auto& opt_val : opts_vals) {
      auto pos = opt_val.find(" ");
      if (pos != string::npos) {
	auto namelen = pos;
	string pf_opt = string("-") + prefix + opt_val.substr(0, namelen);
	string val = opt_val.substr(pos + 1, opt_val.size() - 1 - namelen);
	PetscOptionsSetValue(petsc_options, pf_opt.c_str(), val.c_str());
      }
      else {
	string pf_opt = string("-") + prefix + opt_val;
	PetscOptionsSetValue(petsc_options, pf_opt.c_str(), NULL);
      }
    }
  }


  void ExportUtils (py::module &m)
  {

    m.def("Initialize", [&](string prog_name, py::kwargs kwargs) {
	Array<string> opts;
	auto ValStr = [&](const auto & Ob) -> string {
	  if (py::isinstance<py::str>(Ob))
	    return Ob.template cast<string>();
	  if (py::isinstance<py::bool_>(Ob))
	    return Ob.template cast<bool>()==true ? "1" : "0";
	  if (py::isinstance<py::float_>(Ob) ||
	      py::isinstance<py::int_>(Ob))
	    return py::str(Ob).cast<string>();
	  return "COULD_NOT_CONVERT";
	};
	for (auto item : kwargs) {
	  string name = "-" + item.first.cast<string>();
	  opts.Append(name);
	  // string val = item.second.cast<string>();
	  string val = ValStr(item.second);
	  opts.Append(val);
	}
	InitializePETSc(prog_name, opts);
      }, py::arg("prog_name") = string("NGsPETScInterface"));


    m.def("Finalize", [&]() {
	FinalizePETSc();
      });

  }


} // ngs_petsc_interface
