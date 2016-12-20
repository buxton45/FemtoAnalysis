///////////////////////////////////////////////////////////////////////////
// InterpolateGPU:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef INTERPOLATEGPU_H
#define INTERPOLATEGPU_H

//includes and any constant variable declarations
#include <stdio.h>
#include <iostream>
#include <vector>



#include "timer.h"
#include "utils.h"


using std::cout;
using std::endl;
using std::vector;

class InterpolateGPU {

public:
  //Any enum types



  //Constructor, destructor, copy constructor, assignment operator
  InterpolateGPU(); //TODO delete this constructor.  Only here for testing
  virtual ~InterpolateGPU();

  vector<double> RunBilinearInterpolate(vector<vector<double> > &aPairsIn, vector<vector<double> > &a2dVecIn);
  double* RunBilinearInterpolate(double* host_out, double* aPairsIn, double* a2dVecIn);


private:

};


//inline stuff

#endif
