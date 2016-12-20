///////////////////////////////////////////////////////////////////////////
// MathematicaSession:                                                   //
//  Do Mathematica magic in c++ programs!                                //
///////////////////////////////////////////////////////////////////////////

//***!!!***!!!***!!! IMPORTANT ***!!!***!!!***!!!***!!!
//     DO NOT name functions the same as Mathematica
//     This gives strange results.  Ex. this is why, in GetGamme
//     I call the function MyGamma, instead of Gamma!


#ifndef MATHEMATICASESSION_H
#define MATHEMATICASESSION_H

#include <stdio.h>
#include <stdlib.h>

#include "wstp.h"

#include <iostream>
#include <complex>
#include "math.h"

using namespace std;

class MathematicaSession {

public:
  enum HFuncType {kHPlus=0, kHMinus=1};

  MathematicaSession();
  virtual ~MathematicaSession();

  void InitializeSession();
  void EndSession();

  void init_and_openlink( int argc, char* argv[]);
  void discardResult(WSLINK lp);
  void error( WSLINK lp);
//  void deinit( void);
//  void closelink( void);

  complex<double> GetGamma(complex<double> aCmplx);
  complex<double> GetDiGamma(complex<double> aCmplx);
  complex<double> GetHyperGeo1F1(complex<double> aA, complex<double> aB, complex<double> aZ);

  complex<double> GetCoulombHpmFunction(double aPhaseShift, double aEta, double aRho, HFuncType aPlusOrMinus, double aL=0);  //L=0 for s-wave
  complex<double> GetCoulombRegularFWaveFunction(double aPhaseShift, double aEta, double aRho, double aL=0);
  complex<double> GetCoulombSingularGWaveFunction(double aPhaseShift, double aEta, double aRho, double aL=0);




  //inline (i.e. simple functions)





private:
//  WSENV ep;
//  WSLINK lp;






#ifdef __ROOT__
  ClassDef(MathematicaSession, 1)
#endif
};



//inline stuff


#endif
