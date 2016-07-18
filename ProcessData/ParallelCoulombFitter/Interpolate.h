///////////////////////////////////////////////////////////////////////////
// Interpolate:                                                        //
///////////////////////////////////////////////////////////////////////////

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

//includes and any constant variable declarations
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <complex>

#include "timer.h"
#include "utils.h"


#include "TH1.h"
#include "TH2.h"
#include "TH2D.h"
#include "TString.h"

#include "TF1.h"
#include "TH1F.h"
#include "TH3.h"
#include "THn.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TObjectTable.h"
#include "TTree.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

#include "Types.h"
#include "ParallelTypes.h"

using std::cout;
using std::endl;
using std::vector;

#include "InterpolateGPU.h"
class InterpolateGPU;

class Interpolate {

public:
  //Any enum types



  //Constructor, destructor, copy constructor, assignment operator
  Interpolate(); //TODO delete this constructor.  Only here for testing
  virtual ~Interpolate();

  int GetBinNumber(double aBinSize, int aNbins, double aValue);
  int GetBinNumber(int aNbins, double aMin, double aMax, double aValue);
  int GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue);

  void BuildPairKStar3dVec(TString aPairKStarNtupleLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, BFieldType aBFieldType, int aNbinsKStar, double aKStarMin, double aKStarMax);
  void MakeOtherArrays(TString aFileBaseName);
  vector<vector<double> > BuildPairs();

  vector<double> RunBilinearInterpolateSerial(vector<vector<double> > &d_PairsIn, vector<vector<double> > &d_2dVecIn);
  vector<double> RunBilinearInterpolateParallel(td2dVec &aPairs);

  vector<vector<double> > ReturnGTildeReal();

private:
  BinInfoGTilde fGTildeInfo;
/*
  double fGTildeReal[160][100];
  double fGTildeImag[160][100];
*/
  vector<vector<double> > fGTildeReal;
  vector<vector<double> > fGTildeImag;

  vector<vector<vector<double> > > fPairKStar3dVec;
  vector<vector<double> > fPairs2dVec;

  InterpolateGPU* fInterpolateGPU;

#ifdef __ROOT__
  ClassDef(Interpolate, 1)
#endif

};


//inline stuff
inline vector<vector<double> > Interpolate::ReturnGTildeReal() {return fGTildeReal;}

#endif
