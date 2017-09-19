#include "InterpolationHistograms.h"
#include <math.h>

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv)
{

  TString tSaveBaseName = "InterpHists";

  int tNbinsKStar = 400;
  double tKStarMin = 0.;
  double tKStarMax = 1.0;

  int tNbinsRStar = 500;
  double tRStarMin = 0.;
  double tRStarMax = 50.;

//  int tNbinsTheta = 180;
//  double tThetaMin = 0.;
//  double tThetaMax = 1.*M_PI;

  double tBinWidthTheta = M_PI/180;
  int tNbinsTheta = 184;  //two bins under 0 and two bins over pi, to be safe
  double tThetaMin = 0. - 2.*tBinWidthTheta;
  double tThetaMax = 1.*M_PI + 2*tBinWidthTheta;

  int tNbinsReF0 = 100;
  double tReF0Min = -10.;
  double tReF0Max = 10.;

  int tNbinsImF0 = 100;
  double tImF0Min = -10.;
  double tImF0Max = 10.;

  int tNbinsD0 = 100;
  double tD0Min = -10.;
  double tD0Max = 10.;

  //---------------------------------------
  AnalysisType tAnalysisType = kXiKchP;

  InterpolationHistograms* myCreator = new InterpolationHistograms(tSaveBaseName,tAnalysisType);

  myCreator->SetKStarBinning(tNbinsKStar,tKStarMin,tKStarMax);
  myCreator->SetRStarBinning(tNbinsRStar,tRStarMin,tRStarMax);
  myCreator->SetThetaBinning(tNbinsTheta,tThetaMin,tThetaMax);
  myCreator->SetReF0Binning(tNbinsReF0,tReF0Min,tReF0Max);
  myCreator->SetImF0Binning(tNbinsImF0,tImF0Min,tImF0Max);
  myCreator->SetD0Binning(tNbinsD0,tD0Min,tD0Max);

//  myCreator->BuildAndSaveAll();  //NO! DON'T USE THIS
  myCreator->BuildAndSaveAllOthers();
//  myCreator->BuildAndSaveSplitScatteringLengthHistograms();
//  myCreator->BuildAndSaveLednickyHFunction();

  return 0;
}
