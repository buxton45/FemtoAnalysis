// GlobalCoulombFitter
// Purpose is to fulfill request in PWGCF approval to fit all
// four systems (XiKchP, AXiKchM, XiKchM, AXiKchP) simultaneously
// with a Coulomb only fit


#ifndef GLOBALCOULOMBFITTER_H
#define GLOBALCOULOMBFITTER_H


#include "CoulombFitter.h"

class GlobalCoulombFitter : public CoulombFitter {

public:
  GlobalCoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar = 0.3);
  virtual ~GlobalCoulombFitter();

  void LoadInterpHistFileOppSign(TString aFileBaseName);  //TODO should this be a virtual function?


  complex<double> BuildScatteringLength(AnalysisType aAnalysisType, double aKStar, double aReF0, double aImF0, double aD0);
  double InterpolateWfSquared(AnalysisType aAnalysisType, double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0);
  double GetFitCfContent(AnalysisType aAnalysisType, double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);
  double GetFitCfContentwStaticPairs(AnalysisType aAnalysisType, double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber);
  void CalculateChi2PML(int &npar, double &chi2, double *par);  //TODO change default to true when matrices are ready

protected:
  bool fInterpHistsLoadedOppSign;

  CoulombType fCoulombTypeOppSign;

  //------Histograms----- Note: Should be deleted if vectors are being built

  TFile *fInterpHistFileOppSign;

  TH2D* fGTildeRealHistOppSign;
  TH2D* fGTildeImagHistOppSign;

  TH3D* fHyperGeo1F1RealHistOppSign;
  TH3D* fHyperGeo1F1ImagHistOppSign;

  //---------------------------




#ifdef __ROOT__
  ClassDef(GlobalCoulombFitter, 1)
#endif
};



#endif
