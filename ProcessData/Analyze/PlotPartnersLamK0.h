///////////////////////////////////////////////////////////////////////////
// PlotPartnersLamK0:                                                   //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PLOTPARTNERSLAMK0_H
#define PLOTPARTNERSLAMK0_H

#include "PlotPartners.h"
class PlotPartners;



class PlotPartnersLamK0 : public PlotPartners {

public:
  PlotPartnersLamK0(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aType=kTrain, int aNPartialAnalysis=2, TString aDirNameModifier="");
  PlotPartnersLamK0(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aType=kTrain, int aNPartialAnalysis=2, TString aDirNameModifier="");
  virtual ~PlotPartnersLamK0();


  virtual TCanvas* DrawPurity(bool aSaveImage=false);
  virtual TCanvas* DrawKStarCfs(bool aSaveImage=false);
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed, bool aSaveImage=false);  // kSame, kRotSame, kMixed, kRotMixed};

  virtual TCanvas* DrawAvgSepCfs(bool aSaveImage=false);
  virtual TCanvas* DrawAvgSepCfs(AnalysisType aAnalysisType, bool aSaveImage=false);

  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage=false);

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false);
  virtual TCanvas* DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false);
  virtual TCanvas* DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false);

private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamK0, 1)
#endif
};


#endif
