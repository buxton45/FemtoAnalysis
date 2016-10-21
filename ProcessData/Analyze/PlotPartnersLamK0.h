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

/*
  virtual TCanvas* DrawPurity();
  virtual TCanvas* DrawKStarCfs();
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed);  // kSame, kRotSame, kMixed, kRotMixed};

  virtual TCanvas* DrawAvgSepCfs();
  virtual TCanvas* DrawAvgSepCfs(AnalysisType aAnalysisType, bool aDrawConj);

  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo);
*/
  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType);
  virtual TCanvas* DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType);
  virtual TCanvas* DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType);

private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamK0, 1)
#endif
};


#endif
