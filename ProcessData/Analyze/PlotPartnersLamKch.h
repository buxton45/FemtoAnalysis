///////////////////////////////////////////////////////////////////////////
// PlotPartnersLamKch:                                                   //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef PLOTPARTNERSLAMKCH_H
#define PLOTPARTNERSLAMKCH_H

#include "PlotPartners.h"
class PlotPartners;



class PlotPartnersLamKch : public PlotPartners {

public:
  PlotPartnersLamKch(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aType=kTrain, int aNPartialAnalysis=2, TString aDirNameModifier="");
  PlotPartnersLamKch(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aType=kTrain, int aNPartialAnalysis=2, TString aDirNameModifier="");
  virtual ~PlotPartnersLamKch();

  virtual TCanvas* DrawPurity();
  virtual TCanvas* DrawKStarCfs();
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed);  // kSame, kRotSame, kMixed, kRotMixed};

  virtual TCanvas* DrawAvgSepCfs();
  virtual TCanvas* DrawAvgSepCfs(AnalysisType aAnalysisType, bool aDrawConj);

  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo);

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType);


private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamKch, 1)
#endif
};


#endif
