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

  virtual TCanvas* DrawPurity(bool aSaveImage=false);
  virtual TCanvas* DrawKStarCfs(bool aSaveImage=false);
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed, bool aSaveImage=false);  // kSame, kRotSame, kMixed, kRotMixed};

  virtual TCanvas* DrawAvgSepCfs(bool aSaveImage=false);
  virtual TCanvas* DrawAvgSepCfs(AnalysisType aAnalysisType, bool aDrawConj, bool aSaveImage=false);

  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage=false);

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false);


private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamKch, 1)
#endif
};


#endif
