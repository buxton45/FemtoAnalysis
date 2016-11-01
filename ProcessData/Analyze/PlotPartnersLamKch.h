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

  TH1* GetMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aNormByNEv=false, int aMarkerColor=1, int aMarkerStyle=20, double aMarkerSize=0.5);
  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false);
  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage=false, TString aText1="No Cut", TString aText2="MisID Cut");
  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, bool aSaveImage=false);

private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamKch, 1)
#endif
};


#endif
