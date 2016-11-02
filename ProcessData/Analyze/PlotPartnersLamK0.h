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

  double GetPurity(AnalysisType aAnalysisType, ParticleType aV0Type);
  virtual TCanvas* DrawPurity(bool aSaveImage=false);
  virtual TCanvas* DrawKStarCfs(bool aSaveImage=false);
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed, bool aSaveImage=false);  // kSame, kRotSame, kMixed, kRotMixed};

  virtual TCanvas* DrawAvgSepCfs(bool aSaveImage=false);
  virtual TCanvas* DrawAvgSepCfs(AnalysisType aAnalysisType, bool aSaveImage=false);

  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage=false);

  TH1* GetMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aNormByNEv=false, int aMarkerColor=1, int aMarkerStyle=20, double aMarkerSize=0.5);
  TH1* GetMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aNormByNEv=false, int aMarkerColor=1, int aMarkerStyle=20, double aMarkerSize=0.5);
  TH1* GetMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aNormByNEv=false, int aMarkerColor=1, int aMarkerStyle=20, double aMarkerSize=0.5);

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false, bool aNormByNEv=false);
  virtual TCanvas* DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false, bool aNormByNEv=false);
  virtual TCanvas* DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage=false, bool aNormByNEv=false);

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage=false, TString aText1="No Cut", TString aText2="MisID Cut");
  virtual TCanvas* DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage=false, TString aText1="No Cut", TString aText2="MisID Cut");
  virtual TCanvas* DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage=false, TString aText1="No Cut", TString aText2="MisID Cut");

  virtual TCanvas* DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage=false);
  virtual TCanvas* DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage=false);
  virtual TCanvas* DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage=false);

  virtual TCanvas* DrawSumMassAssumingLambdaAndAntiLambdaHypotheses(AnalysisType aAnalysisType, bool aSaveImage=false);

private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamK0, 1)
#endif
};


#endif
