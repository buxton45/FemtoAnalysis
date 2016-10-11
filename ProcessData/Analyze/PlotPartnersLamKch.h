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
  PlotPartnersLamKch(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis=5, bool aIsTrainResults=false);
  PlotPartnersLamKch(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis=5, bool aIsTrainResults=false);
  virtual ~PlotPartnersLamKch();

  virtual TCanvas* DrawPurity();
  virtual TCanvas* DrawKStarCfs();
  virtual TCanvas* DrawKStarTrueVsRec(KStarTrueVsRecType aType=kMixed);  // kSame, kRotSame, kMixed, kRotMixed};
  virtual TCanvas* DrawAvgSepCfs();
  virtual TCanvas* ViewPart1MassFail(bool aDrawWideRangeToo);


private:


#ifdef __ROOT__
  ClassDef(PlotPartnersLamKch, 1)
#endif
};


#endif
