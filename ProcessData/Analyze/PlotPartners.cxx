///////////////////////////////////////////////////////////////////////////
// PlotPartners:                                                         //
///////////////////////////////////////////////////////////////////////////


#include "PlotPartners.h"

#ifdef __ROOT__
ClassImp(PlotPartners)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
PlotPartners::PlotPartners(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis, bool aIsTrainResults) :
  fContainsMC(false),
  fAnalysis1(0),
  fConjAnalysis1(0),
  fAnalysis2(0),
  fConjAnalysis2(0),

  fAnalysisMC1(0),
  fConjAnalysisMC1(0),
  fAnalysisMC2(0),
  fConjAnalysisMC2(0)

{

  switch(aAnalysisType) {
  case kLamK0:
  case kALamK0:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = NULL;
    fConjAnalysis2 = NULL;
    break;

  case kLamKchP:
  case kALamKchP:
  case kLamKchM:
  case kALamKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = new Analysis(aFileLocationBase,kLamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kALamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    break;

  case kXiKchP:
  case kAXiKchP:
  case kXiKchM:
  case kAXiKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kAXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = new Analysis(aFileLocationBase,kXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kAXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    break;

  default:
    cout << "Error in PlotPartners constructor, invalide aAnalysisType = " << aAnalysisType << " selected." << endl;
    assert(0);
  }

}

//________________________________________________________________________________________________________________
PlotPartners::PlotPartners(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis, bool aIsTrainResults) :
  fContainsMC(false),
  fAnalysis1(0),
  fConjAnalysis1(0),
  fAnalysis2(0),
  fConjAnalysis2(0),

  fAnalysisMC1(0),
  fConjAnalysisMC1(0),
  fAnalysisMC2(0),
  fConjAnalysisMC2(0)

{
  assert(aCentralityType == k0010);  //currently only have MC data for 0-10%

  switch(aAnalysisType) {
  case kLamK0:
  case kALamK0:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = NULL;
    fConjAnalysis2 = NULL;

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kLamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kALamK0,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysisMC2 = NULL;
    fConjAnalysisMC2 = NULL;

    if(fAnalysisMC1 && fConjAnalysisMC1) fContainsMC = true;
    break;

  case kLamKchP:
  case kALamKchP:
  case kLamKchM:
  case kALamKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = new Analysis(aFileLocationBase,kLamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kALamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kLamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kALamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysisMC2 = new Analysis(aFileLocationBaseMC,kLamKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysisMC2 = new Analysis(aFileLocationBaseMC,kALamKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);

    if(fAnalysisMC1 && fConjAnalysisMC1 && fAnalysisMC2 && fConjAnalysisMC2) fContainsMC = true;
    break;

  case kXiKchP:
  case kAXiKchP:
  case kXiKchM:
  case kAXiKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kAXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysis2 = new Analysis(aFileLocationBase,kXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kAXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kAXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fAnalysisMC2 = new Analysis(aFileLocationBaseMC,kXiKchM,aCentralityType,aNPartialAnalysis,aIsTrainResults);
    fConjAnalysisMC2 = new Analysis(aFileLocationBaseMC,kAXiKchP,aCentralityType,aNPartialAnalysis,aIsTrainResults);

    if(fAnalysisMC1 && fConjAnalysisMC1 && fAnalysisMC2 && fConjAnalysisMC2) fContainsMC = true;
    break;

  default:
    cout << "Error in PlotPartners constructor, invalide aAnalysisType = " << aAnalysisType << " selected." << endl;
    assert(0);
  }

}


//________________________________________________________________________________________________________________
PlotPartners::~PlotPartners()
{
}


//________________________________________________________________________________________________________________
void PlotPartners::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetRangeUser(aMin,aMax);

  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}

//________________________________________________________________________________________________________________
void PlotPartners::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
{
  aAxis->SetTitle(aTitle);
  aAxis->SetTitleSize(aTitleSize);
  aAxis->SetTitleOffset(aTitleOffset);
  if(aCenterTitle) {aAxis->CenterTitle();}

  aAxis->SetLabelSize(aLabelSize);
  aAxis->SetLabelOffset(aLabelOffset);

  aAxis->SetNdivisions(aNdivisions);

}

