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
PlotPartners::PlotPartners(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  fContainsMC(false),
  fSaveLocationBase(""),
  fDirNameModifier(aDirNameModifier),
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
    fAnalysis1 = new Analysis(aFileLocationBase,kLamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = NULL;
    fConjAnalysis2 = NULL;
    break;

  case kLamKchP:
  case kALamKchP:
  case kLamKchM:
  case kALamKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = new Analysis(aFileLocationBase,kLamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kALamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    break;

  case kXiKchP:
  case kAXiKchP:
  case kXiKchM:
  case kAXiKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kAXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = new Analysis(aFileLocationBase,kXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kAXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    break;

  default:
    cout << "Error in PlotPartners constructor, invalide aAnalysisType = " << aAnalysisType << " selected." << endl;
    assert(0);
  }

}

//________________________________________________________________________________________________________________
PlotPartners::PlotPartners(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  fContainsMC(false),
  fSaveLocationBase(""),
  fDirNameModifier(aDirNameModifier),
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
    fAnalysis1 = new Analysis(aFileLocationBase,kLamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = NULL;
    fConjAnalysis2 = NULL;

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kLamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kALamK0,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysisMC2 = NULL;
    fConjAnalysisMC2 = NULL;

    if(fAnalysisMC1 && fConjAnalysisMC1) fContainsMC = true;
    break;

  case kLamKchP:
  case kALamKchP:
  case kLamKchM:
  case kALamKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kLamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kALamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = new Analysis(aFileLocationBase,kLamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kALamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kLamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kALamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysisMC2 = new Analysis(aFileLocationBaseMC,kLamKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysisMC2 = new Analysis(aFileLocationBaseMC,kALamKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);

    if(fAnalysisMC1 && fConjAnalysisMC1 && fAnalysisMC2 && fConjAnalysisMC2) fContainsMC = true;
    break;

  case kXiKchP:
  case kAXiKchP:
  case kXiKchM:
  case kAXiKchM:
    fAnalysis1 = new Analysis(aFileLocationBase,kXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis1 = new Analysis(aFileLocationBase,kAXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysis2 = new Analysis(aFileLocationBase,kXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysis2 = new Analysis(aFileLocationBase,kAXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);

    fAnalysisMC1 = new Analysis(aFileLocationBaseMC,kXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysisMC1 = new Analysis(aFileLocationBaseMC,kAXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fAnalysisMC2 = new Analysis(aFileLocationBaseMC,kXiKchM,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);
    fConjAnalysisMC2 = new Analysis(aFileLocationBaseMC,kAXiKchP,aCentralityType,aRunType,aNPartialAnalysis,fDirNameModifier);

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

//________________________________________________________________________________________________________________
void PlotPartners::PrintAnalysisType(TPad* aPad, AnalysisType aAnType, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  aPad->cd();

  float tLeftMargin = aPad->GetLeftMargin();
  float tRightMargin = aPad->GetRightMargin();
  float tTopMargin = aPad->GetTopMargin();
  float tBottomMargin = aPad->GetBottomMargin();

  float tReNormalizedWidth = 1. - (tLeftMargin+tRightMargin);
  float tReNormalizedHeight = 1. - (tTopMargin+tBottomMargin);

  //------------------------------------

  double tNormalizedTextXmin = tLeftMargin + aTextXmin*tReNormalizedWidth;
  double tNormalizedTextYmin = tBottomMargin + aTextYmin*tReNormalizedHeight;

  double tNormalizedTextXmax = tNormalizedTextXmin + aTextWidth*tReNormalizedWidth;
  double tNormalizedTextYmax = tNormalizedTextYmin + aTextHeight*tReNormalizedHeight;

  //------------------------------------

  TString tText = TString(cAnalysisRootTags[aAnType]);

  TPaveText* tPaveText = new TPaveText(tNormalizedTextXmin,tNormalizedTextYmin,tNormalizedTextXmax,tNormalizedTextYmax,"NDC");
    tPaveText->SetFillColor(0);
    tPaveText->SetBorderSize(0);
    tPaveText->SetTextAlign(22);
    tPaveText->SetTextFont(aTextFont);
    tPaveText->SetTextSize(aTextSize);
    tPaveText->AddText(tText);

  tPaveText->Draw();
}

//________________________________________________________________________________________________________________
void PlotPartners::PrintText(TPad* aPad, TString aText, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  aPad->cd();

  float tLeftMargin = aPad->GetLeftMargin();
  float tRightMargin = aPad->GetRightMargin();
  float tTopMargin = aPad->GetTopMargin();
  float tBottomMargin = aPad->GetBottomMargin();

  float tReNormalizedWidth = 1. - (tLeftMargin+tRightMargin);
  float tReNormalizedHeight = 1. - (tTopMargin+tBottomMargin);

  //------------------------------------

  double tNormalizedTextXmin = tLeftMargin + aTextXmin*tReNormalizedWidth;
  double tNormalizedTextYmin = tBottomMargin + aTextYmin*tReNormalizedHeight;

  double tNormalizedTextXmax = tNormalizedTextXmin + aTextWidth*tReNormalizedWidth;
  double tNormalizedTextYmax = tNormalizedTextYmin + aTextHeight*tReNormalizedHeight;

  //------------------------------------
  TPaveText* tPaveText = new TPaveText(tNormalizedTextXmin,tNormalizedTextYmin,tNormalizedTextXmax,tNormalizedTextYmax,"NDC");
    tPaveText->SetFillColor(0);
    tPaveText->SetBorderSize(0);
    tPaveText->SetTextAlign(22);
    tPaveText->SetTextFont(aTextFont);
    tPaveText->SetTextSize(aTextSize);
    tPaveText->AddText(aText);

  tPaveText->Draw();
}


//________________________________________________________________________________________________________________
void PlotPartners::ExistsSaveLocationBase()
{
  if(!fSaveLocationBase.IsNull()) return;

  cout << "fSaveLocationBase is Null!!!!!" << endl;
  cout << "Create? (0=No 1=Yes)" << endl;
  int tResponse;
  cin >> tResponse;
  if(!tResponse) return;

  cout << "Enter base:" << endl;
  cin >> fSaveLocationBase;
  if(fSaveLocationBase[fSaveLocationBase.Length()] != '/') fSaveLocationBase += TString("/");
  return;

}

