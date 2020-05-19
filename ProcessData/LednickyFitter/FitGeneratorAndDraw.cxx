/* FitGeneratorAndDraw.cxx */

#include "FitGeneratorAndDraw.h"

#ifdef __ROOT__
ClassImp(FitGeneratorAndDraw)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitGeneratorAndDraw::FitGeneratorAndDraw(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier, bool aUseStavCf) :
  FitGenerator(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aCentralityTypes, aRunType, aNPartialAnalysis, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier, aUseStavCf)
{

}


//________________________________________________________________________________________________________________
FitGeneratorAndDraw::FitGeneratorAndDraw(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier, bool aUseStavCf) :
  FitGenerator(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aCentralityType, aRunType, aNPartialAnalysis, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier, aUseStavCf)
{

}

//________________________________________________________________________________________________________________
FitGeneratorAndDraw::~FitGeneratorAndDraw()
{
  cout << "FitGeneratorAndDraw object is being deleted!!!!!" << endl;
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitGeneratorAndDraw::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitGeneratorAndDraw::CreateParamInitValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;

  tLambda = fFitParamsPerPad[tPosition][0].GetStartValue();
  tRadius = fFitParamsPerPad[tPosition][1].GetStartValue();
  tReF0 = fScattFitParams[0].GetStartValue();
  tImF0 = fScattFitParams[1].GetStartValue();
  tD0 = fScattFitParams[2].GetStartValue();

  assert(tReF0 == fFitParamsPerPad[tPosition][2].GetStartValue());
  assert(tImF0 == fFitParamsPerPad[tPosition][3].GetStartValue());
  assert(tD0 == fFitParamsPerPad[tPosition][4].GetStartValue());

  TPaveText *tText = aCanPart->SetupTPaveText("Initial Values",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.3f",tLambda));
  tText->AddText(TString::Format("R = %0.3f",tRadius));
  tText->AddText(TString::Format("Re[f0] = %0.3f",tReF0));
  tText->AddText(TString::Format("Im[f0] = %0.3f",tImF0));
  tText->AddText(TString::Format("d0 = %0.3f",tD0));

  tText->SetTextAlign(33);

//  tText->GetLine(0)->SetTextSize(0.08);
  tText->GetLine(0)->SetTextFont(43);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::CreateParamFinalValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
//TODO currently all fit values and errors are 0.
  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  tLambda = fFitParamsPerPad[tPosition][0].GetFitValue();
  tRadius = fFitParamsPerPad[tPosition][1].GetFitValue();
  tReF0 = fScattFitParams[0].GetFitValue();
  tImF0 = fScattFitParams[1].GetFitValue();
  tD0 = fScattFitParams[2].GetFitValue();

  tLambdaErr = fFitParamsPerPad[tPosition][0].GetFitValueError();
  tRadiusErr = fFitParamsPerPad[tPosition][1].GetFitValueError();
  tReF0Err = fScattFitParams[0].GetFitValueError();
  tImF0Err = fScattFitParams[1].GetFitValueError();
  tD0Err = fScattFitParams[2].GetFitValueError();

  double tChi2 = fLednickyFitter->GetChi2();
  int tNDF = fLednickyFitter->GetNDF();

  assert(tReF0 == fFitParamsPerPad[tPosition][2].GetFitValue());
  assert(tImF0 == fFitParamsPerPad[tPosition][3].GetFitValue());
  assert(tD0 == fFitParamsPerPad[tPosition][4].GetFitValue());

  TPaveText *tText = aCanPart->SetupTPaveText("Fit Values",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f",tLambda,tLambdaErr));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f",tRadius,tRadiusErr));
  tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f",tReF0,tReF0Err));
  tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f",tImF0,tImF0Err));
  tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f",tD0,tD0Err));

  tText->AddText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF));

  tText->SetTextAlign(33);

//  tText->GetLine(0)->SetTextSize(0.08);
  tText->GetLine(0)->SetTextFont(43);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::CreateParamFinalValuesText(AnalysisType aAnType, CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const td1dVec &aSysErrors, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize, bool aDrawAll)
{
  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  tLambda = aFit->GetParameter(0);
//  if(fIncludeResidualsType != kIncludeNoResiduals) tLambda /= cAnalysisLambdaFactors[aAnType];
  if(fIncludeResidualsType != kIncludeNoResiduals) tLambda /= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][aAnType];
  tRadius = aFit->GetParameter(1);
  tReF0 = aFit->GetParameter(2);
  tImF0 = aFit->GetParameter(3);
  tD0 = aFit->GetParameter(4);

  tLambdaErr = aFit->GetParError(0);
//  if(fIncludeResidualsType != kIncludeNoResiduals) tLambdaErr /= cAnalysisLambdaFactors[aAnType];
  if(fIncludeResidualsType != kIncludeNoResiduals) tLambdaErr /= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][aAnType];
  tRadiusErr = aFit->GetParError(1);
  tReF0Err = aFit->GetParError(2);
  tImF0Err = aFit->GetParError(3);
  tD0Err = aFit->GetParError(4);

  double tChi2 = fLednickyFitter->GetChi2();
  int tNDF = fLednickyFitter->GetNDF();

//  TPaveText *tText = aCanPart->SetupTPaveText("Fit Values",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  if(!aDrawAll) {aTextHeight /= 7; aTextHeight *= 3; aTextYmin += 1.25*aTextHeight; if(aNy==2) aTextYmin -= 0.25*aTextHeight;}
  TPaveText *tText = aCanPart->SetupTPaveText(/*"       stat.     sys."*/"",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f #pm %0.2f",tLambda,tLambdaErr,aSysErrors[0]));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f #pm %0.2f",tRadius,tRadiusErr,aSysErrors[1]));
  if(aDrawAll)
  {
    tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f #pm %0.2f",tReF0,tReF0Err,aSysErrors[2]));
    tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f #pm %0.2f",tImF0,tImF0Err,aSysErrors[3]));
    tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f #pm %0.2f",tD0,tD0Err,aSysErrors[4]));

//    tText->AddText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF));
  }

/*
  if(!aDrawAll) {aTextHeight /= 3; aTextYmin += 2*aTextHeight;}
  TPaveText *tText = aCanPart->SetupTPaveText("",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f (stat.) #pm %0.2f (sys.)",tLambda,tLambdaErr,aSysErrors[0]));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f (stat.) #pm %0.2f (sys.)",tRadius,tRadiusErr,aSysErrors[1]));
  if(aDrawAll)
  {
    tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f (stat.) #pm %0.2f (sys.)",tReF0,tReF0Err,aSysErrors[2]));
    tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f (stat.) #pm %0.2f (sys.)",tImF0,tImF0Err,aSysErrors[3]));
    tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f (stat.) #pm %0.2f (sys.)",tD0,tD0Err,aSysErrors[4]));

    tText->AddText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF));
  }
*/
  tText->SetTextAlign(33);

//  tText->GetLine(0)->SetTextSize(0.08);
//  tText->GetLine(0)->SetTextFont(43);
  aCanPart->AddPadPaveText(tText,aNx,aNy);

  //--------------------------------
  if(aNx==0 && aNy==0)
  {
    TPaveText *tText2 = aCanPart->SetupTPaveText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF),aNx,aNy,0.125,0.05,aTextWidth,0.10,aTextFont,0.9*aTextSize);
    aCanPart->AddPadPaveText(tText2,aNx,aNy);

    TPaveText *tText3 = aCanPart->SetupTPaveText("val. #pm stat. #pm sys.",aNx,aNy,0.255,0.48,aTextWidth,0.10,aTextFont,aTextSize);
    aCanPart->AddPadPaveText(tText3,aNx,aNy);
  }
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::CreateParamFinalValuesTextTwoColumns(CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const td1dVec &aSysErrors, double aText1Xmin, double aText1Ymin, double aText1Width, double aText1Height, bool aDrawText1, double aText2Xmin, double aText2Ymin, double aText2Width, double aText2Height, bool aDrawText2, double aTextFont, double aTextSize)
{
  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  int tPosition = aNx + aNy*tNx;

  double tLambda, tRadius, tReF0, tImF0, tD0;
  double tLambdaErr, tRadiusErr, tReF0Err, tImF0Err, tD0Err;

  tLambda = aFit->GetParameter(0);
  tRadius = aFit->GetParameter(1);
  tReF0 = aFit->GetParameter(2);
  tImF0 = aFit->GetParameter(3);
  tD0 = aFit->GetParameter(4);

  tLambdaErr = aFit->GetParError(0);
  tRadiusErr = aFit->GetParError(1);
  tReF0Err = aFit->GetParError(2);
  tImF0Err = aFit->GetParError(3);
  tD0Err = aFit->GetParError(4);

  double tChi2 = fLednickyFitter->GetChi2();
  int tNDF = fLednickyFitter->GetNDF();

  if(aDrawText1)
  {
    TPaveText *tText1 = aCanPart->SetupTPaveText("",aNx,aNy,aText1Xmin,aText1Ymin,aText1Width,aText1Height,aTextFont,aTextSize);
    tText1->AddText(TString::Format("#lambda = %0.2f #pm %0.2f #pm %0.2f",tLambda,tLambdaErr,aSysErrors[0]));
    tText1->AddText(TString::Format("R = %0.2f #pm %0.2f #pm %0.2f",tRadius,tRadiusErr,aSysErrors[1]));
    tText1->SetTextAlign(33);
    aCanPart->AddPadPaveText(tText1,aNx,aNy);
  }

  if(aDrawText2)
  {
    TPaveText *tText2 = aCanPart->SetupTPaveText("",aNx,aNy,aText2Xmin,aText2Ymin,aText2Width,aText2Height,aTextFont,aTextSize);
    tText2->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f #pm %0.2f",tReF0,tReF0Err,aSysErrors[2]));
    tText2->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f #pm %0.2f",tImF0,tImF0Err,aSysErrors[3]));
    tText2->AddText(TString::Format("d0 = %0.2f #pm %0.2f #pm %0.2f",tD0,tD0Err,aSysErrors[4]));
    tText2->AddText(TString::Format("#chi^{2}/NDF = %0.1f/%d",tChi2,tNDF));
    tText2->SetTextAlign(33);
    aCanPart->AddPadPaveText(tText2,aNx,aNy);
  }
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::AddTextCorrectionInfo(CanvasPartition *aCanPart, int aNx, int aNy, bool aMomResCorrect, bool aNonFlatCorrect, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  if(!aMomResCorrect && !aNonFlatCorrect) return;

  TPaveText *tText = aCanPart->SetupTPaveText("",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  if(aMomResCorrect) tText->AddText("Mom. Res. Correction");
  if(aNonFlatCorrect) tText->AddText("Non-flat Bgd Correction");
  tText->SetTextAlign(33);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::AddColoredLinesLabels(CanvasPartition *aCanPart, int aNx, int aNy, bool aZoomROP, double aScaleFactor)
{
  aCanPart->GetPad(aNx, aNy)->cd();

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.125);

  double tYText1 = 0.97;
  double tXText1 = 0.15;

  double tYText2 = 0.93;
  double tXText2 = 0.15;

  double tYText3 = 0.89;
  double tXText3 = 0.15;

  double tLineWidth = 0.025;
  double tSpaceBetween = 0.015;

  if(!aZoomROP)
  {
    tXText1 = 0.50;
    tXText2 = 0.50;
    tXText3 = 0.50;

    tLineWidth = 0.075;
    tSpaceBetween = 0.045;
  }

  TLine* tLine;

  tTex->DrawLatex(tXText1, tYText1, TString::Format("#scale[%f]{Primary #LambdaK}", aScaleFactor));
  tLine = new TLine(tXText1-tSpaceBetween-tLineWidth, tYText1, tXText1-tSpaceBetween, tYText1);
  tLine->SetLineColor(kBlack);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(3);
  tLine->Draw();
  
//  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Non-femto. bgd.}", aScaleFactor));
  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Scaled Bgd. Fit}", aScaleFactor));  
  tLine = new TLine(tXText2-tSpaceBetween-tLineWidth, tYText2, tXText2-tSpaceBetween, tYText2);
  tLine->SetLineColor(tColorNonFlatBgd);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(7);
  tLine->Draw();

  tTex->DrawLatex(tXText3, tYText3, TString::Format("#scale[%f]{Full fit}", aScaleFactor));
  tLine = new TLine(tXText3-tSpaceBetween-tLineWidth, tYText3, tXText3-tSpaceBetween, tYText3);
  tLine->SetLineColor(tColorCorrectFit);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(1);
  tLine->Draw();
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::AddColoredLinesLabelsAndData(CanvasPartition *aCanPart, int aNx, int aNy, bool aZoomROP, double aScaleFactor)
{
  aCanPart->GetPad(aNx, aNy)->cd();

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(42);
  tTex->SetTextSize(0.125);
  
  double tYText0 = 0.975;
  double tXText0 = 0.14;

  double tYText1 = 0.94;
  double tXText1 = 0.14;

  double tYText2 = 0.9125;
  double tXText2 = 0.14;

  double tYText3 = 0.885;
  double tXText3 = 0.14;
  


  double tLineWidth = 0.025;
  double tSpaceBetween = 0.015;

  if(!aZoomROP)
  {
    tXText0 = 0.50;
    tXText1 = 0.50;
    tXText2 = 0.50;
    tXText3 = 0.50;

    tLineWidth = 0.075;
    tSpaceBetween = 0.045;
  }

  TLine* tLine;
  
  tTex->DrawLatex(tXText0, tYText0, TString::Format("ALICE #scale[%f]{#LambdaK#scale[0.5]{ }#oplus#scale[0.5]{ }#bar{#LambdaK}}", aScaleFactor));  
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(1.0);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);
  tMarker->DrawMarker(tXText0-tSpaceBetween-(tLineWidth/2), tYText0);

  tTex->DrawLatex(tXText1, tYText1, TString::Format("#scale[%f]{Primary fit comp.}", aScaleFactor));
  tLine = new TLine(tXText1-tSpaceBetween-tLineWidth, tYText1, tXText1-tSpaceBetween, tYText1);
  tLine->SetLineColor(kBlack);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(3);
  tLine->Draw();
  
//  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Non-femto. bgd.}", aScaleFactor));
  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Scaled Bgd. Fit}", aScaleFactor));  
  tLine = new TLine(tXText2-tSpaceBetween-tLineWidth, tYText2, tXText2-tSpaceBetween, tYText2);
  tLine->SetLineColor(tColorNonFlatBgd);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(7);
  tLine->Draw();

  tTex->DrawLatex(tXText3, tYText3, TString::Format("#scale[%f]{Full fit}", aScaleFactor));
  tLine = new TLine(tXText3-tSpaceBetween-tLineWidth, tYText3, tXText3-tSpaceBetween, tYText3);
  tLine->SetLineColor(tColorCorrectFit);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(1);
  tLine->Draw();
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::AddColoredLinesLabelsAndDatav2(CanvasPartition *aCanPart, int aNx, int aNy, bool aZoomROP, double aScaleFactor)
{
  aCanPart->GetPad(aNx, aNy)->cd();

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;

  TLatex* tTex = new TLatex();
  tTex->SetTextAlign(12);
  tTex->SetLineWidth(2);
  tTex->SetTextFont(43);
  tTex->SetTextSize(50);
  
  double tYText0 = 0.975;
  double tXText0 = 0.115;

  double tYText1 = 0.93;
  double tXText1 = 0.115;

  double tYText2 = 0.8925;
  double tXText2 = 0.115;

  double tYText3 = 0.855;
  double tXText3 = 0.115;
  


  double tLineWidth = 0.025;
  double tSpaceBetween = 0.015;

  if(!aZoomROP)
  {
    tXText0 = 0.50;
    tXText1 = 0.50;
    tXText2 = 0.50;
    tXText3 = 0.50;

    tLineWidth = 0.075;
    tSpaceBetween = 0.045;
  }

  TLine* tLine;
  
  tTex->DrawLatex(tXText0, tYText0, TString::Format("ALICE #scale[%f]{#LambdaK#scale[0.5]{ }#oplus#scale[0.5]{ }#bar{#LambdaK}}", aScaleFactor));  
  TMarker *tMarker = new TMarker();
  tMarker->SetMarkerSize(1.5);
  tMarker->SetMarkerStyle(20);
  tMarker->SetMarkerColor(kBlack);
  tMarker->DrawMarker(tXText0-tSpaceBetween-(tLineWidth/2), tYText0);

  tTex->DrawLatex(tXText1, tYText1, TString::Format("#scale[%f]{Primary fit comp.}", aScaleFactor));
  tLine = new TLine(tXText1-tSpaceBetween-tLineWidth, tYText1, tXText1-tSpaceBetween, tYText1);
  tLine->SetLineColor(kBlack);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(3);
  tLine->Draw();
  
//  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Non-femto. bgd.}", aScaleFactor));
  tTex->DrawLatex(tXText2, tYText2, TString::Format("#scale[%f]{Scaled Bgd. Fit}", aScaleFactor));
  tLine = new TLine(tXText2-tSpaceBetween-tLineWidth, tYText2, tXText2-tSpaceBetween, tYText2);
  tLine->SetLineColor(tColorNonFlatBgd);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(7);
  tLine->Draw();

  tTex->DrawLatex(tXText3, tYText3, TString::Format("#scale[%f]{Full fit}", aScaleFactor));
  tLine = new TLine(tXText3-tSpaceBetween-tLineWidth, tYText3, tXText3-tSpaceBetween, tYText3);
  tLine->SetLineColor(tColorCorrectFit);
  tLine->SetLineWidth(2);
  tLine->SetLineStyle(1);
  tLine->Draw();
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCfClone();
    SetupAxis(tCfToDraw->GetXaxis(),"k* (GeV/c)");
    SetupAxis(tCfToDraw->GetYaxis(),"C(k*)");

  tCfToDraw->GetXaxis()->SetRangeUser(aXmin,aXmax);
  tCfToDraw->GetYaxis()->SetRangeUser(aYmin,aYmax);

  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);

  tCfToDraw->Draw(aOption);

  TLine *line = new TLine(aXmin,1,aXmax,1);
  line->SetLineColor(14);
  line->Draw();
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  gStyle->SetOptFit();

  gStyle->SetStatH(0.15);
  gStyle->SetStatW(0.30);

  gStyle->SetStatX(0.85);
  gStyle->SetStatY(0.60);


  TH1* tCfToDraw = fSharedAn->GetKStarCfHeavy(aPairAnNumber)->GetHeavyCfClone();
    SetupAxis(tCfToDraw->GetXaxis(),"k* (GeV/c)");
    SetupAxis(tCfToDraw->GetYaxis(),"C(k*)");

  tCfToDraw->GetXaxis()->SetRangeUser(aXmin,aXmax);
  tCfToDraw->GetYaxis()->SetRangeUser(aYmin,aYmax);

  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);

  tCfToDraw->Draw(aOption);

  TF1* tFit = fSharedAn->GetFitPairAnalysis(aPairAnNumber)->GetPrimaryFit();
  tFit->SetLineColor(1);
  tFit->Draw("same");

  TLine *line = new TLine(aXmin,1,aXmax,1);
  line->SetLineColor(14);
  line->Draw();
}

//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawKStarCfs(bool aSaveImage, bool aDrawSysErrors)
{
  TString tCanvasName = TString("canKStarCfs");
  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);


  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  int tMarkerColor = 1;
  double tMarkerSize = 0.5;

  if(fPairType==kLamK0 || fPairType==kALamK0) tMarkerColor = 1;
  else if(fPairType==kLamKchP || fPairType==kALamKchM) tMarkerColor = 2;
  else if(fPairType==kLamKchM || fPairType==kALamKchP) tMarkerColor = 4;
  else tMarkerColor=1;

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
      TH1* tCfwSysErrs;
      if(aDrawSysErrors)
      {
        tCfwSysErrs = (TH1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCfwSysErrors();
          tCfwSysErrs->SetFillStyle(0);  //for box error bars to draw correctly
      }

//TODO 
//If the binnings are unequal, I must regenerate the plots with Analyze/Systematics/BuildErrorBars.C
//This is because a Cf should not simply be rebinned, but rather the Num and Den should be rebinned, and the Cf rebuilt
//Ths incorrect method would be Cf->Rebin(aRebin); Cf->Scale(1./aRebin);
/*
      double tDesiredBinWidth, tBinWidth;
      tDesiredBinWidth = ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1);
      tBinWidth = tCfwSysErrs->GetBinWidth(1);

      if( (tDesiredBinWidth != tBinWidth) && (fmod(tDesiredBinWidth,tBinWidth) == 0) )
      {
        int tScale = tDesiredBinWidth/tBinWidth;
        tCfwSysErrs->Rebin(tScale);
        tCfwSysErrs->Scale(1./tScale);
      }
      else if(tDesiredBinWidth != tBinWidth)
      {
        cout << "ERROR: FitGeneratorAndDraw::DrawKStarCfs: Histogram containing systematic error bars does not have the correct bin size and" << endl;
        cout << "DNE an appropriate scale to resolve the issue" << endl;
        assert(0);
      }
*/
//      assert(tCfwSysErrs->GetBinWidth(1) == tDesiredBinWidth);
      if(aDrawSysErrors) assert(tCfwSysErrs->GetBinWidth(1) == ((TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone())->GetBinWidth(1));
      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",tMarkerStyle,tMarkerColor,tMarkerSize);
      if(aDrawSysErrors) tCanPart->AddGraph(i,j,tCfwSysErrs,"",tMarkerStyle,tMarkerColor,tMarkerSize,"e2psame");

      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
TH1D* FitGeneratorAndDraw::Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle)
{
  assert(aCfVec.size() == aKStarBinCenters.size());

  double tBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];
  int tNbins = aKStarBinCenters.size();
  double tKStarMin = aKStarBinCenters[0]-tBinWidth/2.0;
  tKStarMin=0.;
  double tKStarMax = aKStarBinCenters[tNbins-1] + tBinWidth/2.0;

  TH1D* tReturnHist = new TH1D(aTitle, aTitle, tNbins, tKStarMin, tKStarMax);
  for(int i=0; i<tNbins; i++) {tReturnHist->SetBinContent(i+1,aCfVec[i]); tReturnHist->SetBinError(i+1,0.);}

  return tReturnHist;
}


//________________________________________________________________________________________________________________
td1dVec FitGeneratorAndDraw::GetSystErrs(IncludeResidualsType aIncResType, AnalysisType aAnType, CentralityType aCentType)
{
  td1dVec tReturnVec = {cFitParamValues[aIncResType][aAnType][aCentType][kLambda][kSystErr], 
                        cFitParamValues[aIncResType][aAnType][aCentType][kRadius][kSystErr], 
                        cFitParamValues[aIncResType][aAnType][aCentType][kRef0][kSystErr], 
                        cFitParamValues[aIncResType][aAnType][aCentType][kImf0][kSystErr], 
                        cFitParamValues[aIncResType][aAnType][aCentType][kd0][kSystErr]};

  return tReturnVec;
}

//________________________________________________________________________________________________________________
td1dVec FitGeneratorAndDraw::GetSystErrs(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  td1dVec tReturnVec(5);
  ParameterType tParamType;
  FitParameter* tFitParam;
  for(unsigned int iParam=0; iParam<5; iParam++)
  {
    tParamType = static_cast<ParameterType>(iParam);
    tFitParam = FitValuesWriterwSysErrs::GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, tParamType);
    tReturnVec[iParam] = tFitParam->GetFitValueSysError();
  }
  return tReturnVec;
}



//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::BuildKStarCfswFitsPanel_PartAn(CanvasPartition* aCanPart, int aAnalysisNumber, BFieldType aBFieldType, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawDataOnTop)
{
  int tColor, tColorTransparent;
  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetAnalysisType();
  CentralityType tCentType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCentralityType();
  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;

  int tOffset = 0;
  if(!fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->AreTrainResults()) tOffset = 2;

  int iPartAn = static_cast<int>(aBFieldType)-tOffset;
  FitPairAnalysis* tPairAn = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  FitPartialAnalysis* tPartAn = tPairAn->GetFitPartialAnalysis(iPartAn);

  //---------------------------------------------------------------------------------------------------------
  TH1* tCfData = (TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetCfLite(iPartAn)->Cf()->Clone();

  TF1* tNonFlatBgd;
  if(fApplyNonFlatBackgroundCorrection)
  {
    if(!fSharedAn->UsingNewBgdTreatment()) tNonFlatBgd = (TF1*)tPartAn->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true);
    else tNonFlatBgd = (TF1*)tPartAn->GetNewNonFlatBackground(aNonFlatBgdFitType);
  }

  tPartAn->CreateFitFunction(true, fIncludeResidualsType, fResPrimMaxDecayType, fLednickyFitter->GetChi2(), fLednickyFitter->GetNDF(), 0.0, 1.0);
  TF1* tPrimaryFit = tPartAn->GetPrimaryFit();

  td1dVec tCorrectedFitVec = tPartAn->GetCorrectedFitVec();
  td1dVec tKStarBinCenters = fLednickyFitter->GetKStarBinCenters();
  TString tCorrectedFitHistName = TString::Format("tCorrectedFitHist_%s%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType], cBFieldTags[aBFieldType]);
  TH1D* tCorrectedFitHist = Convert1dVecToHist(tCorrectedFitVec, tKStarBinCenters, tCorrectedFitHistName);
    tCorrectedFitHist->SetLineWidth(2);

  //---------------------------------------------------------------------------------------------------------

  aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,0.5,"ex0");  //ex0 suppresses the error along x
  if(fApplyNonFlatBackgroundCorrection) aCanPart->AddGraph(tColumn,tRow,tNonFlatBgd,"",20,tColorNonFlatBgd);
  aCanPart->AddGraph(tColumn,tRow,tPrimaryFit,"");
  aCanPart->AddGraph(tColumn,tRow,tCorrectedFitHist,"",20,tColorCorrectFit,0.5,"lsame");
  if(aDrawDataOnTop) aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,0.5,"ex0same");  //draw again so data on top
}

//________________________________________________________________________________________________________________
CanvasPartition* FitGeneratorAndDraw::BuildKStarCfswFitsCanvasPartition_PartAn(BFieldType aBFieldType, TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aZoomROP)
{
  TString tCanvasName = aCanvasBaseName;
  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

  tCanvasName += TString(cBFieldTags[aBFieldType]);

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(1400,1500);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      CentralityType tCentType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      BuildKStarCfswFitsPanel_PartAn(tCanPart, tAnalysisNumber, aBFieldType, i, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);

      TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
      TString tTextBField = TString(cBFieldTags[aBFieldType]);

      TString tCombinedText = TString::Format("%s  %s%s", tTextAnType.Data(), tTextCentrality.Data(), tTextBField.Data());
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.65,0.825,0.15,0.10,43,13);
      tCanPart->AddPadPaveText(tCombined,i,j);

      if(i==0 && j==0)
      {
//        TString tTextAlicePrelim = TString("ALICE Preliminary");
        TString tTextAlicePrelim = TString("ALICE");
        TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.05,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tAlicePrelim,i,j);
      }

      if(i==1 && j==0)
      {
        TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.125,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }

      td1dVec tSysErrors;
      if(fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()) tSysErrors = GetSystErrs(fIncludeResidualsType, tAnType, tCentType);
      else tSysErrors = GetSystErrs(fMasterFileLocation, fSystematicsFileLocation, fSaveNameModifier, tAnType, tCentType);

      bool bDrawAll = false;
      if(i==0 && j==0) bDrawAll = true;
      CreateParamFinalValuesText(tAnType, tCanPart,i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.73,0.09,0.25,0.53,43,12.0,bDrawAll);

    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart;
}

//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP)
{
  TString tCanvasBaseName = "canKStarCfwFits";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition_PartAn(aBFieldType, tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::BuildKStarCfswFitsPanel(CanvasPartition* aCanPart, int aAnalysisNumber, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aDrawDataOnTop)
{
  int tColor, tColorTransparent;
  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetAnalysisType();
  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;




  //---------------------------------------------------------------------------------------------------------
  TH1* tCfData = (TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone();

  TF1* tNonFlatBgd;
  if(fApplyNonFlatBackgroundCorrection)
  {
    if(!fSharedAn->UsingNewBgdTreatment()) tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);
    else tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNewNonFlatBackground(aNonFlatBgdFitType, true);  //TODO second argument should be set automatically

    tNonFlatBgd->SetLineStyle(7);
  }

  TF1* tPrimaryFit = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetPrimaryFit();
  tPrimaryFit->SetLineStyle(3);

//TODO currently GetCorrectedFitHistv2 is the method which can also include residuals in the fit
//  TH1* tCorrectedFitHisto = (TH1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCorrectedFitHisto(aMomResCorrectFit,aNonFlatBgdCorrectFit,false,aNonFlatBgdFitType);
  TH1F* tCorrectedFitHisto = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCorrectedFitHistv2();
    tCorrectedFitHisto->SetLineWidth(2);

  //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
  TH1* tCfwSysErrs;
  if(aDrawSysErrors)
  {
    tCfwSysErrs = (TH1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCfwSysErrors();
      //tCfwSysErrs->SetFillStyle(0);  //for box error bars to draw correctly
      tCfwSysErrs->SetFillColor(tColorTransparent);
      tCfwSysErrs->SetFillStyle(1000);
      tCfwSysErrs->SetLineColor(0);
      tCfwSysErrs->SetLineWidth(0);
  }

  //---------------------------------------------------------------------------------------------------------

//TODO 
//If the binnings are unequal, I must regenerate the plots with Analyze/Systematics/BuildErrorBars.C
//This is because a Cf should not simply be rebinned, but rather the Num and Den should be rebinned, and the Cf rebuilt
//Ths incorrect method would be Cf->Rebin(aRebin); Cf->Scale(1./aRebin);
/*
      double tDesiredBinWidth, tBinWidth;
      tDesiredBinWidth = tCfData->GetBinWidth(1);
      tBinWidth = tCfwSysErrs->GetBinWidth(1);

      if( (tDesiredBinWidth != tBinWidth) && (fmod(tDesiredBinWidth,tBinWidth) == 0) )
      {
        int tScale = tDesiredBinWidth/tBinWidth;
        tCfwSysErrs->Rebin(tScale);
        tCfwSysErrs->Scale(1./tScale);
      }
      else if(tDesiredBinWidth != tBinWidth)
      {
        cout << "ERROR: FitGeneratorAndDraw::DrawKStarCfswFits: Histogram containing systematic error bars does not have the correct bin size and" << endl;
        cout << "DNE an appropriate scale to resolve the issue" << endl;
        assert(0);
      }
*/
//      assert(tCfwSysErrs->GetBinWidth(1) == tDesiredBinWidth);
  if(aDrawSysErrors) assert(tCfwSysErrs->GetBinWidth(1) == tCfData->GetBinWidth(1));
  //---------------------------------------------------------------------------------------------------------

  aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,0.5,"ex0");  //ex0 suppresses the error along x
  if(fApplyNonFlatBackgroundCorrection) aCanPart->AddGraph(tColumn,tRow,tNonFlatBgd,"",20,tColorNonFlatBgd);
  aCanPart->AddGraph(tColumn,tRow,tPrimaryFit,"");
  aCanPart->AddGraph(tColumn,tRow,tCorrectedFitHisto,"",20,tColorCorrectFit,0.5,"lsame");
  if(aDrawSysErrors) aCanPart->AddGraph(tColumn,tRow,tCfwSysErrs,"",20,tColorTransparent,0.5,"e2psame");
  if(aDrawDataOnTop) aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,0.5,"ex0same");  //draw again so data on top
}


//________________________________________________________________________________________________________________
CanvasPartition* FitGeneratorAndDraw::BuildKStarCfswFitsCanvasPartition(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  TString tCanvasName = aCanvasBaseName;
  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");
  if(aLabelLines) tCanvasName += TString("_LabelLines");

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(1400,1500);

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;
      AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      CentralityType tCentType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType();

      BuildKStarCfswFitsPanel(tCanPart, tAnalysisNumber, i, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

      TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
      TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);

      TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.70,0.825,0.15,0.10,43,20);;
      tCanPart->AddPadPaveText(tCombined,i,j);

      if(i==(tNx-1) && j==(tNy-1))
      {
        TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.20,0.125,0.725,0.15,43,17);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }

      td1dVec tSysErrors;
      if(fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()) 
      {
        if(aDrawSysErrors)
        {
          cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
          int tResponse;
          cin >> tResponse;
          assert(tResponse);
        }
        tSysErrors = GetSystErrs(fIncludeResidualsType, tAnType, tCentType);
      }
      else tSysErrors = GetSystErrs(fMasterFileLocation, fSystematicsFileLocation, fSaveNameModifier, tAnType, tCentType);

      bool bDrawAll = false;
      if(i==0 && j==0) bDrawAll = true;
      if(!aSuppressFitInfoOutput) CreateParamFinalValuesText(tAnType, tCanPart,i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.73,0.09,0.25,0.53,43,12.0,bDrawAll);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  if(aLabelLines)
  {
    assert(aSuppressFitInfoOutput);  //Not enough room for everyone!
    AddColoredLinesLabels(tCanPart, 0, 0, aZoomROP);
  }

  return tCanPart;
}

//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  TString tCanvasBaseName = "canKStarCfwFits";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition(tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
double FitGeneratorAndDraw::GetWeightedAnalysisNorm(FitPairAnalysis* aPairAn)
{
  double tOverallNum=0., tOverallDen=0.;
  double tNumScale=0., tNorm=0.;
  for(int iPartAn=0; iPartAn<aPairAn->GetNFitPartialAnalysis(); iPartAn++)
  {
    tNumScale = aPairAn->GetFitPartialAnalysis(iPartAn)->GetKStarCfLite()->GetNumScale();
    tNorm = aPairAn->GetFitPartialAnalysis(iPartAn)->GetFitNormParameter()->GetFitValue();

    tOverallNum += tNumScale*tNorm;
    tOverallDen += tNumScale;
  }
  double tOverallScale = tOverallNum/tOverallDen;
  return tOverallScale;
}


//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawResiduals(int aAnalysisNumber, CentralityType aCentralityType, TString aCanvasName, bool aSaveImage)
{
  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnType = tFitPairAnalysis->GetAnalysisType();

  double tOverallLambdaPrimary = tFitPairAnalysis->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tFitPairAnalysis->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
  double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);
  //-----------------------------------------------------------------------------------

  TString tCanvasName = aCanvasName;
  tCanvasName += TString(cAnalysisBaseTags[tAnType]);
  tCanvasName += TString(cCentralityTags[aCentralityType]);
  TCanvas *tCan = new TCanvas(tCanvasName,tCanvasName);

  int tNx = 2;
  int tNy = 5;

  tCan->Divide(tNx,tNy, 0.001, 0.001);

  double tXLow = 0.0;
  double tXHigh = 0.3;

  vector<int> tResBaseColors{7,8,9,30,33,40,41,  44,46,47,49};
  vector<int> tResMarkerStyles{20,21,22,33,34,29,23,  20,21,22,33};
  vector<int> tTransformedResMarkerStyles{24,25,26,27,28,30,32,  24,25,26,27};

  //-----------------------------------------------------------------------------------
  int tNResiduals;
  if(fIncludeResidualsType==kInclude10Residuals) tNResiduals = 10;  //really are 11, but will skip Omega
  else if(fIncludeResidualsType==kInclude3Residuals) tNResiduals = 3;
  else assert(0);

  int tNNeutral=0, tNCharged=0;
  TPad *tPadA, *tPadB;
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    if( (tTempResidualType != kResOmegaK0) && (tTempResidualType != kResAOmegaK0) )
    {
      TString tTempName = TString(cAnalysisBaseTags[tTempResidualType]);
      TString tTempName1 = TString("Residual_") + tTempName;
      TString tTempName2 = TString("TransformedResidual_") + tTempName;
      TH1D* tTempHist1 = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName1, tParamsOverall.data(), tWeightedNorm);
      TH1D* tTempHist2 = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName2, tParamsOverall.data(), tWeightedNorm);

      tNNeutral++;
      tCan->cd(tNNeutral);

      tPadA = new TPad(TString::Format("tPad_%iA",tNNeutral), TString::Format("tPad_%iA",tNNeutral), 0.0, 0.0, 0.5, 1.0);
      tPadA->Draw();
      tPadB = new TPad(TString::Format("tPad_%iB",tNNeutral), TString::Format("tPad_%iB",tNNeutral), 0.5, 0.0, 1.0, 1.0);
      tPadB->Draw();

      tTempHist1->SetMarkerStyle(tResMarkerStyles[iRes]);
      tTempHist1->SetMarkerColor(tResBaseColors[iRes]);
      tTempHist1->SetLineColor(tResBaseColors[iRes]);
      tTempHist1->SetMarkerSize(0.5);

      tTempHist2->SetMarkerStyle(tTransformedResMarkerStyles[iRes]);
      tTempHist2->SetMarkerColor(tResBaseColors[iRes]);
      tTempHist2->SetLineColor(tResBaseColors[iRes]);
      tTempHist2->SetMarkerSize(0.5);

      tTempHist1->GetXaxis()->SetRangeUser(tXLow,tXHigh);
      tTempHist2->GetXaxis()->SetRangeUser(tXLow,tXHigh);

      tPadA->cd();
      tTempHist1->Draw("ex0");
      tTempHist2->Draw("ex0same");

      TLegend* tLeg = new TLegend(0.60, 0.20, 0.85, 0.70);
      tLeg->SetHeader(TString::Format("%s Residuals", cAnalysisRootTags[tTempResidualType]));
      tLeg->AddEntry((TObject*)0, TString::Format("#lambda = %0.3f \t #lambda_{Tot} = %0.3f", tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetLambdaFactor(), tOverallLambdaPrimary), "");
      tLeg->AddEntry(tTempHist1, "Residual", "p");
      tLeg->AddEntry(tTempHist2, "Transformed", "p");
      tLeg->Draw();

      tPadB->cd();
      tTempHist2->Draw("ex0");
    }
  }
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    if( (tTempResidualType != kResOmegaKchP) && (tTempResidualType != kResAOmegaKchM) && (tTempResidualType != kResOmegaKchM) && (tTempResidualType != kResAOmegaKchP) )
    {
      TString tTempName = TString(cAnalysisBaseTags[tTempResidualType]);
      TString tTempName1 = TString("Residual_") + tTempName;
      TString tTempName2 = TString("TransformedResidual_") + tTempName;
      TH1D* tTempHist1 = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName1, tParamsOverall.data(), tWeightedNorm);
      TH1D* tTempHist2 = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName2, tParamsOverall.data(), tWeightedNorm);

      tNCharged++;
      tCan->cd(tNNeutral+tNCharged);

      tPadA = new TPad(TString::Format("tPad_%iA",tNNeutral+tNCharged), TString::Format("tPad_%iA",tNNeutral+tNCharged), 0.0, 0.0, 0.5, 1.0);
      tPadA->Draw();
      tPadB = new TPad(TString::Format("tPad_%iB",tNNeutral+tNCharged), TString::Format("tPad_%iB",tNNeutral+tNCharged), 0.5, 0.0, 1.0, 1.0);
      tPadB->Draw();

      tTempHist1->SetMarkerStyle(tResMarkerStyles[iRes+tNNeutral]);
      tTempHist1->SetMarkerColor(tResBaseColors[iRes+tNNeutral]);
      tTempHist1->SetLineColor(tResBaseColors[iRes+tNNeutral]);
      tTempHist1->SetMarkerSize(1.0);

      tTempHist2->SetMarkerStyle(tTransformedResMarkerStyles[iRes+tNNeutral]);
      tTempHist2->SetMarkerColor(tResBaseColors[iRes+tNNeutral]);
      tTempHist2->SetLineColor(tResBaseColors[iRes+tNNeutral]);
      tTempHist2->SetMarkerSize(1.0);

      tTempHist1->GetXaxis()->SetRangeUser(tXLow,tXHigh);
      tTempHist2->GetXaxis()->SetRangeUser(tXLow,tXHigh);

      tPadA->cd();
      tTempHist1->Draw("ex0");
      tTempHist2->Draw("ex0same");

      TLegend* tLeg = new TLegend(0.60, 0.20, 0.85, 0.70);
      tLeg->SetHeader(TString::Format("%s Residuals", cAnalysisRootTags[tTempResidualType]));
      tLeg->AddEntry((TObject*)0, TString::Format("#lambda = %0.3f \t #lambda_{Tot} = %0.3f", tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetLambdaFactor(), tOverallLambdaPrimary), "");
      tLeg->AddEntry(tTempHist1, "Residual", "p");
      tLeg->AddEntry(tTempHist2, "Transformed", "p");
      tLeg->Draw();

      tPadB->cd();
      tTempHist2->Draw("ex0");
    }
  }

  assert((tNNeutral+tNCharged) == tNResiduals);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    TString tSaveLocationDir = TString::Format("%sResiduals%s/%s/%s/", fSaveLocationBase.Data(), cIncludeResidualsTypeTags[fIncludeResidualsType], cAnalysisBaseTags[tAnType], cCentralityTags[aCentralityType]);
    gSystem->mkdir(tSaveLocationDir, true);
    tCan->SaveAs(tSaveLocationDir+tCan->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }


  return tCan;
}

//________________________________________________________________________________________________________________
TObjArray* FitGeneratorAndDraw::DrawAllResiduals(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++)
  {
    CentralityType tCentType = fSharedAn->GetFitPairAnalysis(i)->GetCentralityType();
    TCanvas* tCan = DrawResiduals(i,tCentType, "Residuals", aSaveImage);
    tReturnArray->Add(tCan);
  }
  return tReturnArray;
}

//________________________________________________________________________________________________________________
template <typename T>
TCanvas* FitGeneratorAndDraw::GetResidualsWithTransformMatrices(int aAnalysisNumber, T& aResidual, td1dVec &aParamsOverall, int aOffset)
{
  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnType = tFitPairAnalysis->GetAnalysisType();
  CentralityType tCentType = tFitPairAnalysis->GetCentralityType();
  double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);
  //-----------------------------------------------------------------------------------
  double tXLow = 0.0;
  double tXHigh = 0.3;

  vector<int> tResBaseColors{7,8,9,30,33,40,41,  44,46,47,49};
  vector<int> tResMarkerStyles{20,21,22,33,34,29,23,  20,21,22,33};
  vector<int> tTransformedResMarkerStyles{24,25,26,27,28,30,32,  24,25,26,27};

  //-----------------------------------------------------------------------------------
  TCanvas* tReturnCan;
  TPad *tPadA, *tPadB, *tPadC, *tPadD;
  TString tCanvasBaseName = TString::Format("Residuals_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);
  TString tTempName, tTempName1, tTempName2;
  TString tMotherName1, tDaughterName1;
  TString tMotherName2, tDaughterName2;
  TString tBoxText;
  TPaveText* tText;
    double tTextXmin = 0.20;
    double tTextXmax = 0.40;
    double tTextYmin = 0.75;
    double tTextYmax = 0.85;


  AnalysisType tTempResidualType = aResidual.GetResidualType();

  tTempName = TString(cAnalysisBaseTags[tTempResidualType]);
  tTempName1 = TString("Residual_") + tTempName;
  tTempName2 = TString("TransformedResidual_") + tTempName;

  tMotherName1 = GetPDGRootName(aResidual.GetMotherType1());
  tMotherName2 = GetPDGRootName(aResidual.GetMotherType2());
  tDaughterName1 = GetPDGRootName(aResidual.GetDaughterType1());
  tDaughterName2 = GetPDGRootName(aResidual.GetDaughterType2());

  tBoxText = TString::Format("%s%s To %s%s", tMotherName1.Data(), tMotherName2.Data(), tDaughterName1.Data(), tDaughterName2.Data());

  TH1D* tTempHist1 = aResidual.GetResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName1, aParamsOverall.data(), tWeightedNorm);
  TH1D* tTempHist2 = aResidual.GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName2, aParamsOverall.data(), tWeightedNorm);

  TH2D* tTempTransformMatrix = aResidual.GetTransformMatrix();
  tTempTransformMatrix->GetXaxis()->SetTitle(TString("#it{k}*_{") + tDaughterName1 + tDaughterName2 + TString("}(GeV/#it{c})"));
    tTempTransformMatrix->GetXaxis()->SetTitleSize(0.045);
    tTempTransformMatrix->GetXaxis()->SetTitleOffset(1.15);

  tTempTransformMatrix->GetYaxis()->SetTitle(TString("#it{k}*_{") + tMotherName1 + tMotherName2 + TString("}(GeV/#it{c})"));
    tTempTransformMatrix->GetYaxis()->SetTitleSize(0.045);
    tTempTransformMatrix->GetYaxis()->SetTitleOffset(1.1);

  tTempTransformMatrix->GetZaxis()->SetLabelSize(0.045);
  tTempTransformMatrix->GetZaxis()->SetLabelOffset(0.0045);

  tReturnCan = new TCanvas(TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]),
                           TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]));
  tReturnCan->cd();

  tPadA = new TPad(TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.5, 0.5, 1.0);
  tPadA->SetLeftMargin(0.125);
  tPadA->SetBottomMargin(0.125);
  tPadA->Draw();

  tPadB = new TPad(TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.0, 0.5, 0.5);
  tPadB->SetLeftMargin(0.125);
  tPadB->SetBottomMargin(0.125);
  tPadB->Draw();

  tPadC = new TPad(TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.5, 0.5, 1.0, 1.0);
  tPadC->SetRightMargin(0.15);
  tPadC->SetLeftMargin(0.125);
  tPadC->SetBottomMargin(0.125);
  tPadC->Draw();

  tPadD = new TPad(TString::Format("tPadD_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadD_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.5, 0.0, 1.0, 0.5);
  tPadD->SetRightMargin(0.15);
  tPadD->SetLeftMargin(0.125);
  tPadD->SetBottomMargin(0.125);
  tPadD->Draw();

  tTempHist1->SetMarkerStyle(tResMarkerStyles[aOffset]);
  tTempHist1->SetMarkerColor(tResBaseColors[aOffset]);
  tTempHist1->SetLineColor(tResBaseColors[aOffset]);
  tTempHist1->SetMarkerSize(0.5);

  tTempHist2->SetMarkerStyle(tTransformedResMarkerStyles[aOffset]);
  tTempHist2->SetMarkerColor(tResBaseColors[aOffset]);
  tTempHist2->SetLineColor(tResBaseColors[aOffset]);
  tTempHist2->SetMarkerSize(0.5);

  tTempHist1->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  tTempHist2->GetXaxis()->SetRangeUser(tXLow,tXHigh);

  tTempHist1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tTempHist1->GetXaxis()->SetTitleSize(0.045);
    tTempHist1->GetXaxis()->SetTitleOffset(1.1);
  tTempHist2->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tTempHist2->GetXaxis()->SetTitleSize(0.045);
    tTempHist2->GetXaxis()->SetTitleOffset(1.1);

  tTempHist1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist1->GetYaxis()->SetTitleSize(0.05);
    tTempHist1->GetYaxis()->SetTitleOffset(1.1);
  tTempHist2->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist2->GetYaxis()->SetTitleSize(0.05);
    tTempHist2->GetYaxis()->SetTitleOffset(1.1);

  tPadA->cd();
  tTempHist1->Draw("ex0");
  tTempHist2->Draw("ex0same");

  TLegend* tLegA = new TLegend(0.55, 0.20, 0.85, 0.70);
  tLegA->SetLineWidth(0);
  tLegA->SetHeader(TString::Format("%s Residuals", cAnalysisRootTags[tTempResidualType]));
  tLegA->AddEntry((TObject*)0, TString::Format("#lambda = %0.3f \t #lambda_{Tot} = %0.3f", aResidual.GetLambdaFactor(), aParamsOverall[0]), "");
  tLegA->AddEntry(tTempHist1, "Residual", "p");
  tLegA->AddEntry(tTempHist2, "Transformed", "p");
  tLegA->Draw();

  tPadB->cd();
  tTempHist2->Draw("ex0");
  TLegend* tLegB = new TLegend(0.40, 0.60, 0.60, 0.70);
  tLegB->SetLineWidth(0);
  tLegB->AddEntry(tTempHist2, "Transformed", "p");
  tLegB->Draw();

  tPadC->cd();
  tTempTransformMatrix->Draw("colz");
  tText = new TPaveText(tTextXmin,tTextYmin,tTextXmax,tTextYmax,"NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->AddText(tBoxText);
  tText->Draw();

  tPadD->cd();
  tPadD->SetLogz(true);
  tTempTransformMatrix->Draw("colz");
  tText = new TPaveText(tTextXmin,tTextYmin,tTextXmax,tTextYmax,"NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->AddText(tBoxText);
  tText->Draw();

  return tReturnCan;
}

//________________________________________________________________________________________________________________
template <typename T>
TCanvas* FitGeneratorAndDraw::GetResidualsWithTransformMatricesv2(int aAnalysisNumber, T& aResidual, td1dVec &aParamsOverall, int aOffset)
{
  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnType = tFitPairAnalysis->GetAnalysisType();
  CentralityType tCentType = tFitPairAnalysis->GetCentralityType();
  double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);
  //-----------------------------------------------------------------------------------
  double tXLow = 0.0;
  double tXHigh = 0.3;

  vector<int> tResBaseColors{7,8,9,30,33,40,41,  44,46,47,49};
  vector<int> tResMarkerStyles{20,21,22,33,34,29,23,  20,21,22,33};
  vector<int> tTransformedResMarkerStyles{24,25,26,27,28,30,32,  24,25,26,27};

  //-----------------------------------------------------------------------------------
  TCanvas* tReturnCan;
  TPad *tPadA, *tPadB, *tPadC;
  TString tCanvasBaseName = TString::Format("Residualsv2_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);
  TString tTempName, tTempName1, tTempName2;
  TString tMotherName1, tDaughterName1;
  TString tMotherName2, tDaughterName2;
  TString tBoxText;
  TPaveText* tText;
    double tTextXmin = 0.20;
    double tTextXmax = 0.40;
    double tTextYmin = 0.75;
    double tTextYmax = 0.85;


  AnalysisType tTempResidualType = aResidual.GetResidualType();

  tTempName = TString(cAnalysisBaseTags[tTempResidualType]);
  tTempName1 = TString("Residual_") + tTempName;
  tTempName2 = TString("TransformedResidual_") + tTempName;

  tMotherName1 = GetPDGRootName(aResidual.GetMotherType1());
  tMotherName2 = GetPDGRootName(aResidual.GetMotherType2());
  tDaughterName1 = GetPDGRootName(aResidual.GetDaughterType1());
  tDaughterName2 = GetPDGRootName(aResidual.GetDaughterType2());

  tBoxText = TString::Format("%s%s To %s%s", tMotherName1.Data(), tMotherName2.Data(), tDaughterName1.Data(), tDaughterName2.Data());

  TH1D* tTempHist1 = aResidual.GetResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName1, aParamsOverall.data(), tWeightedNorm);
  TH1D* tTempHist2 = aResidual.GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName2, aParamsOverall.data(), tWeightedNorm);

  TH2D* tTempTransformMatrix = aResidual.GetTransformMatrix();
  tTempTransformMatrix->GetXaxis()->SetTitle(TString("#it{k}*_{") + tDaughterName1 + tDaughterName2 + TString("}(GeV/#it{c})"));
    tTempTransformMatrix->GetXaxis()->SetTitleSize(0.045);
    tTempTransformMatrix->GetXaxis()->SetTitleOffset(1.15);

  tTempTransformMatrix->GetYaxis()->SetTitle(TString("#it{k}*_{") + tMotherName1 + tMotherName2 + TString("}(GeV/#it{c})"));
    tTempTransformMatrix->GetYaxis()->SetTitleSize(0.045);
    tTempTransformMatrix->GetYaxis()->SetTitleOffset(1.1);

  tTempTransformMatrix->GetZaxis()->SetLabelSize(0.045);
  tTempTransformMatrix->GetZaxis()->SetLabelOffset(0.0045);

  tReturnCan = new TCanvas(TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]),
                           TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]));
  tReturnCan->cd();

  tPadA = new TPad(TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.5, 0.5, 1.0);
  tPadA->SetLeftMargin(0.125);
  tPadA->SetBottomMargin(0.125);
  tPadA->Draw();

  tPadB = new TPad(TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.0, 0.5, 0.5);
  tPadB->SetLeftMargin(0.125);
  tPadB->SetBottomMargin(0.125);
  tPadB->Draw();

  tPadC = new TPad(TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.5, 0.25, 1.0, 0.75);
  tPadC->SetRightMargin(0.15);
  tPadC->SetLeftMargin(0.125);
  tPadC->SetBottomMargin(0.125);
  tPadC->Draw();

  tTempHist1->SetMarkerStyle(tResMarkerStyles[aOffset]);
  tTempHist1->SetMarkerColor(tResBaseColors[aOffset]);
  tTempHist1->SetLineColor(tResBaseColors[aOffset]);
  tTempHist1->SetMarkerSize(0.5);

  tTempHist2->SetMarkerStyle(tTransformedResMarkerStyles[aOffset]);
  tTempHist2->SetMarkerColor(tResBaseColors[aOffset]);
  tTempHist2->SetLineColor(tResBaseColors[aOffset]);
  tTempHist2->SetMarkerSize(0.5);

  tTempHist1->GetXaxis()->SetRangeUser(tXLow,tXHigh);
  tTempHist2->GetXaxis()->SetRangeUser(tXLow,tXHigh);

  tTempHist1->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tTempHist1->GetXaxis()->SetTitleSize(0.045);
    tTempHist1->GetXaxis()->SetTitleOffset(1.1);
  tTempHist2->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tTempHist2->GetXaxis()->SetTitleSize(0.045);
    tTempHist2->GetXaxis()->SetTitleOffset(1.1);

  tTempHist1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist1->GetYaxis()->SetTitleSize(0.05);
    tTempHist1->GetYaxis()->SetTitleOffset(1.1);
  tTempHist2->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist2->GetYaxis()->SetTitleSize(0.05);
    tTempHist2->GetYaxis()->SetTitleOffset(1.1);

  tPadA->cd();
  tTempHist1->Draw("ex0");
  tTempHist2->Draw("ex0same");

  TLegend* tLegA = new TLegend(0.55, 0.20, 0.85, 0.70);
  tLegA->SetLineWidth(0);
  tLegA->SetHeader(TString::Format("%s Residuals", cAnalysisRootTags[tTempResidualType]));
  tLegA->AddEntry((TObject*)0, TString::Format("#lambda = %0.3f \t #lambda_{Tot} = %0.3f", aResidual.GetLambdaFactor(), aParamsOverall[0]), "");
  tLegA->AddEntry(tTempHist1, "Residual", "p");
  tLegA->AddEntry(tTempHist2, "Transformed", "p");
  tLegA->Draw();

  tPadB->cd();
  tTempHist2->Draw("ex0");
  TLegend* tLegB = new TLegend(0.40, 0.60, 0.60, 0.70);
  tLegB->SetLineWidth(0);
  tLegB->AddEntry(tTempHist2, "Transformed", "p");
  tLegB->Draw();

  tPadC->cd();
  tTempTransformMatrix->Draw("colz");
  tText = new TPaveText(tTextXmin,tTextYmin,tTextXmax,tTextYmax,"NDC");
    tText->SetFillColor(0);
    tText->SetBorderSize(0);
    tText->AddText(tBoxText);
  tText->Draw();

  return tReturnCan;
}

//________________________________________________________________________________________________________________
TObjArray* FitGeneratorAndDraw::DrawResidualsWithTransformMatrices(int aAnalysisNumber, bool aSaveImage, bool aDrawv2)
{
  TObjArray* tReturnArray = new TObjArray();

  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnType = tFitPairAnalysis->GetAnalysisType();
  CentralityType tCentType = tFitPairAnalysis->GetCentralityType();

  double tOverallLambdaPrimary = tFitPairAnalysis->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tFitPairAnalysis->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
  //-----------------------------------------------------------------------------------
  int tNResiduals;
  if(fIncludeResidualsType==kInclude10Residuals) tNResiduals = 10;  //really are 11, but will skip Omega
  else if(fIncludeResidualsType==kInclude3Residuals) tNResiduals = 3;
  else assert(0);

  int tNNeutral=0, tNCharged=0;

  TCanvas* tCan;

  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    if( (tTempResidualType != kResOmegaK0) && (tTempResidualType != kResAOmegaK0) )
    {
      if(!aDrawv2) tCan = GetResidualsWithTransformMatrices(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes], tParamsOverall, iRes);
      else         tCan = GetResidualsWithTransformMatricesv2(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes], tParamsOverall, iRes);
      tReturnArray->Add(tCan);
      tNNeutral++;
    }
  }
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    if( (tTempResidualType != kResOmegaKchP) && (tTempResidualType != kResAOmegaKchM) && (tTempResidualType != kResOmegaKchM) && (tTempResidualType != kResAOmegaKchP) )
    {
      if(!aDrawv2) tCan = GetResidualsWithTransformMatrices(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes], tParamsOverall, iRes+tNNeutral);
      else         tCan = GetResidualsWithTransformMatricesv2(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes], tParamsOverall, iRes+tNNeutral);
      tReturnArray->Add(tCan);
      tNCharged++;
    }
  }

  assert((tNNeutral+tNCharged) == tNResiduals);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    TString tSaveLocationDir = TString::Format("%sResiduals%s/%s/%s/", fSaveLocationBase.Data(), cIncludeResidualsTypeTags[fIncludeResidualsType], cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);
    gSystem->mkdir(tSaveLocationDir, true);
    for(int i=0; i<tReturnArray->GetEntries(); i++)
    {
      TCanvas* tSaveCan = (TCanvas*)tReturnArray->At(i);
      tSaveCan->SaveAs(tSaveLocationDir+tSaveCan->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
    }
  }


  return tReturnArray;
}


//________________________________________________________________________________________________________________
TObjArray* FitGeneratorAndDraw::DrawAllResidualsWithTransformMatrices(bool aSaveImage, bool aDrawv2)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++)
  {
    CentralityType tCentType = fSharedAn->GetFitPairAnalysis(i)->GetCentralityType();
    TObjArray* tArr = DrawResidualsWithTransformMatrices(i, aSaveImage, aDrawv2);
    tReturnArray->Add(tArr);
  }
  return tReturnArray;
}





//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::CheckCorrectedCf_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType)
{
  int tOffset = 0;
  if(!fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->AreTrainResults()) tOffset = 2;

  int iPartAn = static_cast<int>(aBFieldType)-tOffset;
  FitPairAnalysis* tPairAn = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  FitPartialAnalysis* tPartAn = tPairAn->GetFitPartialAnalysis(iPartAn);

  //---------------------------------------------------------------------------------------------------------
  TH1* tCfData = (TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetCfLite(iPartAn)->Cf()->Clone();

  TF1* tNonFlatBgd;
  td1dVec tNonFlatBgdVec(0);
  if(fApplyNonFlatBackgroundCorrection)
  {
    if(!fSharedAn->UsingNewBgdTreatment()) tNonFlatBgd = (TF1*)tPartAn->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true);
    else tNonFlatBgd = (TF1*)tPartAn->GetNewNonFlatBackground(aNonFlatBgdFitType);

    for(int i=1; i<=tCfData->GetNbinsX(); i++) tNonFlatBgdVec.push_back(tNonFlatBgd->Eval(tCfData->GetBinCenter(i)));
  }

  tPartAn->CreateFitFunction(true, fIncludeResidualsType, fResPrimMaxDecayType, fLednickyFitter->GetChi2(), fLednickyFitter->GetNDF(), 0.0, 1.0);
  TF1* tPrimaryFit = tPartAn->GetPrimaryFit();
  td1dVec tPrimaryFitVec(0);
  for(int i=1; i<=tCfData->GetNbinsX(); i++) tPrimaryFitVec.push_back(tPrimaryFit->Eval(tCfData->GetBinCenter(i)));

  td1dVec tCorrectedFitVec = tPartAn->GetCorrectedFitVec();
  //---------------------------------------------------------------------------------------------------------
  //Residuals-----
  double tOverallLambdaPrimary = tPairAn->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tPairAn->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tPairAn->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tPairAn->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tPairAn->GetFitParameter(kd0)->GetFitValue();
  double tNorm = tPartAn->GetFitNormParameter()->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};

  td2dVec tResidualVecs(0);

  TH1D* tTempHist;
  td1dVec tTempVec(0);
  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    tTempHist = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    for(int i=1; i<=tTempHist->GetNbinsX(); i++) tTempVec.push_back(tTempHist->GetBinContent(i));
    tResidualVecs.push_back(tTempVec);
    tTempVec.clear();
  }
  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    tTempHist = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    for(int i=1; i<=tTempHist->GetNbinsX(); i++) tTempVec.push_back(tTempHist->GetBinContent(i));
    tResidualVecs.push_back(tTempVec);
    tTempVec.clear();
  }
  //---------------------------------------------------------------------------------------------------------
  assert(tPrimaryFitVec.size() >= tCorrectedFitVec.size());
  if(fApplyNonFlatBackgroundCorrection) assert(tPrimaryFitVec.size() == tNonFlatBgdVec.size());
  for(unsigned int i=0; i<tResidualVecs.size(); i++) assert(tPrimaryFitVec.size() == tResidualVecs[i].size());
  //---------------------------------------------------------------------------------------------------------

  td1dVec tPrimPlusRes(tCorrectedFitVec.size());
  for(unsigned int i=0; i<tCorrectedFitVec.size(); i++)
  {
    tPrimPlusRes[i] = 0.;
    tPrimPlusRes[i] += tPrimaryFitVec[i];
    for(unsigned int iRes=0; iRes<tResidualVecs.size(); iRes++) tPrimPlusRes[i] += (tResidualVecs[iRes][i]-1.0);
  }

  td1dVec tCalculatedFitCf;
  if(fApplyMomResCorrection)
  {
    td1dVec tKStarBinCenters = fLednickyFitter->GetKStarBinCenters();
    tCalculatedFitCf = LednickyFitter::ApplyMomResCorrection(tPrimPlusRes, tKStarBinCenters, tPairAn->GetModelKStarTrueVsRecMixed());
  }
  else tCalculatedFitCf = tPrimPlusRes;

  if(fApplyNonFlatBackgroundCorrection)
  {
    for(unsigned int i=0; i<tCorrectedFitVec.size(); i++) tCalculatedFitCf[i] *= tNonFlatBgdVec[i];
  }

  //---------------------------------------------------------------------------------------------------------

  cout << endl << "CheckCorrectedCf_PartAn for: " << endl;
  cout << "\t AnalysisType   = " << cAnalysisBaseTags[tPartAn->GetAnalysisType()] << endl;
  cout << "\t CentralityType = " << cCentralityTags[tPartAn->GetCentralityType()] << endl;
  cout << "\t BFieldType     = " << cBFieldTags[tPartAn->GetBFieldType()] << endl;
  cout << "--------------------------------------" << endl;

  for(unsigned int i=0; i<tCorrectedFitVec.size(); i++)
  {
    cout << "tPrimaryFitVec[" << i << "] = " << tPrimaryFitVec[i] << endl;
    for(unsigned int iRes=0; iRes<tResidualVecs.size(); iRes++) cout << TString::Format("tResidualVecs[%d][%d] - 1.0 = %0.6f", iRes, i, (tResidualVecs[iRes][i]-1.)) << endl;
    cout << "tPrimPlusRes[" << i << "] = " << tPrimPlusRes[i] << endl;
    if(fApplyNonFlatBackgroundCorrection) cout << "tNonFlatBgdVec[" << i << "] = " << tNonFlatBgdVec[i] << endl;
    cout << "\t tCalculatedFitCf[" << i << "] = " << tCalculatedFitCf[i] << endl;
    cout << "\t tCorrectedFitVec[" << i << "] = " << tCorrectedFitVec[i] << endl;
    cout << "\t\t % Diff = " << (tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] << endl;
    if(abs(tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] > 0.0001) {cout << "WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!! Diff > 0.00001 !!!!!!!!!!!!!!!!!!!" << endl; assert(0);}
    cout << endl;
  }
  cout << endl << endl << endl;
}


//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawSingleKStarCfwFitAndResiduals_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  if(aOutputCheckCorrectedCf) CheckCorrectedCf_PartAn(aAnalysisNumber, aBFieldType, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType);

  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetAnalysisType();
  CentralityType tCentType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCentralityType();

  int tOffset = 0;
  if(!fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->AreTrainResults()) tOffset = 2;

  int iPartAn = static_cast<int>(aBFieldType)-tOffset;
  FitPairAnalysis* tPairAn = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  FitPartialAnalysis* tPartAn = tPairAn->GetFitPartialAnalysis(iPartAn);

  TString tCanvasName = TString::Format("canKStarCfwFitsAndResiduals_AnNum%i%s_", aAnalysisNumber, cBFieldTags[aBFieldType]);

  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

  int tNx=2, tNy=1;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(1400,1500);


  int tColor, tColorTransparent;
  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
  double tMarkerSize = 0.50;
  //---------------- Residuals ----------------------------------------
  double tOverallLambdaPrimary = tPairAn->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tPairAn->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tPairAn->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tPairAn->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tPairAn->GetFitParameter(kd0)->GetFitValue();
  double tNorm = tPartAn->GetFitNormParameter()->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
      
  vector<int> tNeutralResBaseColors{7,8,9,30,33,40,41};
  vector<int> tNeutralResMarkerStyles{24,25,26,27,28,30,32};
  vector<int> tChargedResBaseColors{44,46,47,49};
  vector<int> tChargedResMarkerStyles{24,25,26,27};



  //---------- Left Pad ----------
  BuildKStarCfswFitsPanel_PartAn(tCanPart, aAnalysisNumber, aBFieldType, 0, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);

  //Residuals-----
  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    tCanPart->AddGraph(0,0,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
  }
  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    tCanPart->AddGraph(0,0,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
  }
  tCanPart->AddGraph(0,0,(TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top
  //End Residuals

  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);
  TString tTextBField = TString(cBFieldTags[aBFieldType]);

  TString tCombinedText = TString::Format("%s  %s%s", tTextAnType.Data(), tTextCentrality.Data(), tTextBField.Data());
  TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.65,0.875,0.15,0.10,43,13);
  tCanPart->AddPadPaveText(tCombined,0,0);


//  TString tTextAlicePrelim = TString("ALICE Preliminary");
  TString tTextAlicePrelim = TString("ALICE");
  TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,0,0,0.05,0.875,0.40,0.10,43,15);
  tCanPart->AddPadPaveText(tAlicePrelim,0,0);


  //---------- Right pad
  if(aDrawData) BuildKStarCfswFitsPanel_PartAn(tCanPart, aAnalysisNumber, aBFieldType, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);
  else BuildKStarCfswFitsPanel_PartAn(tCanPart, aAnalysisNumber, aBFieldType, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, false);

  //Residuals-----
  tCanPart->SetupTLegend(TString("Residuals"), 1, 0, 0.45, 0.05, 0.50, 0.35, 2);
  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();

    if(tTempResidualType==kResOmegaK0 || tTempResidualType==kResAOmegaK0) continue;

    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tPairAn->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    tCanPart->AddGraph(1,0,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
    tCanPart->AddLegendEntry(1, 0, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
  }

  for(unsigned int iRes=0; iRes<tPairAn->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();

    if(tTempResidualType==kResOmegaKchP || tTempResidualType==kResAOmegaKchM || tTempResidualType==kResOmegaKchM || tTempResidualType==kResAOmegaKchP) continue;

    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tPairAn->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tNorm);
    tCanPart->AddGraph(1,0,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
    tCanPart->AddLegendEntry(1, 0, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
  }

  //End Residuals

  double tMinZoomRes = 0.985, tMaxZoomRes = 1.015;
  ((TH1*)tCanPart->GetGraphsInPad(1,0)->At(0))->GetYaxis()->SetRangeUser(tMinZoomRes, tMaxZoomRes);
  tCanPart->ReplaceGraphDrawOption(1, 0, 0, "AXIS Y+");


  TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,1,0,0.50,0.875,0.40,0.10,43,15);
  tCanPart->AddPadPaveText(tSysInfo,1,0);

  td1dVec tSysErrors;
  if(fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()) tSysErrors = GetSystErrs(fIncludeResidualsType, tAnType, tCentType);
  else tSysErrors = GetSystErrs(fMasterFileLocation, fSystematicsFileLocation, fSaveNameModifier, tAnType, tCentType);

  bool bDrawAll = true;
  CreateParamFinalValuesText(tAnType, tCanPart,0,0,(TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.73,0.09,0.25,0.45,43,12.0,bDrawAll);


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);


  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    TString tSaveLocationDir = TString::Format("%sResiduals%s/%s/", fSaveLocationBase.Data(), cIncludeResidualsTypeTags[fIncludeResidualsType], cAnalysisBaseTags[tAnType]);
    gSystem->mkdir(tSaveLocationDir, true);
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }


  return tCanPart->GetCanvas();

}

//________________________________________________________________________________________________________________
TObjArray* FitGeneratorAndDraw::DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++) tReturnArray->Add(DrawSingleKStarCfwFitAndResiduals_PartAn(i, aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::CheckCorrectedCf(int aAnalysisNumber, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType)
{
  //---------------------------------------------------------------------------------------------------------
  TH1* tCfData = (TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone();

  TF1* tNonFlatBgd;
  td1dVec tNonFlatBgdVec(0);
  if(fApplyNonFlatBackgroundCorrection)
  {
    if(!fSharedAn->UsingNewBgdTreatment()) tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);
    else tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNewNonFlatBackground(aNonFlatBgdFitType, true);  //TODO second argument should be set automatically

    for(int i=1; i<=tCfData->GetNbinsX(); i++) tNonFlatBgdVec.push_back(tNonFlatBgd->Eval(tCfData->GetBinCenter(i)));
  }

  TF1* tPrimaryFit = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetPrimaryFit();
  td1dVec tPrimaryFitVec(0);
  for(int i=1; i<=tCfData->GetNbinsX(); i++) tPrimaryFitVec.push_back(tPrimaryFit->Eval(tCfData->GetBinCenter(i)));

  td1dVec tCorrectedFitVec(0);
  TH1F* tCorrectedFitHisto = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCorrectedFitHistv2(tCfData->GetXaxis()->GetBinUpEdge(tCfData->GetNbinsX()));
  for(int i=1; i<=tCorrectedFitHisto->GetNbinsX(); i++) tCorrectedFitVec.push_back(tCorrectedFitHisto->GetBinContent(i));
  //---------------------------------------------------------------------------------------------------------
  //Residuals-----
  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  double tOverallLambdaPrimary = tFitPairAnalysis->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tFitPairAnalysis->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
  double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);

  td2dVec tResidualVecs(0);

  TH1D* tTempHist;
  td1dVec tTempVec(0);
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    tTempHist = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    for(int i=1; i<=tTempHist->GetNbinsX(); i++) tTempVec.push_back(tTempHist->GetBinContent(i));
    tResidualVecs.push_back(tTempVec);
    tTempVec.clear();
  }
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    tTempHist = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    for(int i=1; i<=tTempHist->GetNbinsX(); i++) tTempVec.push_back(tTempHist->GetBinContent(i));
    tResidualVecs.push_back(tTempVec);
    tTempVec.clear();
  }
  //---------------------------------------------------------------------------------------------------------
  assert(tPrimaryFitVec.size() >= tCorrectedFitVec.size());
  if(fApplyNonFlatBackgroundCorrection) assert(tPrimaryFitVec.size() == tNonFlatBgdVec.size());
  for(unsigned int i=0; i<tResidualVecs.size(); i++) assert(tPrimaryFitVec.size() == tResidualVecs[i].size());
  //---------------------------------------------------------------------------------------------------------

  td1dVec tPrimPlusRes(tCorrectedFitVec.size());
  for(unsigned int i=0; i<tCorrectedFitVec.size(); i++)
  {
    tPrimPlusRes[i] = 0.;
    tPrimPlusRes[i] += tPrimaryFitVec[i];
    for(unsigned int iRes=0; iRes<tResidualVecs.size(); iRes++) tPrimPlusRes[i] += (tResidualVecs[iRes][i]-1.0);
  }

  td1dVec tCalculatedFitCf;
  if(fApplyMomResCorrection)
  {
    td1dVec tKStarBinCenters = fLednickyFitter->GetKStarBinCenters();
    tCalculatedFitCf = LednickyFitter::ApplyMomResCorrection(tPrimPlusRes, tKStarBinCenters, tFitPairAnalysis->GetModelKStarTrueVsRecMixed());
  }
  else tCalculatedFitCf = tPrimPlusRes;

  if(fApplyNonFlatBackgroundCorrection)
  {
    for(unsigned int i=0; i<tCorrectedFitVec.size(); i++) tCalculatedFitCf[i] *= tNonFlatBgdVec[i];
  }

  //---------------------------------------------------------------------------------------------------------

  cout << endl << "CheckCorrectedCf for: " << endl;
  cout << "\t AnalysisType   = " << cAnalysisBaseTags[tFitPairAnalysis->GetAnalysisType()] << endl;
  cout << "\t CentralityType = " << cCentralityTags[tFitPairAnalysis->GetCentralityType()] << endl;
  cout << "--------------------------------------" << endl;

  for(unsigned int i=0; i<tCorrectedFitVec.size(); i++)
  {
    cout << "tPrimaryFitVec[" << i << "] = " << tPrimaryFitVec[i] << endl;
    for(unsigned int iRes=0; iRes<tResidualVecs.size(); iRes++) cout << TString::Format("tResidualVecs[%d][%d] - 1.0 = %0.6f", iRes, i, (tResidualVecs[iRes][i]-1.)) << endl;
    cout << "tPrimPlusRes[" << i << "] = " << tPrimPlusRes[i] << endl;
    if(fApplyNonFlatBackgroundCorrection) cout << "tNonFlatBgdVec[" << i << "] = " << tNonFlatBgdVec[i] << endl;
    cout << "\t tCalculatedFitCf[" << i << "] = " << tCalculatedFitCf[i] << endl;
    cout << "\t tCorrectedFitVec[" << i << "] = " << tCorrectedFitVec[i] << endl;
    cout << "\t\t % Diff = " << (tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] << endl;
    if(abs(tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] > 0.0001) {cout << "WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!! Diff > 0.00001 !!!!!!!!!!!!!!!!!!!" << endl; assert(0);}
    cout << endl;
  }
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawSingleKStarCfwFitAndResiduals(int aAnalysisNumber, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  if(aOutputCheckCorrectedCf) CheckCorrectedCf(aAnalysisNumber, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType);

  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetAnalysisType();
  CentralityType tCentType = fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetCentralityType();

  TString tCanvasName = TString::Format("canKStarCfwFitsAndResiduals_AnNum%i_", aAnalysisNumber);

  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

  int tNx=2, tNy=1;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
  tCanPart->GetCanvas()->SetCanvasSize(1400,500);


  int tColor, tColorTransparent;
  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
  double tMarkerSize = 0.75;
  //---------------- Residuals ----------------------------------------
  FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(aAnalysisNumber);
  double tOverallLambdaPrimary = tFitPairAnalysis->GetFitParameter(kLambda)->GetFitValue();
  double tRadiusPrimary = tFitPairAnalysis->GetFitParameter(kRadius)->GetFitValue();
  double tReF0Primary = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
  double tImF0Primary = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
  double tD0Primary = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();

  td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
  double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);
      
  vector<int> tNeutralResBaseColors{7,8,9,30,33,40,41};
  vector<int> tNeutralResMarkerStyles{24,25,26,27,28,30,32};
  vector<int> tChargedResBaseColors{44,46,47,49};
  vector<int> tChargedResMarkerStyles{24,25,26,27};



  //---------- Left Pad ----------
  BuildKStarCfswFitsPanel(tCanPart, aAnalysisNumber, 0, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

  //Residuals-----
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    tCanPart->AddGraph(0,0,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
  }
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    tCanPart->AddGraph(0,0,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
  }
  tCanPart->AddGraph(0,0,(TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top
  //End Residuals

  TString tTextAnType = TString(cAnalysisRootTags[tAnType]);
  TString tTextCentrality = TString(cPrettyCentralityTags[tCentType]);

  TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
  TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.875,0.15,0.10,43,25);
  tCanPart->AddPadPaveText(tCombined,0,0);

/*
  TString tTextAlicePrelim = TString("ALICE Preliminary");
  TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,0,0,0.10,0.875,0.40,0.10,43,25);
  tCanPart->AddPadPaveText(tAlicePrelim,0,0);
*/

  //---------- Right pad
  if(aDrawData) BuildKStarCfswFitsPanel(tCanPart, aAnalysisNumber, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);
  else BuildKStarCfswFitsPanel(tCanPart, aAnalysisNumber, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, false, false);

  //Residuals-----
  tCanPart->SetupTLegend(TString("Residuals"), 1, 0, 0.45, 0.65, 0.50, 0.35, 2);
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();

    if(tTempResidualType==kResOmegaK0 || tTempResidualType==kResAOmegaK0) continue;

    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    tCanPart->AddGraph(1,0,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
    tCanPart->AddLegendEntry(1, 0, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
  }

  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();

    if(tTempResidualType==kResOmegaKchP || tTempResidualType==kResAOmegaKchM || tTempResidualType==kResOmegaKchM || tTempResidualType==kResAOmegaKchP) continue;

    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
    tCanPart->AddGraph(1,0,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
    tCanPart->AddLegendEntry(1, 0, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
  }
  tCanPart->AddGraph(1,0,(TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top
  //End Residuals

  double tMinZoomRes = 0.985, tMaxZoomRes = 1.015;
  ((TH1*)tCanPart->GetGraphsInPad(1,0)->At(0))->GetYaxis()->SetRangeUser(tMinZoomRes, tMaxZoomRes);
  tCanPart->ReplaceGraphDrawOption(1, 0, 0, "AXIS Y+");


  TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
//  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,1,0,0.50,0.875,0.40,0.10,43,25);
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,0,0,0.10,0.875,0.40,0.10,43,25);
  tCanPart->AddPadPaveText(tSysInfo,1,0);

  td1dVec tSysErrors;
  if(fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()) tSysErrors = GetSystErrs(fIncludeResidualsType, tAnType, tCentType);
  else tSysErrors = GetSystErrs(fMasterFileLocation, fSystematicsFileLocation, fSaveNameModifier, tAnType, tCentType);

  bool bDrawAll = true;
  CreateParamFinalValuesText(tAnType, tCanPart,0,0,(TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.73,0.09,0.25,0.45,43,12.0,bDrawAll);


  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);


  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    TString tSaveLocationDir = TString::Format("%sResiduals%s/%s/", fSaveLocationBase.Data(), cIncludeResidualsTypeTags[fIncludeResidualsType], cAnalysisBaseTags[tAnType]);
    gSystem->mkdir(tSaveLocationDir, true);
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }


  return tCanPart->GetCanvas();

}

//________________________________________________________________________________________________________________
TObjArray* FitGeneratorAndDraw::DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++) tReturnArray->Add(DrawSingleKStarCfwFitAndResiduals(i, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aZoomResiduals)
{
  TString tCanvasName = "canKStarCfwFitsAndResiduals";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition(tCanvasName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP, aZoomResiduals);

  TString tZoomResModifier = "";
  if(aZoomResiduals) tZoomResModifier = TString("_ZoomResiduals");

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  double tMarkerSize = 0.35;

  int tNx_Leg=0, tNy_Leg=1;
  if(aZoomResiduals) tNy_Leg=0;

  assert(tNx*tNy == fNAnalyses);
  int tAnalysisNumber=0;
  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      int tColor, tColorTransparent;
      AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType();
      if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
      else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
      else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
      else tColor=1;

      tColorTransparent = TColor::GetColorTransparent(tColor,0.2);
      //---------------- Residuals ----------------------------------------
      FitPairAnalysis* tFitPairAnalysis = fSharedAn->GetFitPairAnalysis(tAnalysisNumber);
      double tOverallLambdaPrimary = tFitPairAnalysis->GetFitParameter(kLambda)->GetFitValue();
      double tRadiusPrimary = tFitPairAnalysis->GetFitParameter(kRadius)->GetFitValue();
      double tReF0Primary = tFitPairAnalysis->GetFitParameter(kRef0)->GetFitValue();
      double tImF0Primary = tFitPairAnalysis->GetFitParameter(kImf0)->GetFitValue();
      double tD0Primary = tFitPairAnalysis->GetFitParameter(kd0)->GetFitValue();

      td1dVec tParamsOverall{tOverallLambdaPrimary, tRadiusPrimary, tReF0Primary, tImF0Primary, tD0Primary};
      double tWeightedNorm = GetWeightedAnalysisNorm(tFitPairAnalysis);
      
      vector<int> tNeutralResBaseColors{7,8,9,30,33,40,41};
      vector<int> tNeutralResMarkerStyles{24,25,26,27,28,30,32};
      vector<int> tChargedResBaseColors{44,46,47,49};
      vector<int> tChargedResMarkerStyles{24,25,26,27};
      if(i==tNx_Leg && j==tNy_Leg) 
      {
        if(!aZoomResiduals) tCanPart->SetupTLegend(TString("Residuals"), i, j, 0.25, 0.05, 0.35, 0.50, 2);
        else tCanPart->SetupTLegend(TString("Residuals"), i, j, 0.50, 0.05, 0.35, 0.50, 2);
      }
      for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
      {
        AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();

        if(tTempResidualType==kResOmegaK0 || tTempResidualType==kResAOmegaK0) continue;

        TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
        TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
        tCanPart->AddGraph(i,j,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
        if(i==tNx_Leg && j==tNy_Leg) tCanPart->AddLegendEntry(i, j, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
      }
      for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
      {
        AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();

        if(tTempResidualType==kResOmegaKchP || tTempResidualType==kResAOmegaKchM || tTempResidualType==kResOmegaKchM || tTempResidualType==kResAOmegaKchP) continue;

        TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
        TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
        tCanPart->AddGraph(i,j,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
        if(i==tNx_Leg && j==tNy_Leg) tCanPart->AddLegendEntry(i, j, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
      }
      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top

      if(aZoomResiduals)
      {
        double tMinZoomRes = 0.961, tMaxZoomRes = 1.024;
        ((TH1*)tCanPart->GetGraphsInPad(i,j)->At(0))->GetYaxis()->SetRangeUser(tMinZoomRes, tMaxZoomRes);
      }
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    TString tSaveLocationDir = TString::Format("%sResiduals%s/%s/", fSaveLocationBase.Data(), cIncludeResidualsTypeTags[fIncludeResidualsType], cAnalysisBaseTags[fSharedAn->GetFitPairAnalysis(0)->GetAnalysisType()]);
    gSystem->mkdir(tSaveLocationDir, true);
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+tZoomResModifier.Data()+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}





//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawModelKStarCfs(bool aSaveImage)
{
  TString tCanvasName = TString("canModelKStarCfs");
  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);

  int tNx=1, tNy=1;
  if(fNAnalyses > 1) tNx = 2;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.71;
  double tYHigh = 1.09;
  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  int tAnalysisNumber=0;

  int tMarkerStyle = 20;
  int tMarkerColor = 1;
  double tMarkerSize = 0.5;

  if(fPairType==kLamK0 || fPairType==kALamK0) tMarkerColor = 1;
  else if(fPairType==kLamKchP || fPairType==kALamKchM) tMarkerColor = 2;
  else if(fPairType==kLamKchM || fPairType==kALamKchP) tMarkerColor = 4;
  else tMarkerColor=1;

  for(int j=0; j<tNy; j++)
  {
    for(int i=0; i<tNx; i++)
    {
      tAnalysisNumber = j*tNx + i;

      //---------------------------------------------------------------------------------------------------------

      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetModelKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",tMarkerStyle,tMarkerColor,tMarkerSize);

      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.8,0.85);
      tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.05,0.85);
      tCanPart->AddPadPaveText(tCentralityName,i,j);
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("k* (GeV/c)");
  tCanPart->DrawYaxisTitle("C(k*)",43,25,0.05,0.75);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
CfHeavy* FitGeneratorAndDraw::CombineTwoHeavyCfs(CfHeavy *aCf1, CfHeavy* aCf2)
{
  TString aReturnCfName = TString::Format("%s_and_%s", aCf1->GetHeavyCfName().Data(), aCf2->GetHeavyCfName().Data());

  vector<CfLite*> tCfLiteColl1 = aCf1->GetCfLiteCollection();
  vector<CfLite*> tCfLiteColl2 = aCf2->GetCfLiteCollection();

  vector<CfLite*> tReturnCfLiteColl(0);
  for(int i=0; i<tCfLiteColl1.size(); i++) tReturnCfLiteColl.push_back(tCfLiteColl1[i]);
  for(int i=0; i<tCfLiteColl2.size(); i++) tReturnCfLiteColl.push_back(tCfLiteColl2[i]);

  CfHeavy* tReturnCf = new CfHeavy(aReturnCfName, aReturnCfName, tReturnCfLiteColl, 0.32, 0.40);
  return tReturnCf;
}

//________________________________________________________________________________________________________________
TH1* FitGeneratorAndDraw::CombineTwoHists(TH1* aHist1, TH1* aHist2, double aNorm1, double aNorm2)
{
  TString aReturnName = TString::Format("%s_and_%s", aHist1->GetName(), aHist2->GetName());

  if(!aHist1->GetSumw2N()) aHist1->Sumw2();
  if(!aHist2->GetSumw2N()) aHist2->Sumw2();

  TH1* tReturnHist = (TH1*)aHist1->Clone(aReturnName);
    tReturnHist->Scale(aNorm1);
  tReturnHist->Add(aHist2, aNorm2);
  tReturnHist->Scale(1./(aNorm1+aNorm2));

  return tReturnHist;
}

//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::BuildKStarCfswFitsPanel_CombineConj(CanvasPartition* aCanPart, int aAnNumber, int aConjAnNumber, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aDrawDataOnTop)
{
  assert(fNAnalyses == 6);

  int tColor, tColorTransparent;

  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnNumber)->GetAnalysisType();
  AnalysisType tConjAnType = fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetAnalysisType();

  assert(fSharedAn->GetFitPairAnalysis(aAnNumber)->GetCentralityType() == fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetCentralityType());

  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  int tColorCorrectFit = kMagenta+1;
  int tColorNonFlatBgd = kGreen+2;
  //---------------------------------------------------------------------------------------------------------
  CfHeavy* tCfHeavyData_An   = (CfHeavy*)fSharedAn->GetKStarCfHeavy(aAnNumber);
  CfHeavy* tCfHeavyData_Conj = (CfHeavy*)fSharedAn->GetKStarCfHeavy(aConjAnNumber);
  CfHeavy* tCfHeavyData      = CombineTwoHeavyCfs(tCfHeavyData_An, tCfHeavyData_Conj);
  TH1* tCfData               = tCfHeavyData->GetHeavyCfClone();


  //TODO tCfHeavyData_An->GetTotalNumScale() or FitGeneratorAndDraw::GetWeightedAnalysisNorm(aAnNumber)???
  double tNorm_An   = tCfHeavyData_An->GetTotalNumScale();
  double tNorm_Conj = tCfHeavyData_Conj->GetTotalNumScale();

  TF1 *tNonFlatBgd_An,     *tNonFlatBgd_Conj;
  TH1 *tNonFlatBgdHist_An, *tNonFlatBgdHist_Conj;
  TH1* tNonFlatBgd;
  if(fApplyNonFlatBackgroundCorrection)
  {
    assert(!fSharedAn->UsingNewBgdTreatment());
    tNonFlatBgd_An   = (TF1*)fSharedAn->GetFitPairAnalysis(aAnNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);
    tNonFlatBgd_Conj = (TF1*)fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);
    tNonFlatBgdHist_An   = (TH1*)tNonFlatBgd_An->GetHistogram()->Clone();
    tNonFlatBgdHist_Conj = (TH1*)tNonFlatBgd_Conj->GetHistogram()->Clone();
    tNonFlatBgd = CombineTwoHists(tNonFlatBgdHist_An, tNonFlatBgdHist_Conj, tNorm_An, tNorm_Conj);
    tNonFlatBgd->SetLineStyle(7);
  }

  TF1* tPrimaryFit_An     = (TF1*)fSharedAn->GetFitPairAnalysis(aAnNumber)->GetPrimaryFit();
  TF1* tPrimaryFit_ConjAn = (TF1*)fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetPrimaryFit();
  TH1* tPrimaryFitHist_An   = (TH1*)tPrimaryFit_An->GetHistogram()->Clone();
  TH1* tPrimaryFitHist_Conj = (TH1*)tPrimaryFit_ConjAn->GetHistogram()->Clone();
  TH1* tPrimaryFit          = CombineTwoHists(tPrimaryFitHist_An, tPrimaryFitHist_Conj, tNorm_An, tNorm_Conj);
    tPrimaryFit->SetLineStyle(3);

  TH1* tCorrectedFitHisto_An   = fSharedAn->GetFitPairAnalysis(aAnNumber)->GetCorrectedFitHistv2();
  TH1* tCorrectedFitHisto_Conj = fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetCorrectedFitHistv2();
  TH1* tCorrectedFitHisto      = CombineTwoHists(tCorrectedFitHisto_An, tCorrectedFitHisto_Conj, tNorm_An, tNorm_Conj);
    tCorrectedFitHisto->SetLineWidth(2);

  //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
  TH1 *tCfwSysErrs_An, *tCfwSysErrs_Conj, *tCfwSysErrs;
  if(aDrawSysErrors)
  {
    tCfwSysErrs_An   = (TH1*)fSharedAn->GetFitPairAnalysis(aAnNumber)->GetCfwSysErrors();
    tCfwSysErrs_Conj = (TH1*)fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetCfwSysErrors();
    tCfwSysErrs = CombineTwoHists(tCfwSysErrs_An, tCfwSysErrs_Conj, tNorm_An, tNorm_Conj);
      tCfwSysErrs->SetFillColor(tColorTransparent);
      tCfwSysErrs->SetFillStyle(1000);
      tCfwSysErrs->SetLineColor(0);
      tCfwSysErrs->SetLineWidth(0);
  }

  //double tMarkerSize = 0.5;
  double tMarkerSize = 0.75;
  //---------------------------------------------------------------------------------------------------------
  if(aDrawSysErrors) assert(tCfwSysErrs->GetBinWidth(1) == tCfData->GetBinWidth(1));
  //---------------------------------------------------------------------------------------------------------
  aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,tMarkerSize,"ex0");  //ex0 suppresses the error along x
  if(fApplyNonFlatBackgroundCorrection) aCanPart->AddGraph(tColumn, tRow, tNonFlatBgd, "", 20, tColorNonFlatBgd, tMarkerSize, "HIST lsame");
  aCanPart->AddGraph(tColumn, tRow, tPrimaryFit, "", 20, 1, 0., "HIST lsame");
  aCanPart->AddGraph(tColumn,tRow,tCorrectedFitHisto, "", 20, tColorCorrectFit, tMarkerSize, "lsame");
  if(aDrawSysErrors) aCanPart->AddGraph(tColumn, tRow, tCfwSysErrs, "", 20, tColorTransparent, tMarkerSize, "e2psame");
  if(aDrawDataOnTop) aCanPart->AddGraph(tColumn, tRow, tCfData, "", 20, tColor, tMarkerSize, "ex0same");  //draw again so data on top
  
/*
  FILE* tOutput = stdout;
  double tXAxisHigh = aCanPart->GetAxesRanges()[1];
  HistInfoPrinter::PrintHistInfowStatAndSystYAML(tCfData, tCfwSysErrs, tOutput, 0., tXAxisHigh);
*/
}


//________________________________________________________________________________________________________________
CanvasPartition* FitGeneratorAndDraw::BuildKStarCfswFitsCanvasPartition_CombineConj(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  assert(fNAnalyses == 6);

  TString tCanvasName = aCanvasBaseName;
  tCanvasName += TString::Format("_%s%s", cAnalysisBaseTags[fPairType], cAnalysisBaseTags[fConjPairType]);


  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");
  if(aLabelLines) tCanvasName += TString("_LabelLines");

  int tNx=1, tNy=3;

  double tXLow = -0.02;
  double tXHigh = 0.99;
  double tYLow = 0.86;
  double tYHigh = 1.07;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
  }

  CanvasPartition* tCanPart = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.0025,0.13,0.0025);
  tCanPart->SetDrawOptStat(false);
//  tCanPart->GetCanvas()->SetCanvasSize(1400,1500);

  int tAnalysisNumberA=0, tAnalysisNumberB=0;
  for(int j=0; j<tNy; j++)
  {
    tAnalysisNumberA = 2*j;
    tAnalysisNumberB = tAnalysisNumberA+1;

    AnalysisType tAnType   = fSharedAn->GetFitPairAnalysis(tAnalysisNumberA)->GetAnalysisType();
    AnalysisType tConjType = fSharedAn->GetFitPairAnalysis(tAnalysisNumberB)->GetAnalysisType();

    CentralityType tCentType = fSharedAn->GetFitPairAnalysis(tAnalysisNumberA)->GetCentralityType();
    assert(tCentType == fSharedAn->GetFitPairAnalysis(tAnalysisNumberB)->GetCentralityType());

    BuildKStarCfswFitsPanel_CombineConj(tCanPart, tAnalysisNumberA, tAnalysisNumberB, 0, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

    TString tCombinedText = TString::Format("%s#scale[0.5]{ }#oplus#scale[0.5]{ }%s  %s", cAnalysisRootTags[tAnType], cAnalysisRootTags[tConjType], cPrettyCentralityTags[tCentType]);
    TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,0,j,0.70,0.825,0.15,0.10,43,20);
    tCanPart->AddPadPaveText(tCombined,0,j);

    if(j==(tNy-1))
    {
      TString tTextSysInfo = TString("ALICE Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
      TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,0,j,0.20,0.125,0.725,0.15,43,17);
      tCanPart->AddPadPaveText(tSysInfo,0,j);
    }

    td1dVec tSysErrors;
    if(fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()) 
    {
      if(aDrawSysErrors)
      {
        cout << "WARNING: fMasterFileLocation.IsNull() || fSystematicsFileLocation.IsNull()" << endl << "Continue?" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
      tSysErrors = GetSystErrs(fIncludeResidualsType, tAnType, tCentType);
    }
    else tSysErrors = GetSystErrs(fMasterFileLocation, fSystematicsFileLocation, fSaveNameModifier, tAnType, tCentType);

    bool bDrawAll = false;
    if(j==0) bDrawAll = true;
    if(!aSuppressFitInfoOutput) CreateParamFinalValuesText(tAnType, tCanPart, 0, j, (TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumberA)->GetPrimaryFit(), tSysErrors, 0.73, 0.09, 0.25, 0.53, 43, 12.0, bDrawAll);
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  if(aLabelLines)
  {
    assert(aSuppressFitInfoOutput);  //Not enough room for everyone!
    AddColoredLinesLabels(tCanPart, 0, 0, aZoomROP);
  }

  return tCanPart;
}

//________________________________________________________________________________________________________________
TCanvas* FitGeneratorAndDraw::DrawKStarCfswFits_CombineConj(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aSuppressFitInfoOutput, bool aLabelLines)
{
  TString tCanvasBaseName = "canKStarCfwFits_CombineConj";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition_CombineConj(tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP, aSuppressFitInfoOutput, aLabelLines);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanPart->GetCanvas()->GetName()+TString::Format(".%s", fSaveFileType.Data()));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
void FitGeneratorAndDraw::BuildKStarCfsPanel_CombineConj(CanvasPartition* aCanPart, int aAnNumber, int aConjAnNumber, int tColumn, int tRow, bool aDrawSysErrors, bool aDrawDataOnTop)
{
  assert(fNAnalyses == 6);

  int tColor, tColorTransparent;

  AnalysisType tAnType = fSharedAn->GetFitPairAnalysis(aAnNumber)->GetAnalysisType();
  AnalysisType tConjAnType = fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetAnalysisType();

  assert(fSharedAn->GetFitPairAnalysis(aAnNumber)->GetCentralityType() == fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetCentralityType());

  if(tAnType==kLamK0 || tAnType==kALamK0) tColor=kBlack;
  else if(tAnType==kLamKchP || tAnType==kALamKchM) tColor=kRed+1;
  else if(tAnType==kLamKchM || tAnType==kALamKchP) tColor=kBlue+1;
  else tColor=1;

  tColorTransparent = TColor::GetColorTransparent(tColor,0.2);

  //---------------------------------------------------------------------------------------------------------
  CfHeavy* tCfHeavyData_An   = (CfHeavy*)fSharedAn->GetKStarCfHeavy(aAnNumber);
  CfHeavy* tCfHeavyData_Conj = (CfHeavy*)fSharedAn->GetKStarCfHeavy(aConjAnNumber);
  CfHeavy* tCfHeavyData      = CombineTwoHeavyCfs(tCfHeavyData_An, tCfHeavyData_Conj);
  TH1* tCfData               = tCfHeavyData->GetHeavyCfClone();


  //TODO tCfHeavyData_An->GetTotalNumScale() or FitGeneratorAndDraw::GetWeightedAnalysisNorm(aAnNumber)???
  double tNorm_An   = tCfHeavyData_An->GetTotalNumScale();
  double tNorm_Conj = tCfHeavyData_Conj->GetTotalNumScale();

  //Include the Cf with statistical errors, and make sure the binning is the same as the fitted Cf ----------
  TH1 *tCfwSysErrs_An, *tCfwSysErrs_Conj, *tCfwSysErrs;
  if(aDrawSysErrors)
  {
    tCfwSysErrs_An   = (TH1*)fSharedAn->GetFitPairAnalysis(aAnNumber)->GetCfwSysErrors();
    tCfwSysErrs_Conj = (TH1*)fSharedAn->GetFitPairAnalysis(aConjAnNumber)->GetCfwSysErrors();
    tCfwSysErrs = CombineTwoHists(tCfwSysErrs_An, tCfwSysErrs_Conj, tNorm_An, tNorm_Conj);
      tCfwSysErrs->SetFillColor(tColorTransparent);
      tCfwSysErrs->SetFillStyle(1000);
      tCfwSysErrs->SetLineColor(0);
      tCfwSysErrs->SetLineWidth(0);
  }

  //---------------------------------------------------------------------------------------------------------
  if(aDrawSysErrors) assert(tCfwSysErrs->GetBinWidth(1) == tCfData->GetBinWidth(1));
  //---------------------------------------------------------------------------------------------------------
  aCanPart->AddGraph(tColumn,tRow,tCfData,"",20,tColor,0.5,"ex0");  //ex0 suppresses the error along x
  if(aDrawSysErrors) aCanPart->AddGraph(tColumn, tRow, tCfwSysErrs, "", 20, tColorTransparent, 0.5, "e2psame");
  if(aDrawDataOnTop) aCanPart->AddGraph(tColumn, tRow, tCfData, "", 20, tColor, 0.5, "ex0same");  //draw again so data on top
}



