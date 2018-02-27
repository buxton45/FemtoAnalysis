///////////////////////////////////////////////////////////////////////////
// FitGenerator:                                                         //
///////////////////////////////////////////////////////////////////////////


#include "FitGenerator.h"

#ifdef __ROOT__
ClassImp(FitGenerator)
#endif


//GLOBAL!!!!!!!!!!!!!!!
LednickyFitter *GlobalFitter = NULL;

//______________________________________________________________________________
void GlobalFCN(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  GlobalFitter->CalculateFitFunction(npar,f,par);
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
  fSaveLocationBase(""),
  fSaveNameModifier(""),
  fContainsMC(false),
  fNAnalyses(0),
  fGeneratorType(aGeneratorType),
  fPairType(kLamK0), fConjPairType(kALamK0),
  fCentralityType(kMB),
  fCentralityTypes(aCentralityTypes),

  fRadiusFitParams(),
  fScattFitParams(),
  fLambdaFitParams(),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fFitParamsPerPad(),

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fApplyMomResCorrection(false),
  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fUsemTScalingOfResidualRadii(false),
  fmTScalingPowerOfResidualRadii(-0.5),

  fSharedAn(0),
  fLednickyFitter(0)

{
  switch(aAnalysisType) {
  case kLamK0:
  case kALamK0:
    fPairType = kLamK0;
    fConjPairType = kALamK0;
    break;

  case kLamKchP:
  case kALamKchM:
    fPairType = kLamKchP;
    fConjPairType = kALamKchM;
    break;

  case kLamKchM:
  case kALamKchP:
    fPairType = kLamKchM;
    fConjPairType = kALamKchP;
    break;

  default:
    cout << "Error in FitGenerator constructor, invalide aAnalysisType = " << aAnalysisType << " selected." << endl;
    assert(0);
  }

  fRadiusFitParams.reserve(10);  //Not really needed, but without these reserve calls, every time a new item is emplaced back
  fScattFitParams.reserve(10);   //  instead of simply adding the object to the vector, the vector is first copied into a new vector
  fLambdaFitParams.reserve(10);  //  with a larger size, and then the new object is included
                                 //  This is why I was initially seeing my FitParameter destructor being called

  fScattFitParams.emplace_back(kRef0, 0.0);
  fScattFitParams.emplace_back(kImf0, 0.0);
  fScattFitParams.emplace_back(kd0, 0.0);

  vector<FitPairAnalysis*> tVecOfPairAn;

  for(unsigned int i=0; i<fCentralityTypes.size(); i++)
  {
    fRadiusFitParams.emplace_back(kRadius, 0.0);

    switch(fGeneratorType) {
    case kPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier));
      break;

    default:
      cout << "Error in FitGenerator constructor, invalide fGeneratorType = " << fGeneratorType << " selected." << endl;
      assert(0);
    }
  }

  fSharedAn = new FitSharedAnalyses(tVecOfPairAn);
  SetNAnalyses();
  SetDefaultSharedParameters();

}




//________________________________________________________________________________________________________________
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier) :
  fSaveLocationBase(""),
  fSaveNameModifier(""),
  fContainsMC(false),
  fNAnalyses(0),
  fGeneratorType(aGeneratorType),
  fPairType(kLamK0), fConjPairType(kALamK0),
  fCentralityType(aCentralityType),
  fCentralityTypes(0),

  fRadiusFitParams(),
  fScattFitParams(),
  fLambdaFitParams(),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fFitParamsPerPad(),

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fApplyMomResCorrection(false),
  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fUsemTScalingOfResidualRadii(false),
  fmTScalingPowerOfResidualRadii(-0.5),

  fSharedAn(0),
  fLednickyFitter(0)

{
  vector<CentralityType> tCentralityTypes(0);
  switch(aCentralityType) {
  case k0010:
  case k1030:
  case k3050:
    tCentralityTypes.push_back(aCentralityType);
    break;

  case kMB:
    tCentralityTypes.push_back(k0010);
    tCentralityTypes.push_back(k1030);
    tCentralityTypes.push_back(k3050);
    break;

  default:
    cout << "Error in FitGenerator constructor, invalide aCentralityType = " << aCentralityType << " selected." << endl;
    assert(0);
  }

  *this = FitGenerator(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, tCentralityTypes, aRunType, aNPartialAnalysis, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier);
}


//________________________________________________________________________________________________________________
FitGenerator::~FitGenerator()
{
  cout << "FitGenerator object is being deleted!!!!!" << endl;
}

//________________________________________________________________________________________________________________
void FitGenerator::SetNAnalyses()
{
  if(fAllShareSingleLambdaParam) fLambdaFitParams.emplace_back(kLambda,0.0);
  else
  {
    for(unsigned int i=0; i<fCentralityTypes.size(); i++)
    {
      fLambdaFitParams.emplace_back(kLambda,0.0);
      if(!fShareLambdaParams) fLambdaFitParams.emplace_back(kLambda,0.0);
    }
  }
  fNAnalyses = (int)fCentralityTypes.size();
  if(fGeneratorType==kPairwConj) fNAnalyses *= 2;

  fFitParamsPerPad.clear();
  fFitParamsPerPad.resize(fNAnalyses);
  for(int i=0; i<fNAnalyses; i++)
  {
    fFitParamsPerPad[i].reserve(8);  //big enough for case of singlet in triplet
    fFitParamsPerPad[i].emplace_back(kLambda,0.);
    fFitParamsPerPad[i].emplace_back(kRadius,0.);
    fFitParamsPerPad[i].emplace_back(kRef0,0.);
    fFitParamsPerPad[i].emplace_back(kImf0,0.);
    fFitParamsPerPad[i].emplace_back(kd0,0.);
  }
}


//________________________________________________________________________________________________________________
void FitGenerator::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitGenerator::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitGenerator::CreateParamInitValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
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
  tText->GetLine(0)->SetTextFont(63);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGenerator::CreateParamFinalValuesText(CanvasPartition *aCanPart, int aNx, int aNy, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
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
  tText->GetLine(0)->SetTextFont(63);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGenerator::CreateParamFinalValuesText(AnalysisType aAnType, CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const double* aSysErrors, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize, bool aDrawAll)
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
//  tText->GetLine(0)->SetTextFont(63);
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
void FitGenerator::CreateParamFinalValuesTextTwoColumns(CanvasPartition *aCanPart, int aNx, int aNy, TF1* aFit, const double* aSysErrors, double aText1Xmin, double aText1Ymin, double aText1Width, double aText1Height, bool aDrawText1, double aText2Xmin, double aText2Ymin, double aText2Width, double aText2Height, bool aDrawText2, double aTextFont, double aTextSize)
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
void FitGenerator::AddTextCorrectionInfo(CanvasPartition *aCanPart, int aNx, int aNy, bool aMomResCorrect, bool aNonFlatCorrect, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight, double aTextFont, double aTextSize)
{
  if(!aMomResCorrect && !aNonFlatCorrect) return;

  TPaveText *tText = aCanPart->SetupTPaveText("",aNx,aNy,aTextXmin,aTextYmin,aTextWidth,aTextHeight,aTextFont,aTextSize);
  if(aMomResCorrect) tText->AddText("Mom. Res. Correction");
  if(aNonFlatCorrect) tText->AddText("Non-flat Bgd Correction");
  tText->SetTextAlign(33);
  aCanPart->AddPadPaveText(tText,aNx,aNy);
}

//________________________________________________________________________________________________________________
void FitGenerator::DrawSingleKStarCf(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
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
void FitGenerator::DrawSingleKStarCfwFit(TPad* aPad, int aPairAnNumber, double aYmin, double aYmax, double aXmin, double aXmax, int aMarkerColor, TString aOption, int aMarkerStyle)
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
TCanvas* FitGenerator::DrawKStarCfs(bool aSaveImage, bool aDrawSysErrors)
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
        cout << "ERROR: FitGenerator::DrawKStarCfs: Histogram containing systematic error bars does not have the correct bin size and" << endl;
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
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString(".eps"));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
TH1D* FitGenerator::Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle)
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
void FitGenerator::BuildKStarCfswFitsPanel_PartAn(CanvasPartition* aCanPart, int aAnalysisNumber, BFieldType aBFieldType, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawDataOnTop)
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
  if(fApplyNonFlatBackgroundCorrection) tNonFlatBgd = (TF1*)tPartAn->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true);

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
CanvasPartition* FitGenerator::BuildKStarCfswFitsCanvasPartition_PartAn(BFieldType aBFieldType, TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aZoomROP)
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
  double tYLow = 0.71;
  double tYHigh = 1.09;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
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

      BuildKStarCfswFitsPanel_PartAn(tCanPart, tAnalysisNumber, aBFieldType, i, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);

      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      TString tTextBField = TString(cBFieldTags[aBFieldType]);

      TString tCombinedText = TString::Format("%s  %s%s", tTextAnType.Data(), tTextCentrality.Data(), tTextBField.Data());
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.65,0.825,0.15,0.10,63,13);
      tCanPart->AddPadPaveText(tCombined,i,j);

      if(i==0 && j==0)
      {
        TString tTextAlicePrelim = TString("ALICE Preliminary");
        TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.05,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tAlicePrelim,i,j);
      }

      if(i==1 && j==0)
      {
        TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.125,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }

      const double* tSysErrors = cSysErrors[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()][fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()];

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
TCanvas* FitGenerator::DrawKStarCfswFits_PartAn(BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP)
{
  TString tCanvasBaseName = "canKStarCfwFits";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition_PartAn(aBFieldType, tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aZoomROP);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString(".eps"));
  }

  return tCanPart->GetCanvas();
}



//________________________________________________________________________________________________________________
void FitGenerator::BuildKStarCfswFitsPanel(CanvasPartition* aCanPart, int aAnalysisNumber, int tColumn, int tRow, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aDrawDataOnTop)
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
  if(fApplyNonFlatBackgroundCorrection) tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);

  TF1* tPrimaryFit = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetPrimaryFit();

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
        cout << "ERROR: FitGenerator::DrawKStarCfswFits: Histogram containing systematic error bars does not have the correct bin size and" << endl;
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
CanvasPartition* FitGenerator::BuildKStarCfswFitsCanvasPartition(TString aCanvasBaseName, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aDrawSysErrors, bool aZoomROP)
{
  TString tCanvasName = aCanvasBaseName;
  if(fGeneratorType==kPairwConj) tCanvasName += TString(cAnalysisBaseTags[fPairType]) + TString("wConj");
  else if(fGeneratorType==kPair) tCanvasName += TString(cAnalysisBaseTags[fPairType]);
  else if(fGeneratorType==kConjPair) tCanvasName += TString(cAnalysisBaseTags[fConjPairType]);
  else assert(0);

  for(unsigned int i=0; i<fCentralityTypes.size(); i++) tCanvasName += TString(cCentralityTags[fCentralityTypes[i]]);
  if(!aZoomROP) tCanvasName += TString("UnZoomed");

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
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
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

      BuildKStarCfswFitsPanel(tCanPart, tAnalysisNumber, i, j, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

      TString tTextAnType = TString(cAnalysisRootTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()]);
      //TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.89,0.85,0.05);
      //TPaveText* tAnTypeName = tCanPart->SetupTPaveText(tTextAnType,i,j,0.715,0.825,0.05);
      //tCanPart->AddPadPaveText(tAnTypeName,i,j);

      TString tTextCentrality = TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()]);
      //TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.12,0.85,0.075);
      //TPaveText* tCentralityName = tCanPart->SetupTPaveText(tTextCentrality,i,j,0.865,0.825,0.075);
      //tCanPart->AddPadPaveText(tCentralityName,i,j);

      TString tCombinedText = tTextAnType + TString("  ") +  tTextCentrality;
      TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,i,j,0.70,0.825,0.15,0.10,63,20);
      tCanPart->AddPadPaveText(tCombined,i,j);

      if(i==0 && j==0)
      {
        TString tTextAlicePrelim = TString("ALICE Preliminary");
        //TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.30,0.85,0.40,0.10,43,15);
        TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,i,j,0.175,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tAlicePrelim,i,j);
      }

      if(i==1 && j==0)
      {
        TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
        //TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.30,0.85,0.40,0.10,43,15);
        TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,i,j,0.125,0.825,0.40,0.10,43,15);
        tCanPart->AddPadPaveText(tSysInfo,i,j);
      }
/*
      if(aZoomROP) CreateParamInitValuesText(tCanPart,i,j,0.35,0.20,0.10,0.40,43,9);
      else CreateParamInitValuesText(tCanPart,i,j,0.25,0.20,0.15,0.45,43,10);
      AddTextCorrectionInfo(tCanPart,i,j,aMomResCorrectFit,aNonFlatBgdCorrectFit,0.25,0.08,0.15,0.10,43,7.5);
*/
      const double* tSysErrors = cSysErrors[fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetAnalysisType()][fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetCentralityType()];

//      bool bDrawAll = true;

      bool bDrawAll = false;
      if(i==0 && j==0) bDrawAll = true;
      CreateParamFinalValuesText(tAnType, tCanPart,i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.73,0.09,0.25,0.53,43,12.0,bDrawAll);
/*
      bool bDrawText1 = true;
      bool bDrawText2 = false;
      if(j==0 && i==0) bDrawText2 = true;
//      CreateParamFinalValuesTextTwoColumns(tCanPart,i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.24,0.15,0.25,0.25,bDrawText1,0.74,0.10,0.25,0.50,bDrawText2,43,11);
      CreateParamFinalValuesTextTwoColumns(tCanPart,i,j,(TF1*)fSharedAn->GetFitPairAnalysis(tAnalysisNumber)->GetPrimaryFit(),tSysErrors,0.74,0.15,0.25,0.25,bDrawText1,0.37,0.10,0.25,0.50,bDrawText2,43,11);
*/
    }
  }

  tCanPart->SetDrawUnityLine(true);
  tCanPart->DrawAll();
  tCanPart->DrawXaxisTitle("#it{k}* (GeV/#it{c})");
  tCanPart->DrawYaxisTitle("#it{C}(#it{k}*)",43,25,0.05,0.85);

  return tCanPart;
}

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfswFits(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP)
{
  TString tCanvasBaseName = "canKStarCfwFits";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition(tCanvasBaseName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString(".eps"));
  }

  return tCanPart->GetCanvas();
}

//________________________________________________________________________________________________________________
double FitGenerator::GetWeightedAnalysisNorm(FitPairAnalysis* aPairAn)
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
TCanvas* FitGenerator::DrawResiduals(int aAnalysisNumber, CentralityType aCentralityType, TString aCanvasName, bool aSaveImage)
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
    tCan->SaveAs(tSaveLocationDir+tCan->GetName()+fSaveNameModifier+TString(".eps"));
  }


  return tCan;
}

//________________________________________________________________________________________________________________
TObjArray* FitGenerator::DrawAllResiduals(bool aSaveImage)
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
TCanvas* FitGenerator::GetResidualsWithTransformMatrices(int aAnalysisNumber, T& aResidual, td1dVec &aParamsOverall, int aOffset)
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
    double tTextXmin = 0.15;
    double tTextXmax = 0.35;
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
  tTempTransformMatrix->GetXaxis()->SetTitle(TString("k*_{") + tDaughterName1 + tDaughterName2 + TString("}(GeV/c)"));
    tTempTransformMatrix->GetXaxis()->SetTitleSize(0.04);
    tTempTransformMatrix->GetXaxis()->SetTitleOffset(1.1);

  tTempTransformMatrix->GetYaxis()->SetTitle(TString("k*_{") + tMotherName1 + tMotherName2 + TString("}(GeV/c)"));
    tTempTransformMatrix->GetYaxis()->SetTitleSize(0.04);
    tTempTransformMatrix->GetYaxis()->SetTitleOffset(1.2);

  tTempTransformMatrix->GetZaxis()->SetLabelSize(0.03);
  tTempTransformMatrix->GetZaxis()->SetLabelOffset(0.004);

  tReturnCan = new TCanvas(TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]),
                           TString::Format("%s_%s", tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]));
  tReturnCan->cd();

  tPadA = new TPad(TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadA_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.5, 0.5, 1.0);
  tPadA->Draw();

  tPadB = new TPad(TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadB_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.0, 0.0, 0.5, 0.5);
  tPadB->Draw();

  tPadC = new TPad(TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadC_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.5, 0.5, 1.0, 1.0);
  tPadC->SetRightMargin(0.15);
  tPadC->Draw();

  tPadD = new TPad(TString::Format("tPadD_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   TString::Format("tPadD_%s_%s",tCanvasBaseName.Data(), cAnalysisBaseTags[tTempResidualType]), 
                   0.5, 0.0, 1.0, 0.5);
  tPadD->SetRightMargin(0.15);
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
    tTempHist1->GetXaxis()->SetTitleSize(0.04);
    tTempHist1->GetXaxis()->SetTitleOffset(1.1);
  tTempHist2->GetXaxis()->SetTitle("#it{k}* (GeV/#it{c})");
    tTempHist2->GetXaxis()->SetTitleSize(0.04);
    tTempHist2->GetXaxis()->SetTitleOffset(1.1);

  tTempHist1->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist1->GetYaxis()->SetTitleSize(0.04);
    tTempHist1->GetYaxis()->SetTitleOffset(1.2);
  tTempHist2->GetYaxis()->SetTitle("#it{C}(#it{k}*)");
    tTempHist2->GetYaxis()->SetTitleSize(0.04);
    tTempHist2->GetYaxis()->SetTitleOffset(1.2);

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
TObjArray* FitGenerator::DrawResidualsWithTransformMatrices(int aAnalysisNumber, bool aSaveImage)
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
      tCan = GetResidualsWithTransformMatrices(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes], tParamsOverall, iRes);
      tReturnArray->Add(tCan);
      tNNeutral++;
    }
  }
  for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();
    if( (tTempResidualType != kResOmegaKchP) && (tTempResidualType != kResAOmegaKchM) && (tTempResidualType != kResOmegaKchM) && (tTempResidualType != kResAOmegaKchP) )
    {
      tCan = GetResidualsWithTransformMatrices(aAnalysisNumber, tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes], tParamsOverall, iRes+tNNeutral);
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
      tSaveCan->SaveAs(tSaveLocationDir+tSaveCan->GetName()+fSaveNameModifier+TString(".eps"));
    }
  }


  return tReturnArray;
}


//________________________________________________________________________________________________________________
TObjArray* FitGenerator::DrawAllResidualsWithTransformMatrices(bool aSaveImage)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++)
  {
    CentralityType tCentType = fSharedAn->GetFitPairAnalysis(i)->GetCentralityType();
    TObjArray* tArr = DrawResidualsWithTransformMatrices(i, aSaveImage);
    tReturnArray->Add(tArr);
  }
  return tReturnArray;
}





//________________________________________________________________________________________________________________
void FitGenerator::CheckCorrectedCf_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType)
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
    tNonFlatBgd = (TF1*)tPartAn->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true);
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
    cout << "tNonFlatBgdVec[" << i << "] = " << tNonFlatBgdVec[i] << endl;
    cout << "\t tCalculatedFitCf[" << i << "] = " << tCalculatedFitCf[i] << endl;
    cout << "\t tCorrectedFitVec[" << i << "] = " << tCorrectedFitVec[i] << endl;
    cout << "\t\t % Diff = " << (tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] << endl;
    if((tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] > 0.00001) {cout << "WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!! Diff > 0.00001 !!!!!!!!!!!!!!!!!!!" << endl; assert(0);}
    cout << endl;
  }
  cout << endl << endl << endl;
}


//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawSingleKStarCfwFitAndResiduals_PartAn(int aAnalysisNumber, BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
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
  double tYLow = 0.71;
  double tYHigh = 1.09;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
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
  TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.65,0.875,0.15,0.10,63,13);
  tCanPart->AddPadPaveText(tCombined,0,0);


  TString tTextAlicePrelim = TString("ALICE Preliminary");
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

  const double* tSysErrors = cSysErrors[tAnType][tCentType];

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
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString(".eps"));
  }


  return tCanPart->GetCanvas();

}

//________________________________________________________________________________________________________________
TObjArray* FitGenerator::DrawAllSingleKStarCfwFitAndResiduals_PartAn(BFieldType aBFieldType, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++) tReturnArray->Add(DrawSingleKStarCfwFitAndResiduals_PartAn(i, aBFieldType, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
void FitGenerator::CheckCorrectedCf(int aAnalysisNumber, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType)
{
  //---------------------------------------------------------------------------------------------------------
  TH1* tCfData = (TH1*)fSharedAn->GetKStarCfHeavy(aAnalysisNumber)->GetHeavyCfClone();

  TF1* tNonFlatBgd;
  td1dVec tNonFlatBgdVec(0);
  if(fApplyNonFlatBackgroundCorrection)
  {
    tNonFlatBgd = (TF1*)fSharedAn->GetFitPairAnalysis(aAnalysisNumber)->GetNonFlatBackground(aNonFlatBgdFitType, fSharedAn->GetFitType(), true, true);
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
    cout << "tNonFlatBgdVec[" << i << "] = " << tNonFlatBgdVec[i] << endl;
    cout << "\t tCalculatedFitCf[" << i << "] = " << tCalculatedFitCf[i] << endl;
    cout << "\t tCorrectedFitVec[" << i << "] = " << tCorrectedFitVec[i] << endl;
    cout << "\t\t % Diff = " << (tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] << endl;
    if((tCalculatedFitCf[i]-tCorrectedFitVec[i])/tCorrectedFitVec[i] > 0.00001) {cout << "WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!! Diff > 0.00001 !!!!!!!!!!!!!!!!!!!" << endl; assert(0);}
    cout << endl;
  }
  cout << endl << endl << endl;
}

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawSingleKStarCfwFitAndResiduals(int aAnalysisNumber, bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
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
  double tYLow = 0.71;
  double tYHigh = 1.09;
  if(aZoomROP)
  {
    tXLow = -0.02;
    tXHigh = 0.329;
    tYLow = 0.86;
    tYHigh = 1.07;
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
  TPaveText* tCombined = tCanPart->SetupTPaveText(tCombinedText,0,0,0.70,0.875,0.15,0.10,63,20);
  tCanPart->AddPadPaveText(tCombined,0,0);


  TString tTextAlicePrelim = TString("ALICE Preliminary");
  TPaveText* tAlicePrelim = tCanPart->SetupTPaveText(tTextAlicePrelim,0,0,0.10,0.875,0.40,0.10,43,15);
  tCanPart->AddPadPaveText(tAlicePrelim,0,0);


  //---------- Right pad
  if(aDrawData) BuildKStarCfswFitsPanel(tCanPart, aAnalysisNumber, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);
  else BuildKStarCfswFitsPanel(tCanPart, aAnalysisNumber, 1, 0, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, false, false);

  //Residuals-----
  tCanPart->SetupTLegend(TString("Residuals"), 1, 0, 0.45, 0.05, 0.50, 0.35, 2);
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

  //End Residuals

  double tMinZoomRes = 0.985, tMaxZoomRes = 1.015;
  ((TH1*)tCanPart->GetGraphsInPad(1,0)->At(0))->GetYaxis()->SetRangeUser(tMinZoomRes, tMaxZoomRes);
  tCanPart->ReplaceGraphDrawOption(1, 0, 0, "AXIS Y+");


  TString tTextSysInfo = TString("Pb-Pb #sqrt{#it{s}_{NN}} = 2.76 TeV");
  TPaveText* tSysInfo = tCanPart->SetupTPaveText(tTextSysInfo,1,0,0.50,0.875,0.40,0.10,43,15);
  tCanPart->AddPadPaveText(tSysInfo,1,0);

  const double* tSysErrors = cSysErrors[tAnType][tCentType];

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
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString(".eps"));
  }


  return tCanPart->GetCanvas();

}

//________________________________________________________________________________________________________________
TObjArray* FitGenerator::DrawAllSingleKStarCfwFitAndResiduals(bool aDrawData, bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP, bool aOutputCheckCorrectedCf)
{
  TObjArray* tReturnArray = new TObjArray();
  for(int i=0; i<fNAnalyses; i++) tReturnArray->Add(DrawSingleKStarCfwFitAndResiduals(i, aDrawData, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aSaveImage, aDrawSysErrors, aZoomROP, aOutputCheckCorrectedCf));
  return tReturnArray;
}


//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawKStarCfswFitsAndResiduals(bool aMomResCorrectFit, bool aNonFlatBgdCorrectFit, NonFlatBgdFitType aNonFlatBgdFitType, bool aSaveImage, bool aDrawSysErrors, bool aZoomROP)
{
  TString tCanvasName = "canKStarCfwFitsAndResiduals";
  CanvasPartition* tCanPart = BuildKStarCfswFitsCanvasPartition(tCanvasName, aMomResCorrectFit, aNonFlatBgdCorrectFit, aNonFlatBgdFitType, aDrawSysErrors, aZoomROP);

  int tNx=0, tNy=0;
  if(fNAnalyses == 6) {tNx=2; tNy=3;}
  else if(fNAnalyses == 4) {tNx=2; tNy=2;}
  else if(fNAnalyses == 3) {tNx=1; tNy=fNAnalyses;}
  else if(fNAnalyses == 2 || fNAnalyses==1) {tNx=fNAnalyses; tNy=1;}
  else assert(0);

  double tMarkerSize = 0.35;

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
      if((i==0 && j==1) || (i==0 && j==2)) tCanPart->SetupTLegend(TString("Residuals"), i, j, 0.35, 0.10, 0.25, 0.50);
      for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection().size(); iRes++)
      {
        AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetResidualType();

        if(tTempResidualType==kResOmegaK0 || tTempResidualType==kResAOmegaK0) continue;

        TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
        TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetNeutralCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
        tCanPart->AddGraph(i,j,tTempHist,"",tNeutralResMarkerStyles[iRes],tNeutralResBaseColors[iRes],tMarkerSize,"ex0same");
        if(i==0 && j==1) tCanPart->AddLegendEntry(i, j, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
      }
      for(unsigned int iRes=0; iRes<tFitPairAnalysis->GetResidualCollection()->GetChargedCollection().size(); iRes++)
      {
        AnalysisType tTempResidualType = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetResidualType();

        if(tTempResidualType==kResOmegaKchP || tTempResidualType==kResAOmegaKchM || tTempResidualType==kResOmegaKchM || tTempResidualType==kResAOmegaKchP) continue;

        TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
        TH1D* tTempHist = tFitPairAnalysis->GetResidualCollection()->GetChargedCollection()[iRes].GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(tTempName, tParamsOverall.data(), tWeightedNorm);
        tCanPart->AddGraph(i,j,tTempHist,"",tChargedResMarkerStyles[iRes],tChargedResBaseColors[iRes],tMarkerSize,"ex0same");
        if(i==0 && j==2) tCanPart->AddLegendEntry(i, j, tTempHist, cAnalysisRootTags[tTempResidualType], "p");
      }
      tCanPart->AddGraph(i,j,(TH1*)fSharedAn->GetKStarCfHeavy(tAnalysisNumber)->GetHeavyCfClone(),"",20,tColor,0.5,"ex0same");  //draw again so data on top
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
    tCanPart->GetCanvas()->SaveAs(tSaveLocationDir+tCanPart->GetCanvas()->GetName()+fSaveNameModifier+TString(".eps"));
  }

  return tCanPart->GetCanvas();
}





//________________________________________________________________________________________________________________
TCanvas* FitGenerator::DrawModelKStarCfs(bool aSaveImage)
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
    tCanPart->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+fSaveNameModifier+TString(".eps"));
  }

  return tCanPart->GetCanvas();
}


//________________________________________________________________________________________________________________
void FitGenerator::SetUseLimits(vector<FitParameter> &aVec, bool aUse)
{
  for(unsigned int i=0; i<aVec.size(); i++)
  {
    if(aUse == false)
    {
      aVec[i].SetLowerBound(0.);
      aVec[i].SetUpperBound(0.);
    }
    else
    {
      if( (aVec[i].GetLowerBound() == 0.) && (aVec[i].GetUpperBound() == 0.) )
      {
        cout << "Warning:  FitGenerator::SetUseLimits set to true but no limits have been set" << endl;
        cout << "Is this alright? (0=No, 1=Yes)" << endl;
        int tResponse;
        cin >> tResponse;
        assert(tResponse);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusStartValue(double aRad, int aIndex)
{
  if(fRadiusFitParams.size()==1) aIndex=0;  //in case, for instance, I want to run k1030 or k3050 by itself
  if(aIndex >= (int)fRadiusFitParams.size())  //if, for instance, I want to run k0010 with k3050
  {
    cout << "aIndex = " << aIndex << " and fRadiusFitParams.size() = " << fRadiusFitParams.size() << endl;
    cout << "Setting aIndex = fRadiusFitParams.size()-1....c'est bon? (0 = non, 1 = oui)" << endl;
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
    aIndex = fRadiusFitParams.size()-1;
  }
  assert(aIndex < (int)fRadiusFitParams.size());
  fRadiusFitParams[aIndex].SetStartValue(aRad);
}
//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusStartValues(const vector<double> &aStartValues)
{
  assert(aStartValues.size() == fRadiusFitParams.size());
  for(unsigned int i=0; i<fRadiusFitParams.size(); i++) fRadiusFitParams[i].SetStartValue(aStartValues[i]);
}


//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusLimits(double aMin, double aMax, int aIndex)
{
  if(fRadiusFitParams.size()==1) aIndex=0;  //in case, for instance, I want to run k1030 or k3050 by itself
  if(aIndex >= (int)fRadiusFitParams.size())  //if, for instance, I want to run k0010 with k3050
  {
    cout << "aIndex = " << aIndex << " and fRadiusFitParams.size() = " << fRadiusFitParams.size() << endl;
    cout << "Setting aIndex = fRadiusFitParams.size()-1....c'est bon? (0 = non, 1 = oui)" << endl;
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
    aIndex = fRadiusFitParams.size()-1;
  }
  assert(aIndex < (int)fRadiusFitParams.size());

  if(aMin==aMax && aMin>0.) fRadiusFitParams[aIndex].SetFixedToValue(aMin);  //aMin>0 bc aMin=aMax=0 mean unbounded
  else
  {
    fRadiusFitParams[aIndex].SetLowerBound(aMin);
    fRadiusFitParams[aIndex].SetUpperBound(aMax);
  }
}
//________________________________________________________________________________________________________________
void FitGenerator::SetRadiusLimits(const td2dVec &aMinMax2dVec)
{
  assert(aMinMax2dVec.size() == fRadiusFitParams.size());
  for(unsigned int i=0; i<aMinMax2dVec.size(); i++) assert(aMinMax2dVec[i].size()==2);
  for(unsigned int i=0; i<fRadiusFitParams.size(); i++) SetRadiusLimits(aMinMax2dVec[i][0],aMinMax2dVec[i][1],i);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetAllRadiiLimits(double aMin, double aMax)
{
  for(unsigned int i=0; i<fRadiusFitParams.size(); i++)
  {
    fRadiusFitParams[i].SetLowerBound(aMin);
    fRadiusFitParams[i].SetUpperBound(aMax);
  }
}


//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamStartValue(double aVal, ParameterType aParamType, bool aIsFixed)
{
  int tIndex = aParamType - kRef0;
  fScattFitParams[tIndex].SetStartValue(aVal);
  fScattFitParams[tIndex].SetFixed(aIsFixed);

  cout << "SetScattParamStartValue: " << TString(cParameterNames[aParamType]) << " = " << aVal << endl;
  cout << "\tDouble Check: tIndex in fScattFitParams = " << tIndex << endl << endl;
}
//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamStartValues(double aReF0, double aImF0, double aD0, bool aAreFixed)
{
  SetScattParamStartValue(aReF0,kRef0,aAreFixed);
  SetScattParamStartValue(aImF0,kImf0,aAreFixed);
  SetScattParamStartValue(aD0,kd0,aAreFixed);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamLimits(double aMin, double aMax, ParameterType aParamType)
{
  int tIndex = aParamType - kRef0;
  fScattFitParams[tIndex].SetLowerBound(aMin);
  fScattFitParams[tIndex].SetUpperBound(aMax);

  if(aMin==0. && aMax==0.) cout << "SetScattParamLimits: " << TString(cParameterNames[aParamType]) << " = NO LIMITS (studios, what's up?)" << endl;
  else cout << "SetScattParamLimits: " << aMin << " < " << TString(cParameterNames[aParamType]) << " < " << aMax << endl;
  cout << "\tDouble Check: tIndex in fScattFitParams = " << tIndex << endl << endl;
}
//________________________________________________________________________________________________________________
void FitGenerator::SetScattParamLimits(const td2dVec &aMinMax2dVec)
{
  assert(aMinMax2dVec.size() == fScattFitParams.size());
  for(unsigned int i=0; i<aMinMax2dVec.size(); i++) assert(aMinMax2dVec[i].size()==2);
  for(unsigned int i=0; i<fScattFitParams.size(); i++) SetScattParamLimits(aMinMax2dVec[i][0],aMinMax2dVec[i][1],static_cast<ParameterType>(i+kRef0));
}



//________________________________________________________________________________________________________________
int FitGenerator::GetLambdaBinNumber(bool tConjPair, CentralityType aCentType)
{
  int tBinNumber = -1;

  if(fAllShareSingleLambdaParam) tBinNumber = 0;
  else
  {
    if(fNAnalyses==1 || fNAnalyses==2) aCentType=k0010;
    else if(fNAnalyses==4) assert(aCentType < k3050);

    if(fShareLambdaParams) tBinNumber = aCentType;
    else
    {
      int tRow = aCentType;
      int tPosition = -1;
      if(!tConjPair) tPosition = 2*tRow;
      else tPosition = 2*tRow+1;

      tBinNumber = tPosition;
    }
  }
  return tBinNumber;

}

//________________________________________________________________________________________________________________
void FitGenerator::SetLambdaParamStartValue(double aLam, bool tConjPair, CentralityType aCentType, bool aIsFixed)
{
  int tBinNumber = GetLambdaBinNumber(tConjPair, aCentType);
  fLambdaFitParams[tBinNumber].SetStartValue(aLam);
  fLambdaFitParams[tBinNumber].SetFixed(aIsFixed);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetAllLambdaParamStartValues(const vector<double> &aLams, bool aAreFixed)
{
  //For now, should only be used for case of all 3 centralities, pair with conj
  //Ordering should be [Pair0010, Conj0010, Pair1030, Conj1030, Pair3050, Conj3050]
  assert(aLams.size()==6);
  assert(aLams.size()==fLambdaFitParams.size());

  SetLambdaParamStartValue(aLams[0], false, k0010, aAreFixed);
  SetLambdaParamStartValue(aLams[1], true,  k0010, aAreFixed);

  SetLambdaParamStartValue(aLams[2], false, k1030, aAreFixed);
  SetLambdaParamStartValue(aLams[3], true,  k1030, aAreFixed);

  SetLambdaParamStartValue(aLams[4], false, k3050, aAreFixed);
  SetLambdaParamStartValue(aLams[5], true,  k3050, aAreFixed);
}


//________________________________________________________________________________________________________________
void FitGenerator::SetLambdaParamLimits(double aMin, double aMax, bool tConjPair, CentralityType aCentType)
{
  int tBinNumber = GetLambdaBinNumber(tConjPair, aCentType);

  fLambdaFitParams[tBinNumber].SetLowerBound(aMin);
  fLambdaFitParams[tBinNumber].SetUpperBound(aMax);
}

//________________________________________________________________________________________________________________
void FitGenerator::SetAllLambdaParamLimits(double aMin, double aMax)
{
  for(unsigned int i=0; i<fLambdaFitParams.size(); i++)
  {
    fLambdaFitParams[i].SetLowerBound(aMin);
    fLambdaFitParams[i].SetUpperBound(aMax);
  }
}

//________________________________________________________________________________________________________________
void FitGenerator::SetDefaultSharedParameters(bool aSetAllUnbounded)
{
//TODO this seems like overkill because this is already handled at the FitPartialAnalysis level
  const double** tStartValuesPair;
  const double** tStartValuesConjPair;

  switch(fPairType) {
  case kLamK0:
    tStartValuesPair = cLamK0StartValues;
    tStartValuesConjPair = cALamK0StartValues;
    break;

  case kLamKchP:
    tStartValuesPair = cLamKchPStartValues;
    tStartValuesConjPair = cALamKchMStartValues;
    break;

  case kLamKchM:
    tStartValuesPair = cLamKchMStartValues;
    tStartValuesConjPair = cALamKchPStartValues;
    break;

  default:
    cout << "ERROR: FitGenerator::SetDefaultSharedParameters:  Invalid fPairType = " << fPairType << endl;
    assert(0);
  }

  //--------------------------------------

  //Kind of unnecessary, since pair and conjugate should have same scattering parameters
  //Nonetheless, this leaves the option open for them to be different
  if(fGeneratorType==kPair || fGeneratorType==kPairwConj)
  {
    SetScattParamStartValues(tStartValuesPair[fCentralityTypes[0]][2],tStartValuesPair[fCentralityTypes[0]][3],tStartValuesPair[fCentralityTypes[0]][4]);
  }
  else if(fGeneratorType==kConjPair)
  {
    SetScattParamStartValues(tStartValuesConjPair[fCentralityTypes[0]][2],tStartValuesConjPair[fCentralityTypes[0]][3],tStartValuesConjPair[fCentralityTypes[0]][4]);
  }
  else assert(0);
  SetScattParamLimits({{0.,0.},{0.,0.},{0.,0.}});

  //--------------------------------------
  double tRadiusMin = 0.;
  double tRadiusMax = 0.;

  double tLambdaMin = 0.0;
  double tLambdaMax = 0.0;

  if(fPairType==kLamK0)
  {
    tLambdaMin = 0.4;
    tLambdaMax = 0.6;
  }

  if(aSetAllUnbounded)
  {
    tRadiusMin = 0.;
    tRadiusMax = 0.;

    tLambdaMin = 0.;
    tLambdaMax = 0.;
  }

  //--------------------------------------

  for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++)
  {

    if(fGeneratorType==kPair)
    {
      SetRadiusStartValue(tStartValuesPair[fCentralityTypes[iCent]][1],fCentralityTypes[iCent]);
      SetRadiusLimits(tRadiusMin,tRadiusMax,iCent);

      SetLambdaParamStartValue(tStartValuesPair[fCentralityTypes[iCent]][0],false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
    }

    else if(fGeneratorType==kConjPair)
    {
      SetRadiusStartValue(tStartValuesConjPair[fCentralityTypes[iCent]][1],fCentralityTypes[iCent]);
      SetRadiusLimits(tRadiusMin,tRadiusMax,iCent);

      SetLambdaParamStartValue(tStartValuesConjPair[fCentralityTypes[iCent]][0],false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
    }

    else if(fGeneratorType==kPairwConj)
    {
      SetRadiusStartValue(tStartValuesPair[fCentralityTypes[iCent]][1],fCentralityTypes[iCent]);
      SetRadiusLimits(tRadiusMin,tRadiusMax,iCent);

      SetLambdaParamStartValue(tStartValuesPair[fCentralityTypes[iCent]][0],false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
      if(!fShareLambdaParams)
      {
        SetLambdaParamStartValue(tStartValuesConjPair[fCentralityTypes[iCent]][0],true,fCentralityTypes[iCent]);
        SetLambdaParamLimits(tLambdaMin,tLambdaMax,true,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
      }
    }

    else assert(0);
  }
}

//________________________________________________________________________________________________________________
void FitGenerator::SetDefaultLambdaParametersWithResiduals(double aMinLambda, double aMaxLambda)
{
  double tLambdaMin = aMinLambda;
  double tLambdaMax = aMaxLambda;
  double tLambdaStart = 0.9;

  for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++)
  {

    if(fGeneratorType==kPair)
    {
      SetLambdaParamStartValue(tLambdaStart,false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
    }

    else if(fGeneratorType==kConjPair)
    {
      SetLambdaParamStartValue(tLambdaStart,false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
    }

    else if(fGeneratorType==kPairwConj)
    {
      SetLambdaParamStartValue(tLambdaStart,false,fCentralityTypes[iCent]);
      SetLambdaParamLimits(tLambdaMin,tLambdaMax,false,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
      if(!fShareLambdaParams)
      {
        SetLambdaParamStartValue(tLambdaStart,true,fCentralityTypes[iCent]);
        SetLambdaParamLimits(tLambdaMin,tLambdaMax,true,fCentralityTypes[iCent]);  //TODO if(fAllShareSingleLambdaParam), do not need to set lambda for each!
      }
    }

    else assert(0);
  }
}

//________________________________________________________________________________________________________________
void FitGenerator::SetAllParameters()
{
  vector<int> Share01 {0,1};
  vector<int> Share23 {2,3};
  vector<int> Share45 {4,5};

  vector<vector<int> > tShares2dVec {{0,1},{2,3},{4,5}};

  //Always shared amongst all
  SetSharedParameter(kRef0,fScattFitParams[0].GetStartValue(),fScattFitParams[0].GetLowerBound(),fScattFitParams[0].GetUpperBound(), fScattFitParams[0].IsFixed());
  SetSharedParameter(kImf0,fScattFitParams[1].GetStartValue(),fScattFitParams[1].GetLowerBound(),fScattFitParams[1].GetUpperBound(), fScattFitParams[1].IsFixed());
  SetSharedParameter(kd0,fScattFitParams[2].GetStartValue(),fScattFitParams[2].GetLowerBound(),fScattFitParams[2].GetUpperBound(), fScattFitParams[2].IsFixed());
  if(fAllShareSingleLambdaParam) SetSharedParameter(kLambda, fLambdaFitParams[0].GetStartValue(), fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound(), fLambdaFitParams[0].IsFixed());

  if(fNAnalyses==1)
  {
    SetSharedParameter(kLambda, fLambdaFitParams[0].GetStartValue(), 
                       fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound(), fLambdaFitParams[0].IsFixed());

    SetSharedParameter(kRadius, fRadiusFitParams[0].GetStartValue(),
                       fRadiusFitParams[0].GetLowerBound(), fRadiusFitParams[0].GetUpperBound(), fRadiusFitParams[0].IsFixed());

    fFitParamsPerPad[0][0] = fLambdaFitParams[0];
    fFitParamsPerPad[0][1] = fRadiusFitParams[0];
  }

  else if(fNAnalyses==3)
  {
    if(!fAllShareSingleLambdaParam)
    {
      SetParameter(kLambda, 0, fLambdaFitParams[0].GetStartValue(),
                   fLambdaFitParams[0].GetLowerBound(), fLambdaFitParams[0].GetUpperBound(), fLambdaFitParams[0].IsFixed());
      SetParameter(kLambda, 1, fLambdaFitParams[1].GetStartValue(),
                   fLambdaFitParams[1].GetLowerBound(), fLambdaFitParams[1].GetUpperBound(), fLambdaFitParams[1].IsFixed());
      SetParameter(kLambda, 2, fLambdaFitParams[2].GetStartValue(),
                   fLambdaFitParams[2].GetLowerBound(), fLambdaFitParams[2].GetUpperBound(), fLambdaFitParams[2].IsFixed());
    }

    SetParameter(kRadius, 0, fRadiusFitParams[0].GetStartValue(),
                 fRadiusFitParams[0].GetLowerBound(), fRadiusFitParams[0].GetUpperBound(), fRadiusFitParams[0].IsFixed());
    SetParameter(kRadius, 1, fRadiusFitParams[1].GetStartValue(),
                 fRadiusFitParams[1].GetLowerBound(), fRadiusFitParams[1].GetUpperBound(), fRadiusFitParams[1].IsFixed());
    SetParameter(kRadius, 2, fRadiusFitParams[2].GetStartValue(),
                 fRadiusFitParams[2].GetLowerBound(), fRadiusFitParams[2].GetUpperBound(), fRadiusFitParams[2].IsFixed());

    for(int i=0; i<fNAnalyses; i++)
    {
      if(!fAllShareSingleLambdaParam) fFitParamsPerPad[i][0] = fLambdaFitParams[i];
      fFitParamsPerPad[i][1] = fRadiusFitParams[i];
    }
  }

  else
  {
    assert(fNAnalyses==2 || fNAnalyses==4 || fNAnalyses==6);  //to be safe, for now
    for(int i=0; i<(fNAnalyses/2); i++)
    {
      SetSharedParameter(kRadius, tShares2dVec[i], fRadiusFitParams[i].GetStartValue(),
                         fRadiusFitParams[i].GetLowerBound(), fRadiusFitParams[i].GetUpperBound(), fRadiusFitParams[i].IsFixed());

      fFitParamsPerPad[2*i][1] = fRadiusFitParams[i];
      fFitParamsPerPad[2*i+1][1] = fRadiusFitParams[i];
    }

    if(!fAllShareSingleLambdaParam)
    {
      if(fShareLambdaParams)
      {
        for(int i=0; i<(fNAnalyses/2); i++)
        {
          SetSharedParameter(kLambda, tShares2dVec[i], fLambdaFitParams[i].GetStartValue(),
                             fLambdaFitParams[i].GetLowerBound(), fLambdaFitParams[i].GetUpperBound(), fLambdaFitParams[i].IsFixed());

          fFitParamsPerPad[2*i][0] = fLambdaFitParams[i];
          fFitParamsPerPad[2*i+1][0] = fLambdaFitParams[i];
        }
      }

      else
      {
        for(int i=0; i<fNAnalyses; i++)
        {
          SetParameter(kLambda, i, fLambdaFitParams[i].GetStartValue(),
                       fLambdaFitParams[i].GetLowerBound(), fLambdaFitParams[i].GetUpperBound(), fLambdaFitParams[i].IsFixed());
          fFitParamsPerPad[i][0] = fLambdaFitParams[i];
        }
      }
    }
  }


  for(int i=0; i<fNAnalyses; i++)
  {
    fFitParamsPerPad[i][2] = fScattFitParams[0];
    fFitParamsPerPad[i][3] = fScattFitParams[1];
    fFitParamsPerPad[i][4] = fScattFitParams[2];

    if(fAllShareSingleLambdaParam) fFitParamsPerPad[i][0] = fLambdaFitParams[0];
  }

}

//________________________________________________________________________________________________________________
void FitGenerator::InitializeGenerator(double aMaxFitKStar)
{
/*
  if(fIncludeResidualsType != kIncludeNoResiduals)  //since this involves the CoulombFitter, I should place limits on parameters used in interpolations
  {
    for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++) SetRadiusLimits(1.,15.,iCent);
    SetScattParamLimits({{-10.,10.},{-10.,10.},{-10.,10.}});
  }
*/

  if(fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll)
  {
    for(unsigned int iCent=0; iCent<fCentralityTypes.size(); iCent++) SetRadiusLimits(1.,12.,iCent);
  }

  SetAllParameters();
  fSharedAn->CreateMinuitParameters();

  fLednickyFitter = new LednickyFitter(fSharedAn,aMaxFitKStar);
  fLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN);
  fLednickyFitter->SetApplyMomResCorrection(fApplyMomResCorrection);
  fLednickyFitter->SetApplyNonFlatBackgroundCorrection(fApplyNonFlatBackgroundCorrection);
  fLednickyFitter->SetNonFlatBgdFitType(fNonFlatBgdFitType);
  fLednickyFitter->SetIncludeResidualCorrelationsType(fIncludeResidualsType);
  fLednickyFitter->SetChargedResidualsType(fChargedResidualsType);
  fLednickyFitter->SetResPrimMaxDecayType(fResPrimMaxDecayType);
  fLednickyFitter->SetUsemTScalingOfResidualRadii(fUsemTScalingOfResidualRadii, fmTScalingPowerOfResidualRadii);

  fLednickyFitter->SetSaveLocationBase(fSaveLocationBase, fSaveNameModifier);
}

//________________________________________________________________________________________________________________
void FitGenerator::DoFit(double aMaxFitKStar)
{
  InitializeGenerator(aMaxFitKStar);
  GlobalFitter = fLednickyFitter;
  fLednickyFitter->DoFit();
}

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::GenerateContourPlots(int aNPoints, const vector<double> &aParams, const vector<double> &aErrVals, TString aSaveNameModifier, bool aFixAllOthers, double aMaxFitKStar)
{
  InitializeGenerator(aMaxFitKStar);
  GlobalFitter = fLednickyFitter;
  TCanvas* tReturnCan = fLednickyFitter->GenerateContourPlots(aNPoints, aParams, aErrVals, aSaveNameModifier, aFixAllOthers);
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* FitGenerator::GenerateContourPlots(int aNPoints, CentralityType aCentType, const vector<double> &aErrVals, bool aFixAllOthers, double aMaxFitKStar)
{
  InitializeGenerator(aMaxFitKStar);
  GlobalFitter = fLednickyFitter;
  TCanvas* tReturnCan = fLednickyFitter->GenerateContourPlots(aNPoints, aCentType, aErrVals, aFixAllOthers);
  return tReturnCan;
}

//________________________________________________________________________________________________________________
void FitGenerator::WriteAllFitParameters(ostream &aOut)
{
  for(int iAn=0; iAn<fSharedAn->GetNFitPairAnalysis(); iAn++)
  {
    aOut << "______________________________" << endl;
    aOut << "AnalysisType = " << cAnalysisBaseTags[fSharedAn->GetFitPairAnalysis(iAn)->GetAnalysisType()] << endl;
    aOut << "CentralityType = " << cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(iAn)->GetCentralityType()] << endl << endl;
    fSharedAn->GetFitPairAnalysis(iAn)->WriteFitParameters(aOut);
    aOut << endl;
  }
}

//________________________________________________________________________________________________________________
vector<TString> FitGenerator::GetAllFitParametersTStringVector()
{
  vector<TString> tReturnVec(0);

  for(int iAn=0; iAn<fSharedAn->GetNFitPairAnalysis(); iAn++)
  {
    tReturnVec.push_back("______________________________");
    TString tAnLine = TString("AnalysisType = ") + TString(cAnalysisBaseTags[fSharedAn->GetFitPairAnalysis(iAn)->GetAnalysisType()]);
      tReturnVec.push_back(tAnLine);
    TString tCentLine = TString("CentralityType = ") + TString(cPrettyCentralityTags[fSharedAn->GetFitPairAnalysis(iAn)->GetCentralityType()]);
      tReturnVec.push_back(tCentLine);
    tReturnVec.push_back(TString(""));

    vector<TString> tParamVec = fSharedAn->GetFitPairAnalysis(iAn)->GetFitParametersTStringVector();
    for(unsigned int iPar=0; iPar<tParamVec.size(); iPar++) tReturnVec.push_back(tParamVec[iPar]);
    tReturnVec.push_back(TString(""));
  }

  tReturnVec.push_back("******************************");
  tReturnVec.push_back(TString::Format("Chi2 = %0.3f  NDF = %d  Chi2/NDF = %0.3f",
                                        GetChi2(), GetNDF(), GetChi2()/GetNDF()));
  tReturnVec.push_back(TString(""));

  return tReturnVec;
}

//________________________________________________________________________________________________________________
void FitGenerator::FindGoodInitialValues(bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection)
{
  SetAllParameters();

  fSharedAn->CreateMinuitParameters();

  fLednickyFitter = new LednickyFitter(fSharedAn);
  fLednickyFitter->GetFitSharedAnalyses()->GetMinuitObject()->SetFCN(GlobalFCN);
  fLednickyFitter->SetApplyMomResCorrection(aApplyMomResCorrection);
  fLednickyFitter->SetApplyNonFlatBackgroundCorrection(aApplyNonFlatBackgroundCorrection);
  GlobalFitter = fLednickyFitter;

  vector<double> tValues = fLednickyFitter->FindGoodInitialValues();

  for(unsigned int i=0; i<tValues.size(); i++) cout << "i = " << i << ": " << tValues[i] << endl;
}


//________________________________________________________________________________________________________________
void FitGenerator::SetSaveLocationBase(TString aBase, TString aSaveNameModifier)
{
  fSaveLocationBase=aBase;
  if(!aSaveNameModifier.IsNull()) fSaveNameModifier = aSaveNameModifier;
}

//________________________________________________________________________________________________________________
void FitGenerator::ExistsSaveLocationBase()
{
  if(!fSaveLocationBase.IsNull()) return;

  cout << "In FitGenerator" << endl;
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


