///////////////////////////////////////////////////////////////////////////
// FitPairAnalysis:                                                      //
///////////////////////////////////////////////////////////////////////////

#include "FitPairAnalysis.h"

#ifdef __ROOT__
ClassImp(FitPairAnalysis)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________



//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aAnalysisName, vector<FitPartialAnalysis*> &aFitPartialAnalysisCollection, bool aIncludeSingletAndTriplet) :
  fAnalysisRunType(kTrain),
  fAnalysisName(aAnalysisName),
  fAnalysisDirectoryName(""),
  fFitPartialAnalysisCollection(aFitPartialAnalysisCollection),
  fNFitPartialAnalysis(fFitPartialAnalysisCollection.size()),

  fAnalysisType(fFitPartialAnalysisCollection[0]->GetAnalysisType()),
  fCentralityType(fFitPartialAnalysisCollection[0]->GetCentralityType()),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(nullptr),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),
  fMaxBgdBuild(2.0),
  fNormalizeBgdFitToCf(false),

  fPrimaryFit(nullptr),
  fNonFlatBackground(nullptr),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),
  f2dBgdParameters(0),

  fModelKStarTrueVsRecMixed(nullptr),
  fModelKStarHeavyCfFake(nullptr),
  fModelKStarHeavyCfFakeIdeal(nullptr),
  fModelCfFakeIdealCfFakeRatio(nullptr),

  fTransformMatrices(0),
  fTransformStorageMapping(0),

  fResidualCollection(nullptr),
  fPrimaryWithResiduals(0)

{

  //set fFitPartialAnalysisNumber in each FitPartialAnalysis object
  for(int i=0; i<fNFitPartialAnalysis; i++) {fFitPartialAnalysisCollection[i]->SetFitPartialAnalysisNumber(i);}

  //make sure partial analyses in collection have same pair type (AnalysisType) and centrality (CentralityType)
  for(int i=1; i<fNFitPartialAnalysis; i++)
  {
    assert(fFitPartialAnalysisCollection[i-1]->GetAnalysisType() == fFitPartialAnalysisCollection[i]->GetAnalysisType());
    assert(fFitPartialAnalysisCollection[i-1]->GetCentralityType() == fFitPartialAnalysisCollection[i]->GetCentralityType());
  }

  //Don't need to make sure they have same particle types, because analysis types are same
  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  double tKStarMinNorm = 0.32, tKStarMaxNorm=0.40;
  BuildKStarCfHeavy(tKStarMinNorm, tKStarMaxNorm);

  if( (fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM) && aIncludeSingletAndTriplet)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters(aIncludeSingletAndTriplet);

}



//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNFitPartialAnalysis, TString aDirNameModifier, bool aIncludeSingletAndTriplet) :
  fAnalysisRunType(aRunType),
  fAnalysisName(0),
  fAnalysisDirectoryName(""),
  fFitPartialAnalysisCollection(0),
  fNFitPartialAnalysis(aNFitPartialAnalysis),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(nullptr),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),
  fMaxBgdBuild(2.0),
  fNormalizeBgdFitToCf(false),

  fPrimaryFit(nullptr),
  fNonFlatBackground(nullptr),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),
  f2dBgdParameters(0),

  fModelKStarTrueVsRecMixed(nullptr),
  fModelKStarHeavyCfFake(nullptr),
  fModelKStarHeavyCfFakeIdeal(nullptr),
  fModelCfFakeIdealCfFakeRatio(nullptr),

  fTransformMatrices(0),
  fTransformStorageMapping(0),

  fResidualCollection(nullptr),
  fPrimaryWithResiduals(0)

{
  fAnalysisName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  fAnalysisDirectoryName = aFileLocationBase;
    int tLastSlash = fAnalysisDirectoryName.Last('/');
    int tLength = fAnalysisDirectoryName.Length();
    fAnalysisDirectoryName.Remove(tLastSlash+1, tLength-tLastSlash-1);

  int iStart;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) iStart=0;
  else iStart = 2;

  for(int i=iStart; i<fNFitPartialAnalysis+iStart; i++)
  {
    BFieldType tBFieldType = static_cast<BFieldType>(i);

    TString tFileLocation = aFileLocationBase + cBFieldTags[tBFieldType];
    tFileLocation += ".root";

    TString tFitPartialAnalysisName = fAnalysisName + cBFieldTags[tBFieldType];

    FitPartialAnalysis* tFitPartialAnalysis = new FitPartialAnalysis(tFileLocation, tFitPartialAnalysisName, fAnalysisType, fCentralityType, tBFieldType, fAnalysisRunType, aDirNameModifier, aIncludeSingletAndTriplet);

    fFitPartialAnalysisCollection.push_back(tFitPartialAnalysis);
  } 

  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  double tKStarMinNorm = 0.32, tKStarMaxNorm=0.40;
  BuildKStarCfHeavy(tKStarMinNorm, tKStarMaxNorm);

  if( (fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM) && aIncludeSingletAndTriplet)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters(aIncludeSingletAndTriplet);

}

//________________________________________________________________________________________________________________
FitPairAnalysis::FitPairAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNFitPartialAnalysis, TString aDirNameModifier, bool aIncludeSingletAndTriplet) :
  fAnalysisRunType(aRunType),
  fAnalysisName(0),
  fAnalysisDirectoryName(""),
  fFitPartialAnalysisCollection(0),
  fNFitPartialAnalysis(aNFitPartialAnalysis),

  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitPairAnalysisNumber(-1),

  fParticleTypes(2),

  fKStarCfHeavy(nullptr),

  fMinBgdFit(0.60),
  fMaxBgdFit(0.90),
  fMaxBgdBuild(2.0),
  fNormalizeBgdFitToCf(false),

  fPrimaryFit(nullptr),
  fNonFlatBackground(nullptr),

  fNFitParams(0),
  fNFitParamsToShare(5),  //sharing Lambda, Radius, Ref0, Imf0, d0
  fNFitNormParams(0),
  fFitNormParameters(0),
  fFitParameters(0),
  f2dBgdParameters(0),

  fModelKStarTrueVsRecMixed(nullptr),
  fModelKStarHeavyCfFake(nullptr),
  fModelKStarHeavyCfFakeIdeal(nullptr),
  fModelCfFakeIdealCfFakeRatio(nullptr),

  fTransformMatrices(0),
  fTransformStorageMapping(0),

  fResidualCollection(nullptr),
  fPrimaryWithResiduals(0)

{
  fAnalysisName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  fAnalysisDirectoryName = aFileLocationBase;
    int tLastSlash = fAnalysisDirectoryName.Last('/');
    int tLength = fAnalysisDirectoryName.Length();
    fAnalysisDirectoryName.Remove(tLastSlash+1, tLength-tLastSlash-1);

  int iStart;
  if(fAnalysisRunType==kTrain || fAnalysisRunType==kTrainSys) iStart=0;
  else iStart = 2;

  for(int i=iStart; i<fNFitPartialAnalysis+iStart; i++)
  {
    BFieldType tBFieldType = static_cast<BFieldType>(i);

    TString tFileLocation = aFileLocationBase + cBFieldTags[tBFieldType];
    tFileLocation += ".root";

    TString tFileLocationMC = aFileLocationBaseMC + cBFieldTags[tBFieldType];
    tFileLocationMC += ".root";

    TString tFitPartialAnalysisName = fAnalysisName + cBFieldTags[tBFieldType];

    FitPartialAnalysis* tFitPartialAnalysis = new FitPartialAnalysis(tFileLocation, tFileLocationMC, tFitPartialAnalysisName, fAnalysisType, fCentralityType, tBFieldType, fAnalysisRunType, aDirNameModifier, aIncludeSingletAndTriplet);

    fFitPartialAnalysisCollection.push_back(tFitPartialAnalysis);
  } 

  fParticleTypes = fFitPartialAnalysisCollection[0]->GetParticleTypes();

  double tKStarMinNorm = 0.32, tKStarMaxNorm=0.40;
  BuildKStarCfHeavy(tKStarMinNorm, tKStarMaxNorm);
  RebinKStarCfHeavy(2, tKStarMinNorm, tKStarMaxNorm);
  BuildModelKStarTrueVsRecMixed(2);
//  BuildModelCfFakeIdealCfFakeRatio(tKStarMinNorm, tKStarMaxNorm, 1);

  if( (fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM) && aIncludeSingletAndTriplet)
  {
    fNFitParamsToShare = 8; //sharing Lambda, Radius, Ref0, Imf0, d0, Ref02, Imf02, d02
  }

  ShareFitParameters(aIncludeSingletAndTriplet);

}





//________________________________________________________________________________________________________________
FitPairAnalysis::~FitPairAnalysis()
{
  cout << "FitPairAnalysis object (name: " << fAnalysisName << " ) is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarTrueVsRecMixed(int aRebinFactor)
{
  TString tName = "ModelKStarTrueVsRecMixed_" + TString(cAnalysisBaseTags[fAnalysisType]);

  TH2* tPre = (TH2*)fFitPartialAnalysisCollection[0]->GetModelKStarTrueVsRecMixed();
  fModelKStarTrueVsRecMixed = (TH2*)tPre->Clone(tName);

  for(int i=1; i<fNFitPartialAnalysis; i++)
  {
    TH2* tToAdd = (TH2*)fFitPartialAnalysisCollection[i]->GetModelKStarTrueVsRecMixed();
    fModelKStarTrueVsRecMixed->Add(tToAdd);
  }

  fModelKStarTrueVsRecMixed->Rebin2D(aRebinFactor,aRebinFactor);

}



//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildKStarCfHeavy(double aKStarMinNorm, double aKStarMaxNorm)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    //first, make sure everything is updated
    fFitPartialAnalysisCollection[iAnaly]->BuildKStarCf(aKStarMinNorm, aKStarMaxNorm);

    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetKStarCfLite());
  }

  TString tCfBaseName = "KStarHeavyCf_";
  TString tCfName = tCfBaseName + cAnalysisBaseTags[fAnalysisType] + cCentralityTags[fCentralityType];

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fKStarCfHeavy = new CfHeavy(tCfName, tTitle, tTempCfLiteCollection, aKStarMinNorm, aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::RebinKStarCfHeavy(int aRebinFactor, double aKStarMinNorm, double aKStarMaxNorm)
{
  fKStarCfHeavy->Rebin(aRebinFactor, aKStarMinNorm, aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::DrawKStarCfHeavy(TPad* aPad, int aMarkerColor, TString aOption, int aMarkerStyle)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  TH1* tCfToDraw = fKStarCfHeavy->GetHeavyCf();

  TAxis *xax1 = tCfToDraw->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = tCfToDraw->GetYaxis();
    yax1->SetRangeUser(0.9,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();


  //------------------------------------------------------
  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  tCfToDraw->SetMarkerStyle(aMarkerStyle);
  tCfToDraw->SetMarkerSize(0.50);
  tCfToDraw->SetMarkerColor(aMarkerColor);
  tCfToDraw->SetLineColor(aMarkerColor);
  //tCfToDraw->SetTitle(tTitle);



  //------------------------------------------------------
  TLine *line = new TLine(0,1,1,1);
  line->SetLineColor(14);

  tCfToDraw->Draw(aOption);
  line->Draw();

}


//________________________________________________________________________________________________________________
void FitPairAnalysis::CreateFitFunction(IncludeResidualsType aIncResType, ResPrimMaxDecayType aResPrimMaxDecayType, double aChi2, int aNDF, 
                                        double aKStarMin, double aKStarMax, TString aBaseName)
{
  //Since the partial analyses in a pair analysis share all parameters except for normalization, the combined fit function is a simple
  // rescaling of the singular fit functions

  double tOverallNum=0., tOverallDen=0.;
  double tNumScale=0., tNorm=0., tNormError=0.;
  double tOverallScaleError=0.;
  for(int iPartAn=0; iPartAn<fNFitPartialAnalysis; iPartAn++)
  {
    //Note: Use aApplyNorm=false in following so the LednickyEq from each PartialAnalysis are exactly the same, and the
    //      simply combined/rescaled fit function is valid
    fFitPartialAnalysisCollection[iPartAn]->CreateFitFunction(false, aIncResType, aResPrimMaxDecayType, aChi2, aNDF);

    tNumScale = fFitPartialAnalysisCollection[iPartAn]->GetKStarCfLite()->GetNumScale();
    tNorm = fFitPartialAnalysisCollection[iPartAn]->GetFitNormParameter()->GetFitValue();
    tNormError = fFitPartialAnalysisCollection[iPartAn]->GetFitNormParameter()->GetFitValueError();

    tOverallNum += tNumScale*tNorm;
    tOverallDen += tNumScale;
    tOverallScaleError += tNormError*tNormError;
  }
  double tOverallScale = tOverallNum/tOverallDen;

  //--------------------------------------------------------
  assert(fNFitParams==5);
  TString tName = TString::Format("%s_%s%s", aBaseName.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);

  fPrimaryFit = new TF1(tName, FitPartialAnalysis::LednickyEqWithNorm, aKStarMin, aKStarMax, fNFitParams+1);
  double tParamValue, tParamError;
  for(int iPar=0; iPar<fNFitParams; iPar++)
  {
    ParameterType tParamType = fFitParameters[iPar]->GetType();
    tParamValue = fFitParameters[iPar]->GetFitValue();
    tParamError = fFitParameters[iPar]->GetFitValueError();
    if(tParamType==kLambda && aIncResType != kIncludeNoResiduals)
    {
      tParamValue *= cAnalysisLambdaFactorsArr[aIncResType][aResPrimMaxDecayType][fAnalysisType];
      tParamError *= cAnalysisLambdaFactorsArr[aIncResType][aResPrimMaxDecayType][fAnalysisType];
    }
    fPrimaryFit->SetParameter(iPar,tParamValue);
    fPrimaryFit->SetParError(iPar,tParamError);
  }

  fPrimaryFit->SetParameter(5, tOverallScale);
  fPrimaryFit->SetParError(5, sqrt(tOverallScaleError));



  fPrimaryFit->SetChisquare(aChi2);
  fPrimaryFit->SetNDF(aNDF);

  fPrimaryFit->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");
//  fKStarCfHeavy->GetHeavyCf()->GetListOfFunctions()->Add(fPrimaryFit);
}



//________________________________________________________________________________________________________________
TF1* FitPairAnalysis::GetNonFlatBackground_FitCombinedPartials(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf)
{
  //Fit combined histograms
  //  In case aFitType==kChi2PML, use simply added numerators and denominators
  //  In case aFitType==kChi2, use fKStarCfHeavy, i.e. weighted combination of Cfs

  if(fNonFlatBackground && aNormalizeFitToCf==fNormalizeBgdFitToCf) return fNonFlatBackground;

  cout << "\t Using method FitPairAnalysis::GetNonFlatBackground_FitCombinedPartials" << endl;
  fNormalizeBgdFitToCf = aNormalizeFitToCf;

  if(aFitType==kChi2PML)
  {
    TH1* tNum = fKStarCfHeavy->GetSimplyAddedNumDen("SimplyAddedNum", true);
    TH1* tDen = fKStarCfHeavy->GetSimplyAddedNumDen("SimplyAddedDen", false);

    fNonFlatBackground = FitPartialAnalysis::FitNonFlatBackground(tNum, tDen, fKStarCfHeavy->GetHeavyCfClone(), aBgdFitType, aFitType, aNormalizeFitToCf, 
                                                                  fMinBgdFit, fMaxBgdFit, fMaxBgdBuild, fKStarCfHeavy->GetMinNorm(), fKStarCfHeavy->GetMaxNorm());
  }
  else 
  {
    assert(!aNormalizeFitToCf);
    fNonFlatBackground = FitPartialAnalysis::FitNonFlatBackground(fKStarCfHeavy->GetHeavyCfClone(), aBgdFitType,
                                                                  fMinBgdFit, fMaxBgdFit, fMaxBgdBuild, fKStarCfHeavy->GetMinNorm(), fKStarCfHeavy->GetMaxNorm());
  }

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* FitPairAnalysis::GetNonFlatBackground_CombinePartialFits(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf)
{
  //Combine fits (technically, probably the most correct method, as the fits to the partial analyses are used during fitting)
  //  i.e. fit NonFlatBgd for partial analyses first individually, and then combine with weighted mean

  if(fNonFlatBackground && aNormalizeFitToCf==fNormalizeBgdFitToCf) return fNonFlatBackground;

  cout << "\t Using method FitPairAnalysis::GetNonFlatBackground_CombinePartialFits" << endl;
  fNormalizeBgdFitToCf = aNormalizeFitToCf;

  assert(fNFitPartialAnalysis==2);  // i.e. this only works for train runs with _FemtoPlus and _FemtoMinus
                                    //      NOT with old grid runs with _Bp1, _Bp2, _Bm1, _Bm2, _Bm3
  fFitPartialAnalysisCollection[0]->SetMinMaxBgdFit(fMinBgdFit, fMaxBgdFit);
  fFitPartialAnalysisCollection[0]->SetMaxBgdBuild(fMaxBgdBuild);
  const TF1* tFit1 = (TF1*)fFitPartialAnalysisCollection[0]->GetNonFlatBackground(aBgdFitType, aFitType, aNormalizeFitToCf);
  double tNumScale1 = fFitPartialAnalysisCollection[0]->GetKStarCfLite()->GetNumScale();

  fFitPartialAnalysisCollection[1]->SetMinMaxBgdFit(fMinBgdFit, fMaxBgdFit);
  fFitPartialAnalysisCollection[1]->SetMaxBgdBuild(fMaxBgdBuild);
  const TF1* tFit2 = (TF1*)fFitPartialAnalysisCollection[1]->GetNonFlatBackground(aBgdFitType, aFitType, aNormalizeFitToCf);
  double tNumScale2 = fFitPartialAnalysisCollection[1]->GetKStarCfLite()->GetNumScale();

  TString tReturnName = TString::Format("NonFlatBgdFit%s_%s", cNonFlatBgdFitTypeTags[aBgdFitType], fKStarCfHeavy->GetHeavyCf()->GetName());
  if(aNormalizeFitToCf) tReturnName = TString("Normalized") + tReturnName;

  assert(tFit1->GetNpar() == tFit2->GetNpar());
  int tNParsSingle = tFit1->GetNpar();
  int tNParsTotal = 2*(tNParsSingle+1); //tNParsTotal equals 2*(tNParsSingle+1) (+1 from inclusion of num scales)

  if(!aNormalizeFitToCf)
  {
    if(aBgdFitType==kLinear)
    {
      assert(tNParsTotal==6);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoFitFunctionsLinear, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kQuadratic)
    {
      assert(tNParsTotal==8);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoFitFunctionsQuadratic, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kGaussian)
    {
      assert(tNParsTotal==10);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoFitFunctionsGaussian, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kPolynomial)
    {
      assert(tNParsTotal==16);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoFitFunctionsPolynomial, 0., fMaxBgdBuild, tNParsTotal);
    }
    else assert(0);
  }
  else
  {
    if(aBgdFitType==kLinear)
    {
      assert(tNParsTotal==8);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoNormalizedFitFunctionsLinear, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kQuadratic)
    {
      assert(tNParsTotal==10);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoNormalizedFitFunctionsQuadratic, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kGaussian)
    {
      assert(tNParsTotal==12);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoNormalizedFitFunctionsGaussian, 0., fMaxBgdBuild, tNParsTotal);
    }
    else if(aBgdFitType == kPolynomial)
    {
      assert(tNParsTotal==18);
      fNonFlatBackground = new TF1(tReturnName, BackgroundFitter::AddTwoNormalizedFitFunctionsPolynomial, 0., fMaxBgdBuild, tNParsTotal);
    }
    else assert(0);
  }

  //---------------------------------

  for(int i=0; i<tNParsSingle; i++) fNonFlatBackground->SetParameter(i, tFit1->GetParameter(i));
  fNonFlatBackground->SetParameter(tNParsSingle, tNumScale1);

  for(int i=0; i<tNParsSingle; i++) fNonFlatBackground->SetParameter(i+tNParsSingle+1, tFit2->GetParameter(i));
  fNonFlatBackground->SetParameter(tNParsTotal-1, tNumScale2);

  for(int i=0; i<tNParsTotal; i++) cout << "PairPar[" << i << "] = " << fNonFlatBackground->GetParameter(i) << endl;

//  delete tFit1;  NO! Deleting these deletes fNonFlatBackground from FitPartialAnalysis!!!!!
//  delete tFit2;

  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* FitPairAnalysis::GetNonFlatBackground(NonFlatBgdFitType aBgdFitType, FitType aFitType, bool aNormalizeFitToCf, bool aCombinePartialFits)
{
  if(fNonFlatBackground && aNormalizeFitToCf==fNormalizeBgdFitToCf) return fNonFlatBackground;

  cout << endl << endl;
  cout << "-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**" << endl;
  cout << "Getting NonFlatBackground for pair analysis " << fAnalysisName << endl;

  if(aCombinePartialFits) fNonFlatBackground = GetNonFlatBackground_CombinePartialFits(aBgdFitType, aFitType, aNormalizeFitToCf);
  else fNonFlatBackground = GetNonFlatBackground_FitCombinedPartials(aBgdFitType, aFitType, aNormalizeFitToCf);
  
  return fNonFlatBackground;
}


//________________________________________________________________________________________________________________
TF1* FitPairAnalysis::GetNewNonFlatBackground(NonFlatBgdFitType aBgdFitType, bool aShareAmongstPartials)
{
  TString tFitName = TString::Format("NonFlatBackgroundFit%s_%s%s", cNonFlatBgdFitTypeTags[aBgdFitType], 
                                                                    cAnalysisBaseTags[fAnalysisType],
                                                                    cCentralityTags[fCentralityType]);
  if(aShareAmongstPartials)
  {
    fNonFlatBackground = fFitPartialAnalysisCollection[0]->GetNewNonFlatBackground(aBgdFitType);
    tFitName += TString("_ShareAmongstPartials");
    fNonFlatBackground->SetName(tFitName);
    return fNonFlatBackground;
  }

  assert(fNFitPartialAnalysis==2);  // i.e. this only works for train runs with _FemtoPlus and _FemtoMinus
                                    //      NOT with old grid runs with _Bp1, _Bp2, _Bm1, _Bm2, _Bm3

  double tNumScale1 = fFitPartialAnalysisCollection[0]->GetKStarCfLite()->GetNumScale();
  double tNumScale2 = fFitPartialAnalysisCollection[1]->GetKStarCfLite()->GetNumScale();

  assert(fFitPartialAnalysisCollection[0]->GetBgdParameters().size() == fFitPartialAnalysisCollection[1]->GetBgdParameters().size());
  int tNParsSingle = fFitPartialAnalysisCollection[0]->GetBgdParameters().size();
  int tNParsTotal = 2*(tNParsSingle+1); //tNParsTotal equals 2*(tNParsSingle+1) (+1 from inclusion of num scales)

  switch(aBgdFitType) {
  case kLinear:
    //2 parameters
    //par[0]*x[0] + par[1]
    assert(tNParsTotal==8);
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::AddTwoFitFunctionsLinear, 0., fMaxBgdBuild, tNParsTotal);
    break;

  case kQuadratic:
    //3 parameters
    //par[0]*x[0]*x[0] + par[1]*x[0] + par[2]
    assert(tNParsTotal==8);
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::AddTwoFitFunctionsQuadratic, 0., fMaxBgdBuild, tNParsTotal);
    break;

  case kGaussian:
    //4 parameters (although, likely par[1] fixed to zero
    //par[0]*exp(-0.5*(pow((x[0]-par[1])/par[2],2.0))) + par[3]
    assert(tNParsTotal==10);
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::AddTwoFitFunctionsGaussian, 0., fMaxBgdBuild, tNParsTotal);
    break;

  case kPolynomial:
    //7 parameters
    //par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4) + par[5]*pow(x[0],5) + par[6]*pow(x[0],6);
    assert(tNParsTotal==16);
    fNonFlatBackground = new TF1(tFitName, BackgroundFitter::AddTwoFitFunctionsPolynomial, 0., fMaxBgdBuild, tNParsTotal);
    break;

  default:
    cout << "FitPairAnalysis::GetNewNonFlatBackground: Invalid NonFlatBgdFitType = " << aBgdFitType << " selected" << endl;
    assert(0);
  }

  for(int i=0; i<tNParsSingle; i++) fNonFlatBackground->SetParameter(i, fFitPartialAnalysisCollection[0]->GetBgdParameter(i)->GetFitValue());
  fNonFlatBackground->SetParameter(tNParsSingle, tNumScale1);

  for(int i=0; i<tNParsSingle; i++) fNonFlatBackground->SetParameter(i+tNParsSingle+1, fFitPartialAnalysisCollection[1]->GetBgdParameter(i)->GetFitValue());
  fNonFlatBackground->SetParameter(tNParsTotal-1, tNumScale2);


}

//________________________________________________________________________________________________________________
void FitPairAnalysis::InitializeBackgroundParams(NonFlatBgdFitType aNonFlatBgdType, bool aShareAmongstPartials)
{
  f2dBgdParameters.clear();
  for(int i=0; i<fNFitPartialAnalysis; i++) fFitPartialAnalysisCollection[i]->InitializeBackgroundParams(aNonFlatBgdType);

  vector<FitParameter*> tTempVec(0);
  if(aShareAmongstPartials)
  {
    vector<int> tAllShared (fNFitPartialAnalysis);
    for(int i=0; i<fNFitPartialAnalysis; i++) {tAllShared[i] = i;}
    
    fFitPartialAnalysisCollection[0]->SetBgdParametersSharedLocal(true, tAllShared);
    tTempVec = fFitPartialAnalysisCollection[0]->GetBgdParameters();
    f2dBgdParameters.push_back(tTempVec);

    for(int i=1; i<fNFitPartialAnalysis; i++)
    {
      fFitPartialAnalysisCollection[i]->SetBgdParametersSharedLocal(true, tAllShared);
      fFitPartialAnalysisCollection[i]->SetBgdParametersShallow(tTempVec);
    }
  }
  else
  {
    for(int i=0; i<fNFitPartialAnalysis; i++)
    {
      tTempVec = fFitPartialAnalysisCollection[i]->GetBgdParameters();
      f2dBgdParameters.push_back(tTempVec);
    }
  }
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::SetBgdParametersShallow(vector<FitParameter*> &aBgdParameters)
{
//TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

  for(int i=0; i<fNFitPartialAnalysis; i++) fFitPartialAnalysisCollection[i]->SetBgdParametersShallow(aBgdParameters);
/*
  for(unsigned int i=0; i<f2dBgdParameters.size(); i++)
  {
    assert(f2dBgdParameters[i].size() == aBgdParameters.size());
    for(unsigned int j=0; j<f2dBgdParameters[i].size(); j++) f2dBgdParameters[i][j] = aBgdParameters[j];
  }
*/
}



//________________________________________________________________________________________________________________
void FitPairAnalysis::SetBgdParametersSharedGlobal(bool aIsShared, vector<int> &aSharedAnalyses)
{
  for(int i=0; i<fNFitPartialAnalysis; i++) fFitPartialAnalysisCollection[i]->SetBgdParametersSharedGlobal(aIsShared, aSharedAnalyses);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::CreateFitNormParameters()
{
  fFitNormParameters.clear();

  for(int i=0; i<fNFitPartialAnalysis; i++)
  {
    fFitNormParameters.push_back(fFitPartialAnalysisCollection[i]->GetFitNormParameter());
  }

  fNFitNormParams = fFitNormParameters.size();

  if(fAnalysisRunType==kGrid && fNFitNormParams != 5)
  {
    cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitNormParams != 5 (the typical value)" << endl;
    cout << "Instead, fNFitNormParams = " << fNFitNormParams << endl;
    cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }
  else if(!(fAnalysisRunType==kGrid) && fNFitNormParams != 2)
  {
    cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitNormParams != 2 (the typical value)" << endl;
    cout << "Instead, fNFitNormParams = " << fNFitNormParams << endl;
    cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }



}


//________________________________________________________________________________________________________________
void FitPairAnalysis::ShareFitParameters(bool aIncludeSingletAndTriplet)
{
  fFitParameters.clear();

  CreateFitNormParameters();

  vector<int> tAllShared (fNFitPartialAnalysis);
  for(int i=0; i<fNFitPartialAnalysis; i++) {tAllShared[i] = i;}

  for(int i=0; i<fNFitParamsToShare; i++)
  {
    ParameterType tParamType = static_cast<ParameterType>(i);
    fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType)->SetSharedLocal(true,tAllShared);
  }

  for(int iAnaly=1; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    for(int iPar=0; iPar<fNFitParamsToShare; iPar++)
    {
      ParameterType tParamType = static_cast<ParameterType>(iPar);
      fFitPartialAnalysisCollection[iAnaly]->GetFitParameter(tParamType)->SetSharedLocal(true,tAllShared);
      fFitPartialAnalysisCollection[iAnaly]->SetFitParameterShallow(fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType));
    }
  }

  for(int i=0; i<fNFitParamsToShare; i++)
  {
    ParameterType tParamType = static_cast<ParameterType>(i);
    fFitParameters.push_back(fFitPartialAnalysisCollection[0]->GetFitParameter(tParamType));
  }


  fNFitParams = fFitParameters.size();

  if((fAnalysisType == kXiKchP || fAnalysisType == kAXiKchP || fAnalysisType == kXiKchM || fAnalysisType == kAXiKchM) && aIncludeSingletAndTriplet)
  {
    if(fNFitParams != 8)
    {
      cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitParams != 8 (the typical value)" << endl;
      cout << "Instead, fNFitParams = " << fNFitParams << endl;
      cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
  }

  else
  {
    if(fNFitParams != 5)
    {
      cout << "WARNING:  In FitPairAnalysis (name: " << fAnalysisName << "), fNFitParams != 5 (the typical value)" << endl;
      cout << "Instead, fNFitParams = " << fNFitParams << endl;
      cout << "Is this alright?  Choose (1) for yes, (0) for no" << endl; 
      int tResponse;
      cin >> tResponse;
      assert(tResponse);
    }
  }

}


//________________________________________________________________________________________________________________
void FitPairAnalysis::WriteFitParameters(ostream &aOut)
{
  for(unsigned int i=0; i<fFitParameters.size(); i++)
  {
    aOut << fFitParameters[i]->GetName() << ": " << fFitParameters[i]->GetFitValue() << " +- " << fFitParameters[i]->GetFitValueError() << endl;
  }
}

//________________________________________________________________________________________________________________
vector<TString> FitPairAnalysis::GetFitParametersTStringVector()
{
  vector<TString> tReturnVec(0);
  TString tLine;
  for(unsigned int i=0; i<fFitParameters.size(); i++)
  {
    tLine = TString::Format("%s: %f +- %f",fFitParameters[i]->GetName().Data(),fFitParameters[i]->GetFitValue(),fFitParameters[i]->GetFitValueError());
    tReturnVec.push_back(tLine);
  }
  return tReturnVec;
}

//________________________________________________________________________________________________________________
vector<double> FitPairAnalysis::GetFitParametersVector()
{
  vector<double> tReturnVec(0);
  for(unsigned int i=0; i<fFitParameters.size(); i++)
  {
    tReturnVec.push_back(fFitParameters[i]->GetFitValue());
  }
  return tReturnVec;
}

/*
//________________________________________________________________________________________________________________
void FitPairAnalysis::SetFitParameterShallow(FitParameter* aParam)
{
  //Created a shallow copy, which I think is what I want
  fFitParameters[aParam->GetType()] = aParam;
}
*/

//________________________________________________________________________________________________________________
void FitPairAnalysis::SetFitParameterShallow(FitParameter* aParam)
{
  assert(int(aParam->GetType()) < 8); //I do not want this function to touch the normalizations (kNorm = 8)

  for(int i=0; i<fNFitPartialAnalysis; i++) {fFitPartialAnalysisCollection[i]->SetFitParameterShallow(aParam);}
  fFitParameters[aParam->GetType()] = aParam;
}

//________________________________________________________________________________________________________________
void FitPairAnalysis::SetupAxis(TAxis* aAxis, double aMin, double aMax, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitPairAnalysis::SetupAxis(TAxis* aAxis, TString aTitle, float aTitleSize, float aTitleOffset, bool aCenterTitle, float aLabelSize, float aLabelOffset, int aNdivisions)
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
void FitPairAnalysis::DrawFit(const char* aTitle)
{
  gStyle->SetOptFit();

  gStyle->SetStatH(0.15);
  gStyle->SetStatW(0.30);

  gStyle->SetStatX(0.85);
  gStyle->SetStatY(0.60);

  TAxis *xax = fKStarCfHeavy->GetHeavyCf()->GetXaxis();
  SetupAxis(xax,0.,0.5,"k* (GeV/c)",0.05,0.9,false,0.03,0.005,510);
  TAxis *yax = fKStarCfHeavy->GetHeavyCf()->GetYaxis();
  SetupAxis(yax,0.9,1.04,"C(k*)",0.05,0.9,false,0.03,0.005,510);

  fKStarCfHeavy->GetHeavyCf()->SetTitle(aTitle);
  fKStarCfHeavy->GetHeavyCf()->SetMarkerStyle(20);
  fKStarCfHeavy->GetHeavyCf()->SetMarkerSize(0.5);

  fKStarCfHeavy->GetHeavyCf()->Draw();
  fPrimaryFit->SetLineColor(1);
  fPrimaryFit->Draw("same");

  TLine *line = new TLine(0,1,0.5,1);
  line->SetLineColor(14);
  line->Draw();
/*
  TH1F* tMomResCorrectedFitHisto = GetCorrectedFitHisto(true,false);
  tMomResCorrectedFitHisto->SetLineColor(2);
  tMomResCorrectedFitHisto->Draw("Lsame");
*/
}

//________________________________________________________________________________________________________________
CfHeavy* FitPairAnalysis::GetModelKStarHeavyCf(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetModelKStarCf(aKStarMinNorm,aKStarMaxNorm,aRebin));
  }

  TString tCfBaseName = "ModelKStarHeavyCf_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  CfHeavy* tReturnCfHeavy = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aKStarMinNorm,aKStarMaxNorm);

  return tReturnCfHeavy;
}

//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarHeavyCfFake(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetModelKStarCfFake(aKStarMinNorm,aKStarMaxNorm,aRebin));
  }

  TString tCfBaseName = "ModelKStarHeavyCfFake_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFake = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aKStarMinNorm,aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelKStarHeavyCfFakeIdeal(double aKStarMinNorm, double aKStarMaxNorm, int aRebin)
{
  vector<CfLite*> tTempCfLiteCollection;

  for(int iAnaly=0; iAnaly<fNFitPartialAnalysis; iAnaly++)
  {
    tTempCfLiteCollection.push_back(fFitPartialAnalysisCollection[iAnaly]->GetModelKStarCfFakeIdeal(aKStarMinNorm,aKStarMaxNorm,aRebin));
  }

  TString tCfBaseName = "ModelKStarHeavyCfFakeIdeal_";
  TString tCfName = tCfBaseName + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  TString tTitle = TString(cRootParticleTags[fParticleTypes[0]]) + TString(cRootParticleTags[fParticleTypes[1]]);

  fModelKStarHeavyCfFakeIdeal = new CfHeavy(tCfName,tTitle,tTempCfLiteCollection,aKStarMinNorm,aKStarMaxNorm);
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::BuildModelCfFakeIdealCfFakeRatio(double aKStarMinNorm, double aKStarMaxNorm, int aRebinFactor)
{
  TString tName = "ModelCfFakeIdealCfFakeRatio_" + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);

  BuildModelKStarHeavyCfFake(aKStarMinNorm,aKStarMaxNorm,aRebinFactor);
  BuildModelKStarHeavyCfFakeIdeal(aKStarMinNorm,aKStarMaxNorm,aRebinFactor);

  fModelCfFakeIdealCfFakeRatio = (TH1*)fModelKStarHeavyCfFakeIdeal->GetHeavyCf()->Clone(tName);
  fModelCfFakeIdealCfFakeRatio->SetTitle(tName);
  fModelCfFakeIdealCfFakeRatio->Divide((TH1*)fModelKStarHeavyCfFake->GetHeavyCf());


}


//________________________________________________________________________________________________________________
TH1F* FitPairAnalysis::GetCorrectedFitHisto(bool aMomResCorrection, bool aNonFlatBgdCorrection, bool aIncludeResiduals, NonFlatBgdFitType aNonFlatBgdFitType, FitType aFitType)
{
  int tNbinsX = fKStarCfHeavy->GetHeavyCf()->GetNbinsX();
  double tKStarMin = fKStarCfHeavy->GetHeavyCf()->GetBinLowEdge(1);
  double tKStarMax = fKStarCfHeavy->GetHeavyCf()->GetBinLowEdge(tNbinsX+1);

  TH1F* tUncorrected = new TH1F("tUncorrected","tUncorrected",tNbinsX,tKStarMin,tKStarMax);

  if(aIncludeResiduals)
  {
    assert(tNbinsX==(int)fPrimaryWithResiduals.size());
    for(int i=1; i<=tNbinsX; i++)
    {
      tUncorrected->SetBinContent(i,fPrimaryWithResiduals[i-1]);
      tUncorrected->SetBinError(i,0.);
    }
  }
  else
  {
    for(int i=1; i<=tNbinsX; i++)
    {
      tUncorrected->SetBinContent(i,fPrimaryFit->Eval(tUncorrected->GetBinCenter(i)));
      tUncorrected->SetBinError(i,0.);
    }
  }

  if(aNonFlatBgdCorrection)
  {
    TF1* tNonFlatBgd = FitPairAnalysis::GetNonFlatBackground(aNonFlatBgdFitType, aFitType, true, true);
    for(int i=1; i<=tUncorrected->GetNbinsX(); i++)
    {
      double tVal = tUncorrected->GetBinContent(i)*tNonFlatBgd->Eval(tUncorrected->GetBinCenter(i));
      tUncorrected->SetBinContent(i,tVal);
    }
  }

  TString tName = "CorrectedFitHisto_" + TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]);
  TH1F* tReturnHisto = new TH1F(tName,tName,tNbinsX,tKStarMin,tKStarMax);

  for(int j=1; j<=tUncorrected->GetNbinsX(); j++)
  {
    if(aMomResCorrection)
    {
      double tValue = 0.;
      assert(tUncorrected->GetBinCenter(j) == fModelKStarTrueVsRecMixed->GetYaxis()->GetBinCenter(j));
      for(int i=1; i<=fModelKStarTrueVsRecMixed->GetNbinsX(); i++)
      {
        assert(tUncorrected->GetBinCenter(i) == fModelKStarTrueVsRecMixed->GetXaxis()->GetBinCenter(i));
        assert(tUncorrected->GetBinContent(i) > 0.);
        tValue += tUncorrected->GetBinContent(i)*fModelKStarTrueVsRecMixed->GetBinContent(i,j);
      }
      tValue /= fModelKStarTrueVsRecMixed->Integral(1,fModelKStarTrueVsRecMixed->GetNbinsX(),j,j);
      tReturnHisto->SetBinContent(j,tValue);
      tReturnHisto->SetBinError(j,0.);
    }
    else
    {
      tReturnHisto->SetBinContent(j,tUncorrected->GetBinContent(j));
      tReturnHisto->SetBinError(j,0);
    }
  }

  //Root is stupid and draws a line connected the underflow bin, which is at 0
  tReturnHisto->SetBinContent(0,tReturnHisto->GetBinContent(1));
  tReturnHisto->GetBinError(0,0.);

  delete tUncorrected;
  return tReturnHisto;
}


//________________________________________________________________________________________________________________
void FitPairAnalysis::LoadTransformMatrices(IncludeResidualsType aIncludeResidualsType, int aRebin, TString aFileLocation)
{
  assert(aIncludeResidualsType != kIncludeNoResiduals);
  if(aFileLocation.IsNull()) aFileLocation = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/TransformMatrices_Mix5.root";

  TFile *tFile = TFile::Open(aFileLocation);
  TString tName2 = cAnalysisBaseTags[fAnalysisType] + TString("Transform");

  TString tName1Sig   = TString("SigTo");
  TString tName1XiC   = TString("XiCTo");
  TString tName1Xi0   = TString("Xi0To");
  TString tName1Omega = TString("OmegaTo");

  TString tFullNameSig, tFullNameXiC, tFullNameXi0, tFullNameOmega;
  TString tFullNameSigStP, tFullNameSigStM, tFullNameSigSt0;

  switch(fAnalysisType) {
  case kLamKchP:
  case kLamKchM:
  case kLamK0:
    tFullNameSig = TString("f") + tName1Sig + tName2;
    tFullNameXiC = TString("f") + tName1XiC + tName2;
    tFullNameXi0 = TString("f") + tName1Xi0 + tName2;
    tFullNameOmega = TString("f") + tName1Omega + tName2;

    tFullNameSigStP = TString("fSigStPTo") + tName2;
    tFullNameSigStM = TString("fSigStMTo") + tName2;
    tFullNameSigSt0 = TString("fSigSt0To") + tName2;
    break;

  case kALamKchP:
  case kALamKchM:
  case kALamK0:
    tFullNameSig = TString("fA") + tName1Sig + tName2;
    tFullNameXiC = TString("fA") + tName1XiC + tName2;
    tFullNameXi0 = TString("fA") + tName1Xi0 + tName2;
    tFullNameOmega = TString("fA") + tName1Omega + tName2;

    tFullNameSigStP = TString("fASigStMTo") + tName2;
    tFullNameSigStM = TString("fASigStPTo") + tName2;
    tFullNameSigSt0 = TString("fASigSt0To") + tName2;
    break;

  default:
    cout << "ERROR:  fAnalysisType = " << fAnalysisType << " is not apropriate" << endl << endl;
    assert(0);
  }

  TString tFullNameLamKSt0, tFullNameSigKSt0, tFullNameXiCKSt0, tFullNameXi0KSt0;
  switch(fAnalysisType) {
  case kLamKchP:
  case kLamK0:
    tFullNameLamKSt0 = TString("fLamKSt0To") + tName2;
    tFullNameSigKSt0 = TString("fSigKSt0To") + tName2;
    tFullNameXiCKSt0 = TString("fXiCKSt0To") + tName2;
    tFullNameXi0KSt0 = TString("fXi0KSt0To") + tName2;
    break;

  case kLamKchM:
    tFullNameLamKSt0 = TString("fLamAKSt0To") + tName2;
    tFullNameSigKSt0 = TString("fSigAKSt0To") + tName2;
    tFullNameXiCKSt0 = TString("fXiCAKSt0To") + tName2;
    tFullNameXi0KSt0 = TString("fXi0AKSt0To") + tName2;
    break;

  case kALamKchP:
  case kALamK0:
    tFullNameLamKSt0 = TString("fALamKSt0To") + tName2;
    tFullNameSigKSt0 = TString("fASigKSt0To") + tName2;
    tFullNameXiCKSt0 = TString("fAXiCKSt0To") + tName2;
    tFullNameXi0KSt0 = TString("fAXi0KSt0To") + tName2;
    break;

  case kALamKchM:
    tFullNameLamKSt0 = TString("fALamAKSt0To") + tName2;
    tFullNameSigKSt0 = TString("fASigAKSt0To") + tName2;
    tFullNameXiCKSt0 = TString("fAXiCAKSt0To") + tName2;
    tFullNameXi0KSt0 = TString("fAXi0AKSt0To") + tName2;
    break;

  default:
    cout << "ERROR:  fAnalysisType = " << fAnalysisType << " is not apropriate" << endl << endl;
    assert(0);
  }


  TH2D* tSig = (TH2D*)tFile->Get(tFullNameSig);
    tSig->SetDirectory(0);
    tSig->Rebin2D(aRebin,aRebin);
  TH2D* tXiC = (TH2D*)tFile->Get(tFullNameXiC);
    tXiC->SetDirectory(0);
    tXiC->Rebin2D(aRebin,aRebin);
  TH2D* tXi0 = (TH2D*)tFile->Get(tFullNameXi0);
    tXi0->SetDirectory(0);
    tXi0->Rebin2D(aRebin,aRebin);
  TH2D* tOmega = (TH2D*)tFile->Get(tFullNameOmega);
    tOmega->SetDirectory(0);
    tOmega->Rebin2D(aRebin,aRebin);

  TH2D* tSigStP = (TH2D*)tFile->Get(tFullNameSigStP);
    tSigStP->SetDirectory(0);
    tSigStP->Rebin2D(aRebin,aRebin);
  TH2D* tSigStM = (TH2D*)tFile->Get(tFullNameSigStM);
    tSigStM->SetDirectory(0);
    tSigStM->Rebin2D(aRebin,aRebin);
  TH2D* tSigSt0 = (TH2D*)tFile->Get(tFullNameSigSt0);
    tSigSt0->SetDirectory(0);
    tSigSt0->Rebin2D(aRebin,aRebin);

  TH2D* tLamKSt0 = (TH2D*)tFile->Get(tFullNameLamKSt0);
    tLamKSt0->SetDirectory(0);
    tLamKSt0->Rebin2D(aRebin,aRebin);
  TH2D* tSigKSt0 = (TH2D*)tFile->Get(tFullNameSigKSt0);
    tSigKSt0->SetDirectory(0);
    tSigKSt0->Rebin2D(aRebin,aRebin);
  TH2D* tXiCKSt0 = (TH2D*)tFile->Get(tFullNameXiCKSt0);
    tXiCKSt0->SetDirectory(0);
    tXiCKSt0->Rebin2D(aRebin,aRebin);
  TH2D* tXi0KSt0 = (TH2D*)tFile->Get(tFullNameXi0KSt0);
    tXi0KSt0->SetDirectory(0);
    tXi0KSt0->Rebin2D(aRebin,aRebin);

  fTransformMatrices.clear();
  fTransformMatrices.push_back((TH2D*)tSig);
  fTransformMatrices.push_back((TH2D*)tXiC);
  fTransformMatrices.push_back((TH2D*)tXi0);
  fTransformMatrices.push_back((TH2D*)tOmega);

  fTransformMatrices.push_back((TH2D*)tSigStP);
  fTransformMatrices.push_back((TH2D*)tSigStM);
  fTransformMatrices.push_back((TH2D*)tSigSt0);

  fTransformMatrices.push_back((TH2D*)tLamKSt0);
  fTransformMatrices.push_back((TH2D*)tSigKSt0);
  fTransformMatrices.push_back((TH2D*)tXiCKSt0);
  fTransformMatrices.push_back((TH2D*)tXi0KSt0);

  //-----------Build mapping vector------------------------
  fTransformStorageMapping.clear();
  switch(fAnalysisType) {
  case kLamKchP:
    fTransformStorageMapping.push_back(kResSig0KchP);
    fTransformStorageMapping.push_back(kResXiCKchP);
    fTransformStorageMapping.push_back(kResXi0KchP);
    fTransformStorageMapping.push_back(kResOmegaKchP);
    fTransformStorageMapping.push_back(kResSigStPKchP);
    fTransformStorageMapping.push_back(kResSigStMKchP);
    fTransformStorageMapping.push_back(kResSigSt0KchP);
    fTransformStorageMapping.push_back(kResLamKSt0);
    fTransformStorageMapping.push_back(kResSig0KSt0);
    fTransformStorageMapping.push_back(kResXiCKSt0);
    fTransformStorageMapping.push_back(kResXi0KSt0);
    break;

  case kLamKchM:
    fTransformStorageMapping.push_back(kResSig0KchM);
    fTransformStorageMapping.push_back(kResXiCKchM);
    fTransformStorageMapping.push_back(kResXi0KchM);
    fTransformStorageMapping.push_back(kResOmegaKchM);
    fTransformStorageMapping.push_back(kResSigStPKchM);
    fTransformStorageMapping.push_back(kResSigStMKchM);
    fTransformStorageMapping.push_back(kResSigSt0KchM);
    fTransformStorageMapping.push_back(kResLamAKSt0);
    fTransformStorageMapping.push_back(kResSig0AKSt0);
    fTransformStorageMapping.push_back(kResXiCAKSt0);
    fTransformStorageMapping.push_back(kResXi0AKSt0);
    break;

  case kALamKchP:
    fTransformStorageMapping.push_back(kResASig0KchP);
    fTransformStorageMapping.push_back(kResAXiCKchP);
    fTransformStorageMapping.push_back(kResAXi0KchP);
    fTransformStorageMapping.push_back(kResAOmegaKchP);
    fTransformStorageMapping.push_back(kResASigStMKchP);
    fTransformStorageMapping.push_back(kResASigStPKchP);
    fTransformStorageMapping.push_back(kResASigSt0KchP);
    fTransformStorageMapping.push_back(kResALamKSt0);
    fTransformStorageMapping.push_back(kResASig0KSt0);
    fTransformStorageMapping.push_back(kResAXiCKSt0);
    fTransformStorageMapping.push_back(kResAXi0KSt0);
    break;

  case kALamKchM:
    fTransformStorageMapping.push_back(kResASig0KchM);
    fTransformStorageMapping.push_back(kResAXiCKchM);
    fTransformStorageMapping.push_back(kResAXi0KchM);
    fTransformStorageMapping.push_back(kResAOmegaKchM);
    fTransformStorageMapping.push_back(kResASigStMKchM);
    fTransformStorageMapping.push_back(kResASigStPKchM);
    fTransformStorageMapping.push_back(kResASigSt0KchM);
    fTransformStorageMapping.push_back(kResALamAKSt0);
    fTransformStorageMapping.push_back(kResASig0AKSt0);
    fTransformStorageMapping.push_back(kResAXiCAKSt0);
    fTransformStorageMapping.push_back(kResAXi0AKSt0);
    break;

  case kLamK0:
    fTransformStorageMapping.push_back(kResSig0K0);
    fTransformStorageMapping.push_back(kResXiCK0);
    fTransformStorageMapping.push_back(kResXi0K0);
    fTransformStorageMapping.push_back(kResOmegaK0);
    fTransformStorageMapping.push_back(kResSigStPK0);
    fTransformStorageMapping.push_back(kResSigStMK0);
    fTransformStorageMapping.push_back(kResSigSt0K0);
    fTransformStorageMapping.push_back(kResLamKSt0ToLamK0);
    fTransformStorageMapping.push_back(kResSig0KSt0ToLamK0);
    fTransformStorageMapping.push_back(kResXiCKSt0ToLamK0);
    fTransformStorageMapping.push_back(kResXi0KSt0ToLamK0);
    break;

  case kALamK0:
    fTransformStorageMapping.push_back(kResASig0K0);
    fTransformStorageMapping.push_back(kResAXiCK0);
    fTransformStorageMapping.push_back(kResAXi0K0);
    fTransformStorageMapping.push_back(kResAOmegaK0);
    fTransformStorageMapping.push_back(kResASigStMK0);
    fTransformStorageMapping.push_back(kResASigStPK0);
    fTransformStorageMapping.push_back(kResASigSt0K0);
    fTransformStorageMapping.push_back(kResALamKSt0ToALamK0);
    fTransformStorageMapping.push_back(kResASig0KSt0ToALamK0);
    fTransformStorageMapping.push_back(kResAXiCKSt0ToALamK0);
    fTransformStorageMapping.push_back(kResAXi0KSt0ToALamK0);
    break;

  default:
    cout << "ERROR:  fAnalysisType = " << fAnalysisType << " is not apropriate" << endl << endl;
    assert(0);
  }

  if(aIncludeResidualsType==kInclude3Residuals)
  {
    fTransformMatrices.resize(3);
    fTransformStorageMapping.resize(3);
  }

}

//________________________________________________________________________________________________________________
vector<TH2D*> FitPairAnalysis::GetTransformMatrices(IncludeResidualsType aIncludeResidualsType, int aRebin, TString aFileLocation)
{
  if(fTransformMatrices.size()==0) LoadTransformMatrices(aIncludeResidualsType, aRebin, aFileLocation);
  return fTransformMatrices;
}

//________________________________________________________________________________________________________________
TH2D* FitPairAnalysis::GetTransformMatrix(IncludeResidualsType aIncludeResidualsType, int aIndex, int aRebin, TString aFileLocation)
{
  if(fTransformMatrices.size()==0) LoadTransformMatrices(aIncludeResidualsType, aRebin, aFileLocation);
  return fTransformMatrices[aIndex];
}

//________________________________________________________________________________________________________________
TH2D* FitPairAnalysis::GetTransformMatrix(IncludeResidualsType aIncludeResidualsType, AnalysisType aResidualType, int aRebin, TString aFileLocation)
{
  if(fTransformMatrices.size()==0) LoadTransformMatrices(aIncludeResidualsType, aRebin, aFileLocation);
  int tIndex = -1;
  for(int i=0; i<(int)fTransformStorageMapping.size(); i++)
  {
    if(aResidualType == fTransformStorageMapping[i]) tIndex = i;
  }
  assert(tIndex > -1);
  return fTransformMatrices[tIndex];
}


//________________________________________________________________________________________________________________
TH1* FitPairAnalysis::GetCfwSysErrors()
{
  TString tDate = fAnalysisDirectoryName;
    int tLastUnderline = tDate.Last('_');
    tDate.Remove(0,tLastUnderline);
    tDate.Remove(tDate.Length()-1);

  TString tFileLocation = TString::Format("%sSystematicResults_%s%s%s.root",fAnalysisDirectoryName.Data(),cAnalysisBaseTags[fAnalysisType],cCentralityTags[fCentralityType],tDate.Data());
  TString tHistName = TString(cAnalysisBaseTags[fAnalysisType]) + TString(cCentralityTags[fCentralityType]) + TString("_wSysErrors");

  TFile tFile(tFileLocation);
  TH1* tReturnHist = (TH1*)tFile.Get(tHistName);
    tReturnHist->SetDirectory(0);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
td1dVec FitPairAnalysis::GetCorrectedFitVec()
{
  double tScale = 0.;
  double tTempScale = 0.;

  td1dVec tReturnVec = fFitPartialAnalysisCollection[0]->GetCorrectedFitVec();
  tTempScale = fFitPartialAnalysisCollection[0]->GetKStarNumScale();
  tScale += tTempScale;
  for(unsigned int j=0; j<tReturnVec.size(); j++) tReturnVec[j] *= tTempScale;

  for(int i=1; i<fNFitPartialAnalysis; i++)
  {
    tTempScale = fFitPartialAnalysisCollection[i]->GetKStarNumScale();
    tScale += tTempScale;
    assert(tReturnVec.size() == fFitPartialAnalysisCollection[i]->GetCorrectedFitVec().size());
    for(unsigned int j=0; j<tReturnVec.size(); j++)
    {
      tReturnVec[j] = tReturnVec[j] + tTempScale*fFitPartialAnalysisCollection[i]->GetCorrectedFitVec()[j];
    }
  }

  for(unsigned int j=0; j<tReturnVec.size(); j++) tReturnVec[j] /= tScale;

  return tReturnVec;
}


//________________________________________________________________________________________________________________
TH1F* FitPairAnalysis::GetCorrectedFitHistv2(double aMaxDrawKStar)
{
  int tNbinsXToFit = fKStarCfHeavy->GetHeavyCf()->FindBin(aMaxDrawKStar-0.0000001);  //-0.0000001 ensures we don't overshoot our desired bin, since xup excluded in TH1
                                                                                     // i.e., if given aMaxDrawKStar=1.0 (with binsize=0.1), without subtraction, this 
                                                                                     // would return 101 instead of 100
  double tKStarMin = fKStarCfHeavy->GetHeavyCf()->GetBinLowEdge(1);
  double tKStarMax = fKStarCfHeavy->GetHeavyCf()->GetBinLowEdge(tNbinsXToFit+1);

  td1dVec tCorrectedFitVec = GetCorrectedFitVec();

  TString tTitle = "testCorrectedFitHist";
  tTitle =+ TString(cAnalysisBaseTags[fAnalysisType]);
  tTitle =+ TString(cAnalysisBaseTags[fCentralityType]);
  TH1F* tCorrectedFitHist = new TH1F(tTitle,tTitle,tNbinsXToFit,tKStarMin,tKStarMax);

//TODO make sure bins (and bin widths) match up correctly
  for(int i=1; i<=tNbinsXToFit; i++)
  {
    tCorrectedFitHist->SetBinContent(i, tCorrectedFitVec[i-1]);
    tCorrectedFitHist->SetBinError(i,0.);
  }

  return tCorrectedFitHist;
}



//________________________________________________________________________________________________________________
void FitPairAnalysis::InitiateResidualCollection(td1dVec &aKStarBinCenters, IncludeResidualsType aIncludeResidualsType, ChargedResidualsType aChargedResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TString aInterpCfsDirectory)
{
  vector<TH2D*> aTransformMatrices = GetTransformMatrices(aIncludeResidualsType);
  vector<AnalysisType> aTransformStorageMapping = GetTransformStorageMapping();
  fResidualCollection = new ResidualCollection(fAnalysisType, aIncludeResidualsType, aChargedResidualsType, aResPrimMaxDecayType, aKStarBinCenters, aTransformMatrices, aTransformStorageMapping, fCentralityType);

  double tSigStRadiusFactor = 1.;
  fResidualCollection->SetRadiusFactorForSigStResiduals(tSigStRadiusFactor);
}

