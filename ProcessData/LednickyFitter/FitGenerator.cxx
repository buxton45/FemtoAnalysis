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
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, const vector<CentralityType> &aCentralityTypes, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier, bool aUseStavCf) :
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
  fLednickyFitter(0),

  fSaveFileType("eps")

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
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier, aUseStavCf));
      break;

    case kConjPair:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier, aUseStavCf));
      break;

    case kPairwConj:
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier, aUseStavCf));
      tVecOfPairAn.push_back(new FitPairAnalysis(aFileLocationBase,aFileLocationBaseMC,fConjPairType,fCentralityTypes[i],aRunType,aNPartialAnalysis,aDirNameModifier, aUseStavCf));
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
FitGenerator::FitGenerator(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam, TString aDirNameModifier, bool aUseStavCf) :
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
  fLednickyFitter(0),

  fSaveFileType("eps")

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

  *this = FitGenerator(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, tCentralityTypes, aRunType, aNPartialAnalysis, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam, aDirNameModifier, aUseStavCf);
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
    fFitParamsPerPad[i].reserve(8);  //big enough for case of singlet or triplet
    fFitParamsPerPad[i].emplace_back(kLambda,0.);
    fFitParamsPerPad[i].emplace_back(kRadius,0.);
    fFitParamsPerPad[i].emplace_back(kRef0,0.);
    fFitParamsPerPad[i].emplace_back(kImf0,0.);
    fFitParamsPerPad[i].emplace_back(kd0,0.);
  }
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


