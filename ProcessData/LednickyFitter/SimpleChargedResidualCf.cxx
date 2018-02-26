/* SimpleChargedResidualCf.cxx */

#include "SimpleChargedResidualCf.h"
#include "FitPairAnalysis.h"

#ifdef __ROOT__
ClassImp(SimpleChargedResidualCf)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
SimpleChargedResidualCf::SimpleChargedResidualCf(AnalysisType aResidualType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters, CentralityType aCentType, TString aFileLocationBase) :
  fResidualType(aResidualType),
  fCentralityType(aCentType),
  fIncludeResidualsType(aIncludeResidualsType),
  fResPrimMaxDecayType(aResPrimMaxDecayType),
  fDaughterType1(kPDGNull),
  fMotherType1(kPDGNull),
  fDaughterType2(kPDGNull),
  fMotherType2(kPDGNull),
  fPairAn(nullptr),  //TODO can I delete this after fExpXiHist is built?
  fExpXiHist(nullptr),
//  fLambdaFactor(cAnalysisLambdaFactors[fResidualType]),
  fLambdaFactor(cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fResidualType]),
  fTransformMatrix(aTransformMatrix),
  fKStarBinCenters(aKStarBinCenters),
  fResCf(0),
  fTransformedResCf(0),
  fUseCoulombOnlyInterpCfs(false),
  f2dCoulombOnlyInterpCfs(nullptr),
  fCurrentRadiusParam(-1.),
  fRadiusFactor(1.)
{
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsY());
  
  SetDaughtersAndMothers();

  omp_set_num_threads(3);
}



//________________________________________________________________________________________________________________
SimpleChargedResidualCf::~SimpleChargedResidualCf()
{}



//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle)
{
  assert(aCfVec.size() == aKStarBinCenters.size());

  double tBinWidth = aKStarBinCenters[1]-aKStarBinCenters[0];
  int tNbins = aKStarBinCenters.size();
  double tKStarMin = aKStarBinCenters[0]-tBinWidth/2.0;
  tKStarMin=0.;
  double tKStarMax = aKStarBinCenters[tNbins-1] + tBinWidth/2.0;

  TH1D* tReturnHist = new TH1D(aTitle, aTitle, tNbins, tKStarMin, tKStarMax);
  for(int i=0; i<tNbins; i++) {tReturnHist->SetBinContent(i+1,aCfVec[i]); tReturnHist->SetBinError(i+1,0.00000000001);}
  //NOTE: Set errors to very small, because if set to zero, just drawing histogram points seems to not work with CanvasPartition package

  return tReturnHist;
}

//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::ConvertHistTo1dVec(TH1* aHist)
{
  td1dVec tReturnVec(0);
  for(int i=1; i<=aHist->GetNbinsX(); i++) tReturnVec.push_back(aHist->GetBinContent(i));
  return tReturnVec;
}

//________________________________________________________________________________________________________________
void SimpleChargedResidualCf::SetDaughtersAndMothers()
{
  vector<ParticlePDGType> tDaughtersAndMothers = GetResidualDaughtersAndMothers(fResidualType);
  fMotherType1 = tDaughtersAndMothers[0];
  fDaughterType1 = tDaughtersAndMothers[1];
  fMotherType2 = tDaughtersAndMothers[2];
  fDaughterType2 = tDaughtersAndMothers[3];
}

//________________________________________________________________________________________________________________
void SimpleChargedResidualCf::BuildExpXiHist(TString aFileLocationBase)
{
  fUseCoulombOnlyInterpCfs = false;

  AnalysisType tAnType;
  AnalysisRunType tRunType=kTrain;
  int tNFitPartialAnalysis=2;

  switch(fResidualType) {
  case kResXiCKchP:
  case kResOmegaKchP:
  case kResSigStPKchM:
  case kResSigStMKchP:
    tAnType = kXiKchP;  //attractive
    break;

  case kResAXiCKchM:
  case kResAOmegaKchM:
  case kResASigStMKchP:
  case kResASigStPKchM:
    tAnType = kAXiKchM;  //attractive
    break;

  case kResXiCKchM:
  case kResOmegaKchM:
  case kResSigStPKchP:
  case kResSigStMKchM:
    tAnType = kXiKchM;  //repulsive
    break;

  case kResAXiCKchP:
  case kResAOmegaKchP:
  case kResASigStMKchM:
  case kResASigStPKchP:
    tAnType = kAXiKchP; //repulsive
    break;

  default:
    cout << "ERROR: SimpleChargedResidualCf::SimpleChargedResidualCf:  fResidualType = " << fResidualType << " is not appropriate" << endl << endl;
    assert(0);
  }
  //----------------------------------------------------------------------------
  cout << "Building SimpleChargedResidualCf object" << endl;
  cout << "\tResidualType   = " << cAnalysisBaseTags[fResidualType] << endl;
  cout << "\tCentralityType = " << cPrettyCentralityTags[fCentralityType] << endl;
  cout << "\tLambdaFactor   = " << fLambdaFactor << endl;
  cout << "\tUsing experimental data from " << cAnalysisBaseTags[tAnType] << " analysis" << endl << endl;

  //-----------------------

  TString tName = TString("ExpXiHist_") + TString(cAnalysisBaseTags[fResidualType]) + TString(cCentralityTags[fCentralityType]);

  //TODO make more general
  fPairAn = new FitPairAnalysis(aFileLocationBase,tAnType,fCentralityType,tRunType,tNFitPartialAnalysis);
  fPairAn->RebinKStarCfHeavy(2,0.32,0.4);
  fExpXiHist = (TH1D*)fPairAn->GetKStarCfHeavy()->GetHeavyCf();
    fExpXiHist->SetName(tName);
    fExpXiHist->SetTitle(tName);
    fExpXiHist->SetDirectory(0);
  assert(fExpXiHist->GetXaxis()->GetBinWidth(1)==0.01);  //TODO make general
  assert(fExpXiHist->GetNbinsX()==100);
}

//________________________________________________________________________________________________________________
void SimpleChargedResidualCf::LoadCoulombOnlyInterpCfs(TString aFileDirectory, double aRadiusFactor)
{
  fRadiusFactor = aRadiusFactor;
  fUseCoulombOnlyInterpCfs = true;

  TString aFileName = aFileDirectory + TString::Format("2dCoulombOnlyInterpCfs_%s.root", cAnalysisBaseTags[fResidualType]);
  TFile aFile(aFileName);
  TH2D* t2dCoulombOnlyInterpCfs = (TH2D*)aFile.Get(TString::Format("t2dCoulombOnlyInterpCfs_%s", cAnalysisBaseTags[fResidualType]));
  assert(t2dCoulombOnlyInterpCfs);
  f2dCoulombOnlyInterpCfs = (TH2D*)t2dCoulombOnlyInterpCfs->Clone();
  f2dCoulombOnlyInterpCfs->SetDirectory(0);

  //----------------------------------------------------------------------------
  cout << "Building SimpleChargedResidualCf object" << endl;
  cout << "\tResidualType   = " << cAnalysisBaseTags[fResidualType] << endl;
  cout << "\tCentralityType = " << cPrettyCentralityTags[fCentralityType] << endl;
  cout << "\tLambdaFactor   = " << fLambdaFactor << endl;
  cout << "\tUsing CoulombOnlyInterpCfs from " << aFileName << endl << endl;
}

//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::ExtractCfFrom2dInterpCfs(double aRadius)
{
  assert(aRadius>0.);
  double tRadius = fRadiusFactor*aRadius;
  assert(f2dCoulombOnlyInterpCfs);  //TODO does this really check that 2dhist is loaded?
/*
  int tBinR = f2dCoulombOnlyInterpCfs->GetYaxis()->FindBin(tRadius);
  TH1D* tTempCf = f2dCoulombOnlyInterpCfs->ProjectionX("tTempCf", tBinR, tBinR);
  td1dVec tReturnVec = ConvertHistTo1dVec(tTempCf);
  return tReturnVec;
*/

  assert(f2dCoulombOnlyInterpCfs->GetNbinsX() >= fKStarBinCenters.size());
  assert( abs(f2dCoulombOnlyInterpCfs->GetXaxis()->GetBinWidth(1) - (fKStarBinCenters[1]-fKStarBinCenters[0])) < 0.0000000001 );

  td1dVec tReturnVec(0);
  double tCfValue = -1.;
  for(unsigned int i=0; i<fKStarBinCenters.size(); i++)
  {
    tCfValue = f2dCoulombOnlyInterpCfs->Interpolate(fKStarBinCenters[i], tRadius);
    assert(tCfValue>0); //This should hopefully kill the program if I go outside of interpolation region
    tReturnVec.push_back(tCfValue);
  }
  return tReturnVec;
}

//________________________________________________________________________________________________________________
bool SimpleChargedResidualCf::IsRadiusParamSame(double aRadiusParam)
{
  if(aRadiusParam==fCurrentRadiusParam) return true;
  else if(abs(aRadiusParam-fCurrentRadiusParam) < std::numeric_limits< double >::min()) return true;  //To be safe, not always precisely equal
  else
  {
    fCurrentRadiusParam = aRadiusParam;
    return false;
  }
}

//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::GetChargedResidualCorrelation(double aRadiusParam)
{
  //If radius parameter is unchanged from previous call, simply return fResCf
  //If radius is different, IsRadiusParamSame will change fCurrentRadiusParam to
  // new value, and complete the calculation
//  bool tIsRadiusSame = IsRadiusParamSame(aRadiusParam);
//  if(tIsRadiusSame) return fResCf;

  if(fResidualType==kResOmegaKchP  || fResidualType==kResAOmegaKchM  ||  //TODO really need to eliminate Omega stuff everywhere
     fResidualType==kResOmegaKchM  || fResidualType==kResAOmegaKchP)
  {
    fResCf = vector<double>(fKStarBinCenters.size(), 0.);
    return fResCf;
  }

  if(fUseCoulombOnlyInterpCfs) fResCf = ExtractCfFrom2dInterpCfs(aRadiusParam);
  else
  {
    double aMaxKStar = fKStarBinCenters[fKStarBinCenters.size()-1] + (fKStarBinCenters[1]-fKStarBinCenters[0])/2;
    int tNbins = std::round(aMaxKStar/fExpXiHist->GetXaxis()->GetBinWidth(1));
    if(fResCf.size()!=tNbins)
    {
      fResCf.clear();
      fResCf.resize(tNbins,0.);
      for(int i=0; i<tNbins; i++) fResCf[i] = fExpXiHist->GetBinContent(i+1);

//    delete tPairAn;
    }
  }
  return fResCf;
}


//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::GetTransformedChargedResidualCorrelation(double aRadiusParam)
{
  omp_set_num_threads(6);

  if(fUseCoulombOnlyInterpCfs || fTransformedResCf.size()==0)
  {
    td1dVec tResCf = GetChargedResidualCorrelation(aRadiusParam);

    unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
    double tDaughterPairKStar, tParentPairKStar;
    assert(tResCf.size() == fKStarBinCenters.size());
    assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsX());
    assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsY());

    fTransformedResCf.clear();
    fTransformedResCf.resize(tResCf.size(),0.);
    td1dVec tNormVec(tResCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

    for(unsigned int i=0; i<tResCf.size(); i++)
    {
      tDaughterPairKStar = fKStarBinCenters[i];
      tDaughterPairKStarBin = fTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

      for(unsigned int j=0; j<tResCf.size(); j++)
      {
        tParentPairKStar = fKStarBinCenters[j];
        tParentPairKStarBin = fTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

        fTransformedResCf[i] += tResCf[j]*fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
        tNormVec[i] += fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      }
      fTransformedResCf[i] /= tNormVec[i];


/*
    double tTransformedResCf_i=0., tNormVec_i=0.;
    #pragma omp parallel for reduction(+: tTransformedResCf_i, tNormVec_i) private(tParentPairKStar, tParentPairKStarBin)
    for(unsigned int j=0; j<tResCf.size(); j++)
    {
      tParentPairKStar = fKStarBinCenters[j];
      tParentPairKStarBin = fTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tTransformedResCf_i += tResCf[j]*fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec_i += fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    tNormVec[i] = tNormVec_i;
    fTransformedResCf[i] = tTransformedResCf_i/tNormVec_i;
*/

    }
  }
  return fTransformedResCf;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam)
{
  td1dVec tResCf = GetChargedResidualCorrelation(aRadiusParam);
  TH1D* tReturnHist = Convert1dVecToHist(tResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetTransformedChargedResidualCorrelationHistogram(TString aTitle, double aRadiusParam)
{
  td1dVec tTransResCf = GetTransformedChargedResidualCorrelation(aRadiusParam);
  TH1D* tReturnHist = Convert1dVecToHist(tTransResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::GetContributionToFitCf(double *aParamsOverall)
{
  double tOverallLambda = aParamsOverall[0];
  double tRadiusParam = aParamsOverall[1];

  td1dVec tReturnVec = GetTransformedChargedResidualCorrelation(tRadiusParam);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] = fLambdaFactor*tOverallLambda*(tReturnVec[i]-1.);
  return tReturnVec;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall)
{
  double tOverallLambda = aParamsOverall[0];
  double tRadiusParam = aParamsOverall[1];

  td1dVec tReturnVec = GetChargedResidualCorrelation(tRadiusParam);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] = 1. + fLambdaFactor*tOverallLambda*(tReturnVec[i]-1.);  //TODO is this right?
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetTransformedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall)
{
  td1dVec tReturnVec = GetContributionToFitCf(aParamsOverall);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] += 1.;
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetResidualCorrelationHistogramWithLambdaAndNormApplied(TString aTitle, double *aParamsOverall, double aNorm)
{
  TH1D* tReturnHist = GetResidualCorrelationHistogramWithLambdaApplied(aTitle, aParamsOverall);
  tReturnHist->Scale(aNorm);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(TString aTitle, double *aParamsOverall, double aNorm)
{
/*
  TH1D* tReturnHist = GetTransformedResidualCorrelationHistogramWithLambdaApplied(aTitle, aParamsOverall);
  tReturnHist->Scale(aNorm);
  return tReturnHist;
*/

  //Want this to return 1 + Norm*Lambda*C, not Norm*(1+Lambda*C)
  // so that, by eye, 1-ReturnValue = contribution to fit Cf

  td1dVec tContributionVec = GetContributionToFitCf(aParamsOverall);
  td1dVec tReturnVec(tContributionVec.size());
  for(unsigned int i=0; i<tContributionVec.size(); i++) tReturnVec[i] = 1.0 + aNorm*tContributionVec[i];
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
void SimpleChargedResidualCf::SetUsemTScalingOfRadii(AnalysisType aParentAnType, double aPower)
{
  double tmTFactorParentPair = cmTFactorsFromTherminator[aParentAnType];
  double tmTFactorResPair = cmTFactorsFromTherminator[fResidualType];
  double tRatio = tmTFactorResPair/tmTFactorParentPair;

  double tScaleFactor = pow(tRatio, aPower);
  SetRadiusFactor(tScaleFactor);

  cout << "Implementing SetUsemTScalingOfRadii for ResidualType = " << cAnalysisBaseTags[fResidualType];
  cout << " of parent AnalysisType = " << cAnalysisBaseTags[aParentAnType] << endl;
  cout << "\taPower       = " << aPower << endl;
  cout << "\tRadiusFactor = " << fRadiusFactor << endl << endl;
}




