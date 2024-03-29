/* NeutralResidualCf.cxx */

#include "NeutralResidualCf.h"

#ifdef __ROOT__
ClassImp(NeutralResidualCf)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
NeutralResidualCf::NeutralResidualCf(AnalysisType aResidualType, IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters) :
  fResidualType(aResidualType),
  fIncludeResidualsType(aIncludeResidualsType),
  fResPrimMaxDecayType(aResPrimMaxDecayType),
  fDaughterType1(kPDGNull),
  fMotherType1(kPDGNull),
  fDaughterType2(kPDGNull),
  fMotherType2(kPDGNull),
//  fLambdaFactor(cAnalysisLambdaFactors[fResidualType]),
  fLambdaFactor(cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fResidualType]),
  fTransformMatrix(aTransformMatrix),
  fKStarBinCenters(aKStarBinCenters),
  fResCf(0),
  fTransformedResCf(0),
  fRadiusFactor(1.)
{
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsY());

  cout << "Building NeutralResidualCf object" << endl;
  cout << "\tResidualType = " << cAnalysisBaseTags[fResidualType] << endl;
  cout << "\tLambdaFactor = " << fLambdaFactor << endl << endl;

  SetDaughtersAndMothers();

  omp_set_num_threads(3);
}



//________________________________________________________________________________________________________________
NeutralResidualCf::~NeutralResidualCf()
{}

//________________________________________________________________________
double NeutralResidualCf::GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double NeutralResidualCf::GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double NeutralResidualCf::LednickyEq(double *x, double *par)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first

  std::complex<double> f0 (par[2],par[3]);
  double Alpha = 0.; // alpha = 0 for non-identical
  double z = 2.*(x[0]/hbarc)*par[1];  //z = 2k*R, to be fed to GetLednickyF1(2)

  double C_QuantumStat = Alpha*exp(-z*z);  // will be zero for my analysis

  std::complex<double> ScattAmp = pow( (1./f0) + 0.5*par[4]*(x[0]/hbarc)*(x[0]/hbarc) - ImI*(x[0]/hbarc),-1);

  double C_FSI = (1+Alpha)*( 0.5*norm(ScattAmp)/(par[1]*par[1])*(1.-1./(2*sqrt(TMath::Pi()))*(par[4]/par[1])) + 2.*real(ScattAmp)/(par[1]*sqrt(TMath::Pi()))*GetLednickyF1(z) - (imag(ScattAmp)/par[1])*GetLednickyF2(z));

  double Cf = 1. + par[0]*(C_QuantumStat + C_FSI);
  //Cf *= par[5];

  return Cf;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::Convert1dVecToHist(td1dVec &aCfVec, td1dVec &aKStarBinCenters, TString aTitle)
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
void NeutralResidualCf::SetDaughtersAndMothers()
{
  vector<ParticlePDGType> tDaughtersAndMothers = GetResidualDaughtersAndMothers(fResidualType);
  fMotherType1 = tDaughtersAndMothers[0];
  fDaughterType1 = tDaughtersAndMothers[1];
  fMotherType2 = tDaughtersAndMothers[2];
  fDaughterType2 = tDaughtersAndMothers[3];
}


//________________________________________________________________________________________________________________
td1dVec NeutralResidualCf::GetNeutralResidualCorrelation(double *aCfParams)
{
  fResCf.clear();
  fResCf.resize(fKStarBinCenters.size(),0.);
  double tKStar[1];
  aCfParams[1] *= fRadiusFactor;
  for(unsigned int i=0; i<fKStarBinCenters.size(); i++)
  {
    tKStar[0] = fKStarBinCenters[i];
    fResCf[i] = LednickyEq(tKStar,aCfParams);
//TODO TODO TODO Not sure whether above is correct
  }
  return fResCf;
}



//________________________________________________________________________________________________________________
td1dVec NeutralResidualCf::GetTransformedNeutralResidualCorrelation(double *aCfParams)
{
  omp_set_num_threads(6);

  td1dVec tResCf = GetNeutralResidualCorrelation(aCfParams);

  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(tResCf.size() == fKStarBinCenters.size());
  assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsY());

  fTransformedResCf.clear();
  fTransformedResCf.resize(tResCf.size(),0.);
  vector<double> tNormVec(tResCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

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
  return fTransformedResCf;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetNeutralResidualCorrelationHistogram(double *aCfParams, TString aTitle)
{
  td1dVec tResCf = GetNeutralResidualCorrelation(aCfParams);
  TH1D* tReturnHist = Convert1dVecToHist(tResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetTransformedNeutralResidualCorrelationHistogram(double *aCfParams, TString aTitle)
{
  td1dVec tTransResCf = GetTransformedNeutralResidualCorrelation(aCfParams);
  TH1D* tReturnHist = Convert1dVecToHist(tTransResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
double* NeutralResidualCf::AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries)
{
  double *tReturnArray = new double[aNEntries];
  tReturnArray[0] = aNewLambda*aParamSet[0];
  for(int i=1; i<aNEntries; i++) tReturnArray[i] = aParamSet[i];

  return tReturnArray;
}



//________________________________________________________________________________________________________________
td1dVec NeutralResidualCf::GetContributionToFitCf(double *aParamsOverall)
{
  double* tNewParams = AdjustLambdaParam(aParamsOverall, fLambdaFactor);
  td1dVec tReturnVec = GetTransformedNeutralResidualCorrelation(tNewParams);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] -= 1.;
  return tReturnVec;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall)
{
  double* tNewParams = AdjustLambdaParam(aParamsOverall, fLambdaFactor);
  td1dVec tReturnVec = GetNeutralResidualCorrelation(tNewParams);
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetTransformedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double *aParamsOverall)
{
  td1dVec tReturnVec = GetContributionToFitCf(aParamsOverall);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] += 1.;
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetResidualCorrelationHistogramWithLambdaAndNormApplied(TString aTitle, double *aParamsOverall, double aNorm)
{
  TH1D* tReturnHist = GetResidualCorrelationHistogramWithLambdaApplied(aTitle, aParamsOverall);
  tReturnHist->Scale(aNorm);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetTransformedResidualCorrelationHistogramWithLambdaAndNormApplied(TString aTitle, double *aParamsOverall, double aNorm)
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
void NeutralResidualCf::SetUsemTScalingOfRadii(AnalysisType aParentAnType, double aPower)
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





