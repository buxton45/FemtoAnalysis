/* NeutralResidualCf.cxx */

#include "NeutralResidualCf.h"

#ifdef __ROOT__
ClassImp(NeutralResidualCf)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
NeutralResidualCf::NeutralResidualCf(AnalysisType aResidualType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters) :
  fResidualType(aResidualType),
  fTransformMatrix(aTransformMatrix),
  fKStarBinCenters(aKStarBinCenters),
  fResCf(0),
  fTransformedResCf(0)
{
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsY());
}



//________________________________________________________________________________________________________________
NeutralResidualCf::~NeutralResidualCf()
{}




//________________________________________________________________________________________________________________
td1dVec NeutralResidualCf::GetNeutralResidualCorrelation(double *aParentCfParams)
{
  fResCf.clear();
  fResCf.resize(fKStarBinCenters.size(),0.);
  double tKStar[1];
  for(unsigned int i=0; i<fKStarBinCenters.size(); i++)
  {
    tKStar[0] = fKStarBinCenters[i];
    fResCf[i] = LednickyFitter::LednickyEq(tKStar,aParentCfParams);
//TODO TODO TODO Not sure whether above is correct
  }
  return fResCf;
}



//________________________________________________________________________________________________________________
td1dVec NeutralResidualCf::GetTransformedNeutralResidualCorrelation(double *aParentCfParams)
{
  td1dVec tResCf = GetNeutralResidualCorrelation(aParentCfParams);

  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(tResCf.size() == fKStarBinCenters.size());
  assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(tResCf.size() == (unsigned int)fTransformMatrix->GetNbinsY());

  vector<double> tTransResCf(tResCf.size(),0.);
  vector<double> tNormVec(tResCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<tResCf.size(); i++)
  {
    tDaughterPairKStar = fKStarBinCenters[i];
    tDaughterPairKStarBin = fTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

    for(unsigned int j=0; j<tResCf.size(); j++)
    {
      tParentPairKStar = fKStarBinCenters[j];
      tParentPairKStarBin = fTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tTransResCf[i] += tResCf[j]*fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec[i] += fTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    tTransResCf[i] /= tNormVec[i];
  }
  return tTransResCf;
}

//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle)
{
  td1dVec tResCf = GetNeutralResidualCorrelation(aParentCfParams);
  TH1D* tReturnHist = LednickyFitter::Convert1dVecToHist(tResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1D* NeutralResidualCf::GetTransformedNeutralResidualCorrelationHistogram(double *aParentCfParams, TString aTitle)
{
  td1dVec tTransResCf = GetTransformedNeutralResidualCorrelation(aParentCfParams);
  TH1D* tReturnHist = LednickyFitter::Convert1dVecToHist(tTransResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}

