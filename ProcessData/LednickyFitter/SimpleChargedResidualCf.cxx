/* SimpleChargedResidualCf.cxx */

#include "SimpleChargedResidualCf.h"
#include "FitPairAnalysis.h"

#ifdef __ROOT__
ClassImp(SimpleChargedResidualCf)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
SimpleChargedResidualCf::SimpleChargedResidualCf(AnalysisType aResidualType, TH2D* aTransformMatrix, td1dVec &aKStarBinCenters, CentralityType aCentType, double aMaxKStar, TString aFileLocationBase) :
  fResidualType(aResidualType),
  fPairAn(nullptr),
  fExpXiHist(nullptr),
  fLambdaFactor(cAnalysisLambdaFactors[fResidualType]),
  fTransformMatrix(aTransformMatrix),
  fKStarBinCenters(aKStarBinCenters),
  fResCf(0),
  fTransformedResCf(0)
{
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsX());
  assert(fKStarBinCenters.size() == (unsigned int)fTransformMatrix->GetNbinsY());

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

  TString tName = TString("ExpXiHist_") + TString(cAnalysisBaseTags[fResidualType]) + TString(cCentralityTags[aCentType]);
  cout << "Building SimpleChargedResidualCf object" << endl;
  cout << "\tResidualType   = " << cAnalysisBaseTags[fResidualType] << endl;
  cout << "\tUsing experimental data from " << cAnalysisBaseTags[tAnType] << " analysis" << endl;
  cout << "\tCentralityType = " << cPrettyCentralityTags[aCentType] << endl;
  cout << "\tLambdaFactor   = " << fLambdaFactor << endl << endl;
  
  fPairAn = new FitPairAnalysis(aFileLocationBase,tAnType,aCentType,tRunType,tNFitPartialAnalysis);
  fPairAn->RebinKStarCfHeavy(2,0.32,0.4);
  fExpXiHist = (TH1D*)fPairAn->GetKStarCfHeavy()->GetHeavyCf();
    fExpXiHist->SetName(tName);
    fExpXiHist->SetTitle(tName);
    fExpXiHist->SetDirectory(0);
  assert(fExpXiHist->GetXaxis()->GetBinWidth(1)==0.01);  //TODO make general
  assert(fExpXiHist->GetNbinsX()==100);
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
td1dVec SimpleChargedResidualCf::GetChargedResidualCorrelation(double aMaxKStar)
{
  int tNbins = std::round(aMaxKStar/fExpXiHist->GetXaxis()->GetBinWidth(1));
  if(fResCf.size()!=tNbins)
  {
    fResCf.clear();
    fResCf.resize(tNbins,0.);
    for(int i=0; i<tNbins; i++) fResCf[i] = fExpXiHist->GetBinContent(i+1);

//    delete tPairAn;
  }
  return fResCf;
}


//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::GetTransformedChargedResidualCorrelation(double aMaxKStar)
{
  if(fTransformedResCf.size()==0)
  {
    td1dVec tResCf = GetChargedResidualCorrelation(aMaxKStar);

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
    }
  }
  return fTransformedResCf;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetChargedResidualCorrelationHistogram(TString aTitle, double aMaxKStar)
{
  td1dVec tResCf = GetChargedResidualCorrelation(aMaxKStar);
  TH1D* tReturnHist = Convert1dVecToHist(tResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetTransformedChargedResidualCorrelationHistogram(TString aTitle, double aMaxKStar)
{
  td1dVec tTransResCf = GetTransformedChargedResidualCorrelation(aMaxKStar);
  TH1D* tReturnHist = Convert1dVecToHist(tTransResCf, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
td1dVec SimpleChargedResidualCf::GetContributionToFitCf(double aOverallLambda, double aMaxKStar)
{
  td1dVec tReturnVec = GetTransformedChargedResidualCorrelation(aMaxKStar);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] = fLambdaFactor*aOverallLambda*(tReturnVec[i]-1.);
  return tReturnVec;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aMaxKStar)
{
  td1dVec tReturnVec = GetChargedResidualCorrelation(aMaxKStar);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] = 1. + fLambdaFactor*aOverallLambda*(tReturnVec[i]-1.);  //TODO is this right?
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1D* SimpleChargedResidualCf::GetTransformedChargedResidualCorrelationHistogramWithLambdaApplied(TString aTitle, double aOverallLambda, double aMaxKStar)
{
  td1dVec tReturnVec = GetContributionToFitCf(aOverallLambda, aMaxKStar);
  for(unsigned int i=0; i<tReturnVec.size(); i++) tReturnVec[i] += 1.;
  TH1D* tReturnHist = Convert1dVecToHist(tReturnVec, fKStarBinCenters, aTitle);
  return tReturnHist;
}


