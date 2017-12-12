///////////////////////////////////////////////////////////////////////////
// SimpleLednickyFitter:                                                 //
///////////////////////////////////////////////////////////////////////////


#include "SimpleLednickyFitter.h"

#ifdef __ROOT__
ClassImp(SimpleLednickyFitter)
#endif

//  Global variables needed to be seen by FCN
/*
vector<TH1F*> gCfsToFit;
int gNpFits;
vector<int> gNpfitsVec;
//vector<double> gMaxFitKStar;
double gMaxFitKStar;
*/


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
SimpleLednickyFitter::SimpleLednickyFitter(AnalysisType aAnalysisType, CfLite *aCfLite, double aMaxFitKStar):
  fAnalysisType(aAnalysisType),
  fCfLite(aCfLite),
  fFitType(kChi2PML),
  fVerbose(false),
  fMinuit(nullptr),
  fCorrectedFitVec(0),
  fMaxFitKStar(aMaxFitKStar),
  fNbinsXToBuild(0),
  fNbinsXToFit(0),
  fKStarBinWidth(0.),
  fKStarBinCenters(0),
  fRejectOmega(false),

  fChi2(0),
  fChi2GlobalMin(1000000000),
  fNpFits(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0),

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFit(nullptr),
  fIncludeResidualsType(kIncludeNoResiduals),
  fTransformMatrices(0),
  fTransformStorageMapping(0),
  fResidualCollection(nullptr)

{
  fMinuit = new TMinuit(50);
}


//________________________________________________________________________________________________________________
SimpleLednickyFitter::SimpleLednickyFitter(AnalysisType aAnalysisType, TString aFileLocation, TString aBaseName, double aMaxFitKStar):
  fAnalysisType(aAnalysisType),
  fCfLite(nullptr),
  fFitType(kChi2PML),
  fVerbose(false),
  fMinuit(nullptr),
  fCorrectedFitVec(0),
  fMaxFitKStar(aMaxFitKStar),
  fNbinsXToBuild(0),
  fNbinsXToFit(0),
  fKStarBinWidth(0.),
  fKStarBinCenters(0),
  fRejectOmega(false),

  fChi2(0),
  fChi2GlobalMin(1000000000),
  fNpFits(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
//  double aMinNorm = 0.32, aMaxNorm = 0.40;
  double aMinNorm = 0.80, aMaxNorm = 0.99;
  TH1D* tNum = Get1dHisto(aFileLocation, TString("Num")+aBaseName+cAnalysisBaseTags[fAnalysisType]);
  TH1D* tDen = Get1dHisto(aFileLocation, TString("Den")+aBaseName+cAnalysisBaseTags[fAnalysisType]);

  fCfLite = new CfLite(TString("CfLite")+aBaseName+cAnalysisBaseTags[fAnalysisType], 
                       TString("CfLite")+aBaseName+cAnalysisBaseTags[fAnalysisType],
                       tNum, tDen, aMinNorm, aMaxNorm);

  fMinuit = new TMinuit(50);
}

//________________________________________________________________________________________________________________
SimpleLednickyFitter::SimpleLednickyFitter(AnalysisType aAnalysisType, vector<double> &aSimParams, double aMaxBuildKStar, double aMaxFitKStar):
  fAnalysisType(aAnalysisType),
  fCfLite(nullptr),
  fFitType(kChi2PML),
  fVerbose(false),
  fMinuit(nullptr),
  fCorrectedFitVec(0),
  fMaxFitKStar(aMaxFitKStar),
  fNbinsXToBuild(0),
  fNbinsXToFit(0),
  fKStarBinWidth(0.),
  fKStarBinCenters(0),
  fRejectOmega(false),

  fChi2(0),
  fChi2GlobalMin(1000000000),
  fNpFits(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  double aMinNorm = 0.32, aMaxNorm = 0.40;
  TH1D* tNum = GetSimluatedNumDen(true, aSimParams, aMaxBuildKStar);
  TH1D* tDen = GetSimluatedNumDen(false, aSimParams, aMaxBuildKStar);

  fCfLite = new CfLite(TString("CfLiteSim")+cAnalysisBaseTags[fAnalysisType], 
                       TString("CfLiteSim")+cAnalysisBaseTags[fAnalysisType],
                       tNum, tDen, aMinNorm, aMaxNorm);

  fMinuit = new TMinuit(50);

}


//________________________________________________________________________________________________________________
SimpleLednickyFitter::~SimpleLednickyFitter()
{
  cout << "LednickyFitter object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
void SimpleLednickyFitter::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
{
  std::default_random_engine tGenerator (std::clock());  //std::clock() is seed
  std::uniform_real_distribution<double> tKStarMagDistribution(aKStarMagMin,aKStarMagMax);
  std::uniform_real_distribution<double> tUnityDistribution(0.,1.);

  double tKStarMag = tKStarMagDistribution(tGenerator);
  double tU = tUnityDistribution(tGenerator);
  double tV = tUnityDistribution(tGenerator);

  double tTheta = acos(2.*tV-1.); //polar angle
  double tPhi = 2.*M_PI*tU; //azimuthal angle

  aKStar3Vec->SetMagThetaPhi(tKStarMag,tTheta,tPhi);
}

//________________________________________________________________________________________________________________
complex<double> SimpleLednickyFitter::GetStrongOnlyWaveFunction(TVector3* aKStar3Vec, TVector3* aRStar3Vec, vector<double> &aSimParams)
{
  if(aRStar3Vec->X()==0 && aRStar3Vec->Y()==0 && aRStar3Vec->Z()==0)  //TODO i.e. if pair originate from single resonance
  {
    double tRoot2 = sqrt(2.);
    double tRadius = 1.0;
    std::default_random_engine generator (std::clock());  //std::clock() is seed
    std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
    std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

    aRStar3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator));
  }


  complex<double> ImI (0., 1.);
  complex<double> tF0 (aSimParams[2], aSimParams[3]);
  double tD0 = aSimParams[4];

  double tKdotR = aKStar3Vec->Dot(*aRStar3Vec);
    tKdotR /= hbarc;
  double tKStarMag = aKStar3Vec->Mag();
    tKStarMag /= hbarc;
  double tRStarMag = aRStar3Vec->Mag();

  complex<double> tScattLenLastTerm (0., tKStarMag);
  complex<double> tScattAmp = pow((1./tF0) + 0.5*tD0*tKStarMag*tKStarMag - tScattLenLastTerm,-1);

  complex<double> tReturnWf = exp(ImI*tKdotR) + tScattAmp*exp(ImI*tKStarMag*tRStarMag)/tRStarMag;
  return tReturnWf;
}

//________________________________________________________________________________________________________________
double SimpleLednickyFitter::GetStrongOnlyWaveFunctionSq(TVector3 *aKStar3Vec, TVector3 *aRStar3Vec, vector<double> &aSimParams)
{
  complex<double> tWf = GetStrongOnlyWaveFunction(aKStar3Vec, aRStar3Vec, aSimParams);
  double tWfSq = norm(tWf);
  return tWfSq;
}



//________________________________________________________________________________________________________________
TH1D* SimpleLednickyFitter::GetSimluatedNumDen(bool aBuildNum, vector<double> &aSimParams, double aMaxBuildKStar, int aNPairsPerKStarBin, double aKStarBinSize)
{
  int tNBins = aMaxBuildKStar/aKStarBinSize;
cout << "tNBins = " << tNBins << endl;
  TH1D* tReturnHist;
  if(aBuildNum) tReturnHist = new TH1D(TString::Format("SimNum_%s", cAnalysisBaseTags[fAnalysisType]),
                                       TString::Format("SimNum_%s", cAnalysisBaseTags[fAnalysisType]),
                                       tNBins, 0., aMaxBuildKStar);
  else tReturnHist = new TH1D(TString::Format("SimDen_%s", cAnalysisBaseTags[fAnalysisType]),
                              TString::Format("SimDen_%s", cAnalysisBaseTags[fAnalysisType]),
                              tNBins, 0., aMaxBuildKStar);
  tReturnHist->Sumw2();

  int tNPairsPerKStarBin = aNPairsPerKStarBin;
  if(!aBuildNum) tNPairsPerKStarBin *= 5. * ((double)rand()/(RAND_MAX));

  assert(aSimParams.size()==6);
  //double tLambda = aSimParams[0];
  //double tRadius = aSimParams[1];
  //double tReF0   = aSimParams[2];
  //double tImF0   = aSimParams[3];
  //double tD0     = aSimParams[4];
  //double tNorm   = aSimParams[5];

  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*aSimParams[1]);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*aSimParams[1]);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*aSimParams[1]);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);
  double tKStarMagMin, tKStarMagMax, tKStarMagAvg;
  double tWeight = 1.;
  for(int iKStarBin=0; iKStarBin<tNBins; iKStarBin++)
  {
    tKStarMagMin = iKStarBin*aKStarBinSize;
    if(iKStarBin==0) tKStarMagMin=0.004;
    tKStarMagMax = (iKStarBin+1)*aKStarBinSize;
    tKStarMagAvg = 0.5*(tKStarMagMin + tKStarMagMax);
    for(int iPair=0; iPair<std::round(tKStarMagAvg*tKStarMagAvg*tNPairsPerKStarBin); iPair++)
    {
      SetRandomKStar3Vec(tKStar3Vec,tKStarMagMin,tKStarMagMax);
      tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric
      if(aBuildNum) tWeight = GetStrongOnlyWaveFunctionSq(tKStar3Vec, tSource3Vec, aSimParams);
      else tWeight = 1.;
      tReturnHist->Fill(tKStar3Vec->Mag(), tWeight);
    }
  }

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TF1* SimpleLednickyFitter::FitNonFlatBackground(TH1* aCf, double aMinFit, double aMaxFit)
{
  TF1* tNonFlatBackground;

  TString tFitName = TString("NonFlatBackgroundFitLinear_") + TString(aCf->GetTitle());
  tNonFlatBackground = new TF1(tFitName,BackgroundFitter::FitFunctionLinear,0.,1.,2);
    tNonFlatBackground->SetParameter(0,0.);
    tNonFlatBackground->SetParameter(1,1.);
  aCf->Fit(tFitName,"0q","",aMinFit,aMaxFit);

  return tNonFlatBackground;
}


//________________________________________________________________________________________________________________
TH1D* SimpleLednickyFitter::Get1dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH1D *ReturnHisto = (TH1D*)f1.Get(HistoName);

  TH1D *ReturnHistoClone = (TH1D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
void SimpleLednickyFitter::CreateMinuitParameters()
{
  int tErrFlg = 0;

  double tStartVal_Lambda = 1.0;
  double tStepSize_Lambda = 0.001;
  double tLowerBound_Lambda = 0.;
  double tUpperBound_Lambda = 0.;
  fMinuit->mnparm(0, TString("Lambda"), tStartVal_Lambda, tStepSize_Lambda, tLowerBound_Lambda, tUpperBound_Lambda, tErrFlg);
//  fMinuit->FixParameter(0);

  double tStartVal_Radius = 5.0;
  double tStepSize_Radius = 0.001;
  double tLowerBound_Radius = 0.;
  double tUpperBound_Radius = 0.;
  if(fIncludeResidualsType != kIncludeNoResiduals) {tLowerBound_Radius=1.; tUpperBound_Radius=12.;}
  fMinuit->mnparm(1, TString("Radius"), tStartVal_Radius, tStepSize_Radius, tLowerBound_Radius, tUpperBound_Radius, tErrFlg);
//  fMinuit->FixParameter(1);

  double tStartVal_ReF0 = -0.5;
  double tStepSize_ReF0 = 0.001;
  double tLowerBound_ReF0 = 0.;
  double tUpperBound_ReF0 = 0.;
  fMinuit->mnparm(2, TString("ReF0"), tStartVal_ReF0, tStepSize_ReF0, tLowerBound_ReF0, tUpperBound_ReF0, tErrFlg);
//  fMinuit->FixParameter(2);

  double tStartVal_ImF0 = 0.5;
  double tStepSize_ImF0 = 0.001;
  double tLowerBound_ImF0 = 0.;
  double tUpperBound_ImF0 = 0.;
  fMinuit->mnparm(3, TString("ImF0"), tStartVal_ImF0, tStepSize_ImF0, tLowerBound_ImF0, tUpperBound_ImF0, tErrFlg);
//  fMinuit->FixParameter(3);

  double tStartVal_D0 = 0.0;
  double tStepSize_D0 = 0.001;
  double tLowerBound_D0 = 0.;
  double tUpperBound_D0 = 0.;
  fMinuit->mnparm(4, TString("D0"), tStartVal_D0, tStepSize_D0, tLowerBound_D0, tUpperBound_D0, tErrFlg);
//  fMinuit->FixParameter(4);

  double tStartVal_Norm = (double)fCfLite->GetNumScale()/(double)fCfLite->GetDenScale();
  double tStepSize_Norm = 0.001;
  double tLowerBound_Norm = 0.;
  double tUpperBound_Norm = 0.;
  fMinuit->mnparm(5, TString("Norm"), tStartVal_Norm, tStepSize_Norm, tLowerBound_Norm, tUpperBound_Norm, tErrFlg);
//  fMinuit->FixParameter(5);
}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
{
  if(fVerbose) LednickyFitter::PrintCurrentParamValues(6,par);
  //---------------------------------------------------------
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  int tNFitParams = 6;

  vector<double> tPrimaryFitCfContent(fNbinsXToBuild,0.);
  vector<double> tNumContent(fNbinsXToBuild,0.);
  vector<double> tDenContent(fNbinsXToBuild,0.);

  fChi2 = 0.;

  fNpFits = 0.;

  TH1* tNum = fCfLite->Num();
  TH1* tDen = fCfLite->Den();
  TH1* tCf = fCfLite->Cf();

  int tLambdaMinuitParamNumber = 0;
  int tRadiusMinuitParamNumber = 1;
  int tRef0MinuitParamNumber = 2;
  int tImf0MinuitParamNumber = 3;
  int td0MinuitParamNumber = 4;
  int tNormMinuitParamNumber = 5;

  double *tParPrim = new double[tNFitParams];

//  if(fIncludeResidualsType != kIncludeNoResiduals) tParPrim[0] = cAnalysisLambdaFactors[fAnalysisType]*par[tLambdaMinuitParamNumber];
  if(fIncludeResidualsType != kIncludeNoResiduals) tParPrim[0] = cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fAnalysisType]*par[tLambdaMinuitParamNumber];
  else tParPrim[0] = par[tLambdaMinuitParamNumber];
  tParPrim[1] = par[tRadiusMinuitParamNumber];
  tParPrim[2] = par[tRef0MinuitParamNumber];
  tParPrim[3] = par[tImf0MinuitParamNumber];
  tParPrim[4] = par[td0MinuitParamNumber];
  tParPrim[5] = par[tNormMinuitParamNumber];

  for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
  {
    if(std::isnan(tParPrim[i])) {cout <<"CRASH:  In CalculateFitFunction, a tParPrim elemement " << i << " DNE!!!!!" << endl;}
    assert(!std::isnan(tParPrim[i]));
  }

  double x[1];
  bool tRejectOmega = false;
  if(fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) tRejectOmega = true; 

  vector<double> tFitCfContent;
  vector<double> tCorrectedFitCfContent;

  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    x[0] = fKStarBinCenters[ix-1];

    tNumContent[ix-1] = tNum->GetBinContent(ix);
    tDenContent[ix-1] = tDen->GetBinContent(ix);

    tPrimaryFitCfContent[ix-1] = LednickyFitter::LednickyEq(x,tParPrim);
  }

  if(fIncludeResidualsType != kIncludeNoResiduals) 
  {
    double *tParOverall = new double[tNFitParams];
    tParOverall[0] = par[tLambdaMinuitParamNumber];
    tParOverall[1] = par[tRadiusMinuitParamNumber];
    tParOverall[2] = par[tRef0MinuitParamNumber];
    tParOverall[3] = par[tImf0MinuitParamNumber];
    tParOverall[4] = par[td0MinuitParamNumber];
    tParOverall[5] = par[tNormMinuitParamNumber];
    tFitCfContent = GetFitCfIncludingResiduals(tPrimaryFitCfContent, tParOverall);
    delete[] tParOverall;
  }
  else tFitCfContent = tPrimaryFitCfContent;

  tCorrectedFitCfContent = tFitCfContent;

  if(fApplyNonFlatBackgroundCorrection) LednickyFitter::ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, fKStarBinCenters, fNonFlatBgdFit);

  fCorrectedFitVec = tCorrectedFitCfContent;
  LednickyFitter::ApplyNormalization(tParPrim[5], tCorrectedFitCfContent);

  for(int ix=0; ix < fNbinsXToFit; ix++)
  {
    if(tRejectOmega && (fKStarBinCenters[ix] > tRejectOmegaLow) && (fKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
    else
    {
      if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
      {
        double tChi2 = 0.;
        if(fFitType == kChi2PML) tChi2 = LednickyFitter::GetPmlValue(tNumContent[ix],tDenContent[ix],tCorrectedFitCfContent[ix]);
        else if(fFitType == kChi2) tChi2 = LednickyFitter::GetChi2Value(ix+1,tCf,tCorrectedFitCfContent[ix]);
        else tChi2 = 0.;

        fChi2 += tChi2;
        fNpFits++;
      }
    }
  }
  delete[] tParPrim;


  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;

  if(fVerbose)
  {
    cout << "fChi2 = " << fChi2 << endl;
    cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;
  }

}

//________________________________________________________________________________________________________________
TF1* SimpleLednickyFitter::CreateFitFunction(TString aName)
{
  int tNFitParams = 5;
  TF1* ReturnFunction = new TF1(aName,LednickyFitter::LednickyEq,0.,0.5,tNFitParams+1);
  double tParamValue, tParamError;
  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    tParamValue = fMinParams[iPar];
    tParamError = fParErrors[iPar];
    if(iPar==0 && fIncludeResidualsType != kIncludeNoResiduals)
    {
//      tParamValue *= cAnalysisLambdaFactors[fAnalysisType];
//      tParamError *= cAnalysisLambdaFactors[fAnalysisType];
      tParamValue *= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fAnalysisType];
      tParamError *= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fAnalysisType];
    }

    ReturnFunction->SetParameter(iPar,tParamValue);
    ReturnFunction->SetParError(iPar,tParamError);
  }

  ReturnFunction->SetParameter(5,1.);
  ReturnFunction->SetParError(5,0.);

  ReturnFunction->SetChisquare(fChi2);
  ReturnFunction->SetNDF(fNDF);

  ReturnFunction->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");

  return ReturnFunction;
}



//________________________________________________________________________________________________________________
void SimpleLednickyFitter::InitializeFitter()
{
  cout << "----- Initializing fitter -----" << endl;
  CreateMinuitParameters();

  fNbinsXToBuild = 0;
  fNbinsXToFit = 0;
  fKStarBinWidth = 0.;
  fKStarBinCenters.clear();
  //-------------------------
  int tTempNbinsXToFit = 0;  //This should equal fNbinsXToFit, but keep it for consistency/sanity check
  int tTempNbinsXToBuild = 0;  //This should equal fNbinsXToBuild, but keep it for consistency/sanity check
  int tNbinsXToBuildMomResCrctn=0;
  int tNbinsXToBuildResiduals=0;

  //----- Set everything using first partial analysis, check consistency in loops below -----
  fNbinsXToFit = fCfLite->Num()->FindBin(fMaxFitKStar);
  if(fCfLite->Num()->GetBinLowEdge(fNbinsXToFit) == fMaxFitKStar) fNbinsXToFit--;

  if(fIncludeResidualsType != kIncludeNoResiduals) tNbinsXToBuildResiduals = GetTransformMatrix(0)->GetNbinsX();
  fNbinsXToBuild = std::max({tNbinsXToBuildMomResCrctn, tNbinsXToBuildResiduals, fNbinsXToFit});

  fKStarBinWidth = fCfLite->Num()->GetXaxis()->GetBinWidth(1);
  //-------------------------------------------------------------------------------------------


  TH1* tNum = fCfLite->Num();
  TH1* tDen = fCfLite->Den();
  TH1* tCf = fCfLite->Cf();

  assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
  assert(tNum->GetXaxis()->GetBinWidth(1) == tCf->GetXaxis()->GetBinWidth(1));
  assert(tNum->GetXaxis()->GetBinWidth(1) == fKStarBinWidth);

  //make sure tNum and tDen and tCf have same bin size as residuals
  if(fIncludeResidualsType != kIncludeNoResiduals)
  {
    assert(tNum->GetXaxis()->GetBinWidth(1) == fTransformMatrices[0]->GetXaxis()->GetBinWidth(1));
    assert(tNum->GetXaxis()->GetBinWidth(1) == fTransformMatrices[0]->GetYaxis()->GetBinWidth(1));
  }

  //make sure tNum and tDen have same number of bins
  assert(tNum->GetNbinsX() == tDen->GetNbinsX());
  assert(tNum->GetNbinsX() == tCf->GetNbinsX());

  tTempNbinsXToFit = tNum->FindBin(fMaxFitKStar);
  if(tNum->GetBinLowEdge(tTempNbinsXToFit) == fMaxFitKStar) tTempNbinsXToFit--;

  if(tTempNbinsXToFit > tNum->GetNbinsX()) {tTempNbinsXToFit = tNum->GetNbinsX();}  //in case I accidentally include an overflow bin in nbinsXToFit
  assert(tTempNbinsXToFit == fNbinsXToFit);

  if(fIncludeResidualsType == kIncludeNoResiduals) fNbinsXToBuild = fNbinsXToFit;

  fKStarBinCenters.resize(fNbinsXToBuild,0.);
  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    fKStarBinCenters[ix-1] = tNum->GetXaxis()->GetBinCenter(ix);
  }

  if(fApplyNonFlatBackgroundCorrection) fNonFlatBgdFit = FitNonFlatBackground(tCf, 0.5, 1.0);

  if(fIncludeResidualsType != kIncludeNoResiduals) InitiateResidualCollection(fKStarBinCenters);

}

//________________________________________________________________________________________________________________
void SimpleLednickyFitter::DoFit()
{
  InitializeFitter();

  cout << "*****************************************************************************" << endl;
  cout << "Starting to fit " << endl;
  cout << "*****************************************************************************" << endl;

  fMinuit->SetPrintLevel(0);  // -1=quiet, 0=normal, 1=verbose (more options using mnexcm("SET PRI", ...)

  double arglist[10];
  fErrFlg = 0;

  // for max likelihood = 0.5, for chisq = 1.0
  arglist[0] = 1.;
  fMinuit->mnexcm("SET ERR", arglist ,1,fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 2;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 50000;
  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,fErrFlg);

  if(fErrFlg != 0)
  {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    //cout << "WARNING!!!!!" << endl << "fErrFlg != 0 for the Cf: " << fCfName << endl;
    cout << "WARNING!!!!!" << endl << "fErrFlg != 0 for the Cf: " << endl;
    cout << "fErrFlg = " << fErrFlg << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }

  // Print results
  fMinuit->mnstat(fChi2,fEdm,fErrDef,fNvpar,fNparx,fIcstat);
  fMinuit->mnprin(3,fChi2);

  //---------------------------------
  Finalize();
}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::Finalize()
{
  int tNParams = 6;
  fNDF = fNpFits-fNvpar;

  //get result
  for(int i=0; i<tNParams; i++)
  {
    double tempMinParam;
    double tempParError;
    fMinuit->GetParameter(i,tempMinParam,tempParError);
    
    fMinParams.push_back(tempMinParam);
    fParErrors.push_back(tempParError);
  }


/*
  TF1* tFit = CreateFitFunction(TString("Fit"));
  vector<double> tCorrectedFitVecWithNorm = fCorrectedFitVec;
  LednickyFitter::ApplyNormalization(fMinParams[5], tCorrectedFitVecWithNorm);
  TH1* tCf = fCfLite->Cf();
  TH1* tNum = fCfLite->Num();
  TH1* tDen = fCfLite->Den();

  for(int i=0; i<30; i++)
  {
    cout << "i = " << i << endl;
    cout << "tCf->GetBinCenter = " << tCf->GetBinCenter(i+1) << endl;
    cout << "tCf->GetBinContent = " << tCf->GetBinContent(i+1) << endl;
    cout << "tNum/tDen = " << tNum->GetBinContent(i+1)/tDen->GetBinContent(i+1) << endl;
    cout << "fCorrectedFitVec = " << fCorrectedFitVec[i] << endl;
    cout << "tCorrectedFitVecWithNorm = " << tCorrectedFitVecWithNorm[i] << endl;
    cout << "tFit = " << tFit->Eval(tCf->GetBinCenter(i+1)) << endl;
    cout << "-------------------------------------------------------" << endl << endl;
  }
*/

}


//________________________________________________________________________________________________________________
TH1D* SimpleLednickyFitter::GetCorrectedFitHist()
{
  assert(fCorrectedFitVec.size());
  assert(fCorrectedFitVec.size() == fKStarBinCenters.size());

  int tNbins = fKStarBinCenters.size();
  double tKStarBinWidth = fKStarBinCenters[1] - fKStarBinCenters[0];
  double tKStarMin = fKStarBinCenters[0] - tKStarBinWidth/2.;
    if(tKStarMin < 0.0001) tKStarMin = 0.;  //not that it's terribly important, but getting tKStarMin ~ 1e-19
  double tKStarMax = fKStarBinCenters[fKStarBinCenters.size()-1] + tKStarBinWidth/2;

  TString tTitle = TString::Format("CorrectedFitHist_%s", cAnalysisBaseTags[fAnalysisType]);
  TH1D* tReturnCf = new TH1D(tTitle, tTitle, tNbins, tKStarMin, tKStarMax);

  for(int i=1; i<=tNbins; i++)
  {
    tReturnCf->SetBinContent(i, fCorrectedFitVec[i-1]);
    tReturnCf->SetBinError(i, 0.);
  }

  return tReturnCf;
}

//________________________________________________________________________________________________________________
TPaveText* SimpleLednickyFitter::CreateParamFinalValuesText(TF1* aFit, double aTextXmin, double aTextYmin, double aTextWidth, double aTextHeight)
{
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

  if(fIncludeResidualsType != kIncludeNoResiduals)
  {
//    tLambda /= cAnalysisLambdaFactors[fAnalysisType];
//    tLambdaErr /= cAnalysisLambdaFactors[fAnalysisType];
    tLambda /= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fAnalysisType];
    tLambdaErr /= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][fAnalysisType];
  }

  TPaveText *tText = new TPaveText(aTextXmin, aTextYmin, aTextXmin+aTextWidth, aTextYmin+aTextHeight, "NDC");
  tText->AddText(TString::Format("#lambda = %0.2f #pm %0.2f",tLambda,tLambdaErr));
  tText->AddText(TString::Format("R = %0.2f #pm %0.2f",tRadius,tRadiusErr));
  tText->AddText(TString::Format("Re[f0] = %0.2f #pm %0.2f",tReF0,tReF0Err));
  tText->AddText(TString::Format("Im[f0] = %0.2f #pm %0.2f",tImF0,tImF0Err));
  tText->AddText(TString::Format("d0 = %0.2f #pm %0.2f",tD0,tD0Err));

  return tText;
}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::DrawCfWithFit(TPad *aPad, TString aDrawOption)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TH1D* tCf = (TH1D*)fCfLite->Cf();
  TF1* tFit = CreateFitFunction(TString("Fit"));


  tCf->Draw(aDrawOption);
  if(aDrawOption.EqualTo("same")) tFit->Draw(aDrawOption);
  else tFit->Draw(aDrawOption+TString("same"));

  TPaveText* tText = CreateParamFinalValuesText(tFit, 0.5, 0.5, 0.25, 0.25);
  tText->Draw();

  if(fApplyNonFlatBackgroundCorrection)
  {
    fNonFlatBgdFit->SetLineColor(kGreen+2);
    fNonFlatBgdFit->Draw("lsame");

    TPaveText* tTextBgd = new TPaveText(0.25, 0.50, 0.45, 0.60, "NDC");
    tTextBgd->AddText(TString::Format("BgdFit(x) = %0.1e*x + %0.2f", fNonFlatBgdFit->GetParameter(0), fNonFlatBgdFit->GetParameter(1)));
    tTextBgd->Draw();

    TH1D* tCorrectedCf = GetCorrectedFitHist();
      tCorrectedCf->SetMarkerColor(kMagenta+1);
      tCorrectedCf->SetLineColor(kMagenta+1);
      tCorrectedCf->SetLineWidth(2);
    tCorrectedCf->Draw("lsame");
  }

}

//________________________________________________________________________________________________________________
void SimpleLednickyFitter::DrawCfWithFitAndResiduals(TPad *aPad)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TH1D* tCf = (TH1D*)fCfLite->Cf();
    tCf->SetMarkerStyle(20);
  TF1* tFit = CreateFitFunction(TString("Fit"));
  TH1D* tCorrectedCf = GetCorrectedFitHist();
    tCorrectedCf->SetMarkerColor(kMagenta+1);
    tCorrectedCf->SetLineColor(kMagenta+1);
    tCorrectedCf->SetLineWidth(2);

  tCf->Draw();
  tFit->Draw("same");
  tCorrectedCf->Draw("lsame");

  TPaveText* tText = CreateParamFinalValuesText(tFit, 0.6, 0.6, 0.25, 0.20);
  tText->Draw();

  //------------------------------------ Residuals ---------------------------------------
  double tOverallLambdaPrimary = fMinParams[0];
  double tRadiusPrimary = fMinParams[1];

  vector<int> tNeutralResBaseColors{7,8,9,30,33,40,41};
  vector<int> tNeutralResMarkerStyles{24,25,26,27,28,30,32};
  vector<int> tChargedResBaseColors{44,46,47,49};
  vector<int> tChargedResMarkerStyles{24,25,26,27};

  //------ Neutral Residuals ------------------
  TLegend *tLegNeutral = new TLegend(0.35, 0.15, 0.55, 0.45);
    tLegNeutral->SetFillColor(0);
    tLegNeutral->SetBorderSize(0);
    tLegNeutral->SetTextAlign(22);
  for(unsigned int iRes=0; iRes<fResidualCollection->GetNeutralCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = fResidualCollection->GetNeutralCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = fResidualCollection->GetNeutralCollection()[iRes].GetTransformedNeutralResidualCorrelationHistogram(tTempName);
      tTempHist->SetMarkerColor(tNeutralResBaseColors[iRes]);
      tTempHist->SetLineColor(tNeutralResBaseColors[iRes]);
      tTempHist->SetMarkerStyle(tNeutralResMarkerStyles[iRes]);

    tLegNeutral->AddEntry(tTempHist, cAnalysisRootTags[tTempResidualType]);

    tTempHist->Draw("ex0same");
  }
  //------ Charged Residuals ------------------
  TLegend *tLegCharged = new TLegend(0.60, 0.25, 0.80, 0.45);
    tLegCharged->SetFillColor(0);
    tLegCharged->SetBorderSize(0);
    tLegCharged->SetTextAlign(22);
  for(unsigned int iRes=0; iRes<fResidualCollection->GetChargedCollection().size(); iRes++)
  {
    AnalysisType tTempResidualType = fResidualCollection->GetChargedCollection()[iRes].GetResidualType();
    TString tTempName = TString(cAnalysisRootTags[tTempResidualType]);
    TH1D* tTempHist = fResidualCollection->GetChargedCollection()[iRes].GetTransformedChargedResidualCorrelationHistogramWithLambdaApplied(tTempName, tOverallLambdaPrimary, tRadiusPrimary);
      tTempHist->SetMarkerColor(tChargedResBaseColors[iRes]);
      tTempHist->SetLineColor(tChargedResBaseColors[iRes]);
      tTempHist->SetMarkerStyle(tChargedResMarkerStyles[iRes]);

    tLegCharged->AddEntry(tTempHist, cAnalysisRootTags[tTempResidualType]);

    tTempHist->Draw("ex0same");
  }

  tLegNeutral->Draw();
  tLegCharged->Draw();

  if(fApplyNonFlatBackgroundCorrection)
  {
    fNonFlatBgdFit->SetLineColor(kGreen+2);
    fNonFlatBgdFit->Draw("lsame");

    TPaveText* tTextBgd = new TPaveText(0.30, 0.60, 0.50, 0.70, "NDC");
    tTextBgd->AddText(TString::Format("BgdFit(x) = %0.1e*x + %0.2f", fNonFlatBgdFit->GetParameter(0), fNonFlatBgdFit->GetParameter(1)));
    tTextBgd->Draw();
  }
}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::DrawCfNumDen(TPad *aPad, TString aDrawOption)
{
  aPad->cd();
  aPad->Divide(3,1);

  TH1D* tCf = (TH1D*)fCfLite->Cf();
  TH1D* tNum = (TH1D*)fCfLite->Num();
  TH1D* tDen = (TH1D*)fCfLite->Den();

  aPad->cd(1);
  tCf->Draw(aDrawOption);

  aPad->cd(2);
  tNum->Draw(aDrawOption);

  aPad->cd(3);
  tDen->Draw(aDrawOption);

}




//________________________________________________________________________________________________________________
void SimpleLednickyFitter::LoadTransformMatrices(int aRebin, TString aFileLocation)
{
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

}

//________________________________________________________________________________________________________________
TH2D* SimpleLednickyFitter::GetTransformMatrix(int aIndex, int aRebin, TString aFileLocation)
{
  if(fTransformMatrices.size()==0) LoadTransformMatrices(aRebin, aFileLocation);
  return fTransformMatrices[aIndex];
}

//________________________________________________________________________________________________________________
vector<TH2D*> SimpleLednickyFitter::GetTransformMatrices(int aRebin, TString aFileLocation)
{
  if(fTransformMatrices.size()==0) LoadTransformMatrices(aRebin, aFileLocation);
  return fTransformMatrices;
}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::InitiateResidualCollection(td1dVec &aKStarBinCenters, ChargedResidualsType aChargedResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, TString aInterpCfsDirectory)
{
  vector<TH2D*> aTransformMatrices = GetTransformMatrices();
  vector<AnalysisType> aTransformStorageMapping = GetTransformStorageMapping();
  fResidualCollection = new ResidualCollection(fAnalysisType, fIncludeResidualsType, aResPrimMaxDecayType, aKStarBinCenters, aTransformMatrices, aTransformStorageMapping, k0010);
  fResidualCollection->SetChargedResidualsType(aInterpCfsDirectory, aChargedResidualsType);

  double tSigStRadiusFactor = 1.;
  fResidualCollection->SetRadiusFactorForSigStResiduals(tSigStRadiusFactor);
}


//________________________________________________________________________________________________________________
vector<double> SimpleLednickyFitter::GetFitCfIncludingResiduals(vector<double> &aPrimaryFitCfContent, double *aParamSet)
{
  td1dVec tFitCfContent = CombinePrimaryWithResiduals(aParamSet, aPrimaryFitCfContent);

  return tFitCfContent;
}

