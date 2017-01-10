///////////////////////////////////////////////////////////////////////////
// LednickyFitter:                                                       //
///////////////////////////////////////////////////////////////////////////


#include "LednickyFitter.h"

#ifdef __ROOT__
ClassImp(LednickyFitter)
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
LednickyFitter::LednickyFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):
  fVerbose(true),
  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),
  fFits(fNAnalyses),
  fMaxFitKStar(aMaxFitKStar),
  fRejectOmega(false),
  fApplyNonFlatBackgroundCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fApplyMomResCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fIncludeResidualCorrelations(false),  //TODO change deault to true here AND in CoulombFitter
  fResidualsInitiated(false),
  fReturnPrimaryWithResidualsToAnalyses(false),

  fResXiCK(),
  fResOmegaK(),
  fResAXiCK(),
  fResAOmegaK(),

  fChi2(0),
  fChi2GlobalMin(1000000000),
  fChi2Vec(fNAnalyses),
  fNpFits(0),
  fNpFitsVec(fNAnalyses),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  AnalysisType tAnType = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType();
  TString tFilesLocationBase = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/CoulombFitter/";
  TString tInterpLocationBase_XiCK, tInterpLocationBase_OmegaK;
  TString tHFcnLocationBase_XiCK, tHFcnLocationBase_OmegaK;
  switch(tAnType) {
  case kLamKchP:
  case kALamKchM:
    tInterpLocationBase_XiCK = tFilesLocationBase + TString("InterpHistsAttractive");
    tHFcnLocationBase_XiCK = tFilesLocationBase + TString("LednickyHFunction");

    tInterpLocationBase_OmegaK = tFilesLocationBase + TString("InterpHists_OmegaKchP");
    tHFcnLocationBase_OmegaK = tFilesLocationBase + TString("LednickyHFunction_OmegaKchP");

    fResXiCK = new ChargedResidualCf(kXiCKchP,tInterpLocationBase_XiCK,tHFcnLocationBase_XiCK);
    fResAXiCK = new ChargedResidualCf(kAXiCKchM,tInterpLocationBase_XiCK,tHFcnLocationBase_XiCK);

    fResOmegaK = new ChargedResidualCf(kOmegaKchP,tInterpLocationBase_OmegaK,tHFcnLocationBase_OmegaK);
    fResAOmegaK = new ChargedResidualCf(kAOmegaKchM,tInterpLocationBase_OmegaK,tHFcnLocationBase_OmegaK);
    break;

  case kLamKchM:
  case kALamKchP:
    tInterpLocationBase_XiCK = tFilesLocationBase + TString("InterpHistsRepulsive");
    tHFcnLocationBase_XiCK = tFilesLocationBase + TString("LednickyHFunction");

    tInterpLocationBase_OmegaK = tFilesLocationBase + TString("InterpHists_OmegaKchM");
    tHFcnLocationBase_OmegaK = tFilesLocationBase + TString("LednickyHFunction_OmegaKchM");

    fResXiCK = new ChargedResidualCf(kXiCKchM,tInterpLocationBase_XiCK,tHFcnLocationBase_XiCK);
    fResAXiCK = new ChargedResidualCf(kAXiCKchP,tInterpLocationBase_XiCK,tHFcnLocationBase_XiCK);

    fResOmegaK = new ChargedResidualCf(kOmegaKchM,tInterpLocationBase_OmegaK,tHFcnLocationBase_OmegaK);
    fResAOmegaK = new ChargedResidualCf(kAOmegaKchP,tInterpLocationBase_OmegaK,tHFcnLocationBase_OmegaK);
    break;


  default:
    cout << "ERROR: LednickyFitter::LednickyFitter  tAnType = " << tAnType << " is not apropriate" << endl << endl;
    assert(0);
  }

  
}


//________________________________________________________________________________________________________________
LednickyFitter::~LednickyFitter()
{
  cout << "LednickyFitter object is being deleted!!!" << endl;
}

//________________________________________________________________________
double LednickyFitter::GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double LednickyFitter::GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double LednickyFitter::LednickyEq(double *x, double *par)
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
void LednickyFitter::PrintCurrentParamValues(int &aNpar, double* aPar)
{
  for(int i=0; i<aNpar; i++) cout << "par[" << i << "] = " << aPar[i] << endl;
  cout << endl;
}


//________________________________________________________________________________________________________________
bool LednickyFitter::AreParamsSame(double *aCurrent, double *aNew, int aNEntries)
{
  bool tAreSame = true;
  for(int i=0; i<aNEntries; i++)
  {
    if(abs(aCurrent[i]-aNew[i]) > std::numeric_limits< double >::min()) tAreSame = false;
  }

  if(!tAreSame)
  {
    for(int i=0; i<aNEntries; i++) aCurrent[i] = aNew[i];
  }

  return tAreSame;
}

//________________________________________________________________________________________________________________
double* LednickyFitter::AdjustLambdaParam(double *aParamSet, double aNewLambda, int aNEntries)
{
  double *tReturnArray = new double[aNEntries];
  tReturnArray[0] = aNewLambda;
  for(int i=1; i<aNEntries; i++) tReturnArray[i] = aParamSet[i];

  return tReturnArray;
}



//________________________________________________________________________________________________________________
void LednickyFitter::ApplyNonFlatBackgroundCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TF1* aNonFlatBgd)
{
  assert(aCf.size() == aKStarBinCenters.size());
  for(unsigned int i=0; i<aCf.size(); i++)
  {
    aCf[i] = aCf[i]*aNonFlatBgd->Eval(aKStarBinCenters[i]);
  }
}


//________________________________________________________________________________________________________________
vector<double> LednickyFitter::ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix)
{
  //TODO probably rebin aMomResMatrix to match bin size of aCf
  //TODO do in both this AND CoulombFitter

  unsigned int tKStarRecBin, tKStarTrueBin;
  double tKStarRec, tKStarTrue;
  assert(aCf.size() == aKStarBinCenters.size());
  assert(aCf.size() == (unsigned int)aMomResMatrix->GetNbinsX());
  assert(aCf.size() == (unsigned int)aMomResMatrix->GetNbinsY());

  vector<double> tReturnCf(aCf.size(),0.);
  vector<double> tNormVec(aCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<aCf.size(); i++)
  {
    tKStarRec = aKStarBinCenters[i];
    tKStarRecBin = aMomResMatrix->GetYaxis()->FindBin(tKStarRec);

    for(unsigned int j=0; j<aCf.size(); j++)
    {
      tKStarTrue = aKStarBinCenters[j];
      tKStarTrueBin = aMomResMatrix->GetXaxis()->FindBin(tKStarTrue);

      tReturnCf[i] += aCf[j]*aMomResMatrix->GetBinContent(tKStarTrueBin,tKStarRecBin);
      tNormVec[i] += aMomResMatrix->GetBinContent(tKStarTrueBin,tKStarRecBin);
    }
    tReturnCf[i] /= tNormVec[i];
  }
  return tReturnCf;
}


//________________________________________________________________________________________________________________
vector<double> LednickyFitter::GetNeutralResidualCorrelation(double *aParentCfParams, vector<double> &aKStarBinCenters, TH2* aTransformMatrix)
{
  vector<double> tParentCf(aKStarBinCenters.size(),0.);
  double tKStar[1];
  for(unsigned int i=0; i<aKStarBinCenters.size(); i++)
  {
    tKStar[0] = aKStarBinCenters[i];
    tParentCf[i] = LednickyEq(tKStar,aParentCfParams);
  }

  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(tParentCf.size() == aKStarBinCenters.size());
  assert(tParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsX());
  assert(tParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsY());

  vector<double> tReturnResCf(tParentCf.size(),0.);
  vector<double> tNormVec(tParentCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<tParentCf.size(); i++)
  {
    tDaughterPairKStar = aKStarBinCenters[i];
    tDaughterPairKStarBin = aTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

    for(unsigned int j=0; j<tParentCf.size(); j++)
    {
      tParentPairKStar = aKStarBinCenters[j];
      tParentPairKStarBin = aTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tReturnResCf[i] += tParentCf[j]*aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec[i] += aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    tReturnResCf[i] /= tNormVec[i];
  }
  return tReturnResCf;
}

//________________________________________________________________________________________________________________
vector<double> LednickyFitter::GetChargedResidualCorrelation(ResidualType aResidualType, double *aParentCfParams, vector<double> &aKStarBinCenters, bool aUseExpXiData)
{
  td1dVec tReturnCfVec;

  switch(aResidualType) {
  case kXiCKchP:
  case kXiCKchM:
    tReturnCfVec = fResXiCK->GetCoulombResidualCorrelation(aParentCfParams,aKStarBinCenters,aUseExpXiData);
    break;

  case kAXiCKchP:
  case kAXiCKchM:
    tReturnCfVec = fResAXiCK->GetCoulombResidualCorrelation(aParentCfParams,aKStarBinCenters,aUseExpXiData);
    break;

  case kOmegaKchP:
  case kOmegaKchM:
    tReturnCfVec = fResOmegaK->GetCoulombResidualCorrelation(aParentCfParams,aKStarBinCenters,aUseExpXiData);
    break;

  case kAOmegaKchP:
  case kAOmegaKchM:
    tReturnCfVec = fResAOmegaK->GetCoulombResidualCorrelation(aParentCfParams,aKStarBinCenters,aUseExpXiData);
    break;


  default:
    cout << "ERROR: LednickyFitter::GetChargedResidualCorrelation  aResidualType = " << aResidualType << " is not apropriate" << endl << endl;
    assert(0);
  }


  return tReturnCfVec;
}

//________________________________________________________________________________________________________________
vector<double> LednickyFitter::CombinePrimaryWithResiduals(td1dVec &aLambdaValues, td2dVec &aCfs)
{
  assert(aLambdaValues.size()==aCfs.size());
  for(unsigned int i=1; i<aCfs.size(); i++) assert(aCfs[i-1].size()==aCfs[i].size());

  vector<double> tReturnCf(aCfs[0].size(),0.);
  for(unsigned int iBin=0; iBin<tReturnCf.size(); iBin++)
  {
    tReturnCf[iBin] = 1.;
    for(unsigned int iCf=0; iCf<aCfs.size(); iCf++)
    {
      //NOTE:  //TODO confusing definitions of Cf and whatnot in Jai's analysis
      if(aCfs[iCf][iBin] > 0.) tReturnCf[iBin] += aLambdaValues[iCf]*((aCfs[iCf][iBin]-1.0)/aLambdaValues[iCf]);
    }
  }
  return tReturnCf;
}

//________________________________________________________________________________________________________________
void LednickyFitter::ApplyNormalization(double aNorm, td1dVec &aCf)
{
  for(unsigned int i=0; i<aCf.size(); i++) aCf[i] *= aNorm;
}




//________________________________________________________________________________________________________________
double LednickyFitter::GetChi2Value(int aKStarBin, TH1* aCfToFit, double* aPar)
{
    double tKStar[1];
    tKStar[0] = aCfToFit->GetXaxis()->GetBinCenter(aKStarBin);
    double tChi = (aCfToFit->GetBinContent(aKStarBin) - LednickyEq(tKStar,aPar))/aCfToFit->GetBinError(aKStarBin);
    return tChi*tChi;
}




//________________________________________________________________________________________________________________
double LednickyFitter::GetPmlValue(double aNumContent, double aDenContent, double aCfContent)
{
  double tTerm1 = aNumContent*log(  (aCfContent*(aNumContent+aDenContent)) / (aNumContent*(aCfContent+1))  );
  double tTerm2 = aDenContent*log(  (aNumContent+aDenContent) / (aDenContent*(aCfContent+1))  );
  double tChi2PML = -2.0*(tTerm1+tTerm2);
  return tChi2PML;
}


//________________________________________________________________________________________________________________
void LednickyFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
{
  if(fVerbose) PrintCurrentParamValues(npar,par);


  //---------------------------------------------------------


  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  int tNFitParPerAnalysis = 5;
//  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
//  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;


  int tNbinsXToBuildMomResCrctn=0, tNbinsXToBuildResiduals=0;
  int tNbinsXToBuildGlobal;  // when applying momentum resolution corrections, many times you must go beyond fitting range to apply correction
  if(fApplyMomResCorrection) tNbinsXToBuildMomResCrctn = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetModelKStarTrueVsRecMixed()->GetNbinsX();
  if(fIncludeResidualCorrelations) tNbinsXToBuildResiduals = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetTransformMatrices()[0]->GetNbinsX();
  tNbinsXToBuildGlobal = std::max({tNbinsXToBuildMomResCrctn, tNbinsXToBuildResiduals, tNbinsXToFitGlobal});

//  vector<double> tPrimaryFitCfContentUnNorm(tNbinsXToBuildGlobal,0.);
  vector<double> tPrimaryFitCfContent(tNbinsXToBuildGlobal,0.);
  vector<double> tNumContent(tNbinsXToBuildGlobal,0.);
  vector<double> tDenContent(tNbinsXToBuildGlobal,0.);
  vector<double> tKStarBinCenters(tNbinsXToBuildGlobal,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    int tNbinsXToBuild;
    TH2* tMomResMatrix = NULL;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
      tNbinsXToBuild = tMomResMatrix->GetNbinsX();
      assert(tNbinsXToBuild == tNbinsXToBuildGlobal);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();
      TH1* tCf = tKStarCfLite->Cf();

      assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tCf->GetXaxis()->GetBinWidth(1));
      //make sure tNum and tDen and tCf have same bin size as tMomResMatrix
      if(fApplyMomResCorrection)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetYaxis()->GetBinWidth(1));
      }
      //make sure tNum and tDen and tCf have same bin size as residuals
      if(fIncludeResidualCorrelations)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrices()[0]->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrices()[0]->GetYaxis()->GetBinWidth(1));
      }

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();

      int tNbinsX = tNum->GetNbinsX();

      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams = tNFitParPerAnalysis+1);

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();


      assert(tNFitParams == 6);
      //NOTE: CANNOT use sizeof(tPar)/sizeof(tPar[0]) trick here becasue tPar is pointer
      double *tPar = new double[tNFitParams];
      double tOverallLambda = par[tLambdaMinuitParamNumber];
      //tPar[0] = par[tLambdaMinuitParamNumber];
      if(fIncludeResidualCorrelations) tPar[0] = 0.25*tOverallLambda;
      else tPar[0] = tOverallLambda;
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateFitFunction, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double x[1];
      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();
      //bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);

      if(!fApplyMomResCorrection && !fIncludeResidualCorrelations) tNbinsXToBuild = tNbinsXToFit;
      for(int ix=1; ix <= tNbinsXToBuild; ix++)
      {
        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);
        x[0] = tKStarBinCenters[ix-1];

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

        tPrimaryFitCfContent[ix-1] = LednickyEq(x,tPar);
      }

      vector<double> tFitCfContent;
      if(fIncludeResidualCorrelations) 
      {
        double tLambda_SigK = 0.20*tOverallLambda;  //for now, primary lambda scaled by some factor
        double *tPar_SigK = AdjustLambdaParam(tPar,tLambda_SigK,tNFitParams);
        td1dVec tResidual_SigK = GetNeutralResidualCorrelation(tPar_SigK,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[0]);

        double tLambda_Xi0K = 0.13*tOverallLambda;  //for now, primary lambda scaled by some factor
        double *tPar_Xi0K = AdjustLambdaParam(tPar,tLambda_Xi0K,tNFitParams);
        td1dVec tResidual_Xi0K = GetNeutralResidualCorrelation(tPar_Xi0K,tKStarBinCenters,tFitPairAnalysis->GetTransformMatrices()[2]);

        AnalysisType tAnType = tFitPartialAnalysis->GetAnalysisType();
        ResidualType tResXiCKType, tResOmegaKType;
        switch(tAnType) {
        case kLamKchP:
          tResXiCKType = kXiCKchP;
          tResOmegaKType = kOmegaKchP;
          break;

        case kLamKchM:
          tResXiCKType = kXiCKchM;
          tResOmegaKType = kOmegaKchM;
          break;

        case kALamKchP:
          tResXiCKType = kAXiCKchP;
          tResOmegaKType = kAOmegaKchP;
          break;

        case kALamKchM:
          tResXiCKType = kAXiCKchM;
          tResOmegaKType = kAOmegaKchM;
          break;

        default:
          cout << "ERROR: LednickyFitter::LednickyFitter  tAnType = " << tAnType << " is not apropriate" << endl << endl;
          assert(0);
        }

        bool tUseExpXiData = false;
        if(tFitPartialAnalysis->GetCentralityType()==k0010) tUseExpXiData=true;

        double tLambda_XiCK = 0.12*tOverallLambda;  //for now, primary lambda scaled by some factor
        double *tPar_XiCK = AdjustLambdaParam(tPar,tLambda_XiCK,tNFitParams);
        td1dVec tResidual_XiCK = GetChargedResidualCorrelation(tResXiCKType,tPar_XiCK,tKStarBinCenters,tUseExpXiData);

        double tLambda_OmegaK = 0.01*tOverallLambda;  //for now, primary lambda scaled by some factor
        double *tPar_OmegaK = AdjustLambdaParam(tPar,tLambda_OmegaK,tNFitParams);
        td1dVec tResidual_OmegaK = GetChargedResidualCorrelation(tResOmegaKType,tPar_OmegaK,tKStarBinCenters,tUseExpXiData);

        vector<double> tLambdas{tPar[0],tLambda_SigK,tLambda_Xi0K,tLambda_XiCK,tLambda_OmegaK};
        td2dVec tAllCfs{tPrimaryFitCfContent,tResidual_SigK,tResidual_Xi0K,tResidual_XiCK,tResidual_OmegaK};
        tFitCfContent = CombinePrimaryWithResiduals(tLambdas, tAllCfs);
        if(fReturnPrimaryWithResidualsToAnalyses) tFitPairAnalysis->SetPrimaryWithResiduals(tFitCfContent);

        delete[] tPar_SigK;
        delete[] tPar_Xi0K;
        delete[] tPar_XiCK;
        delete[] tPar_OmegaK;
      }
      else tFitCfContent = tPrimaryFitCfContent;

      vector<double> tCorrectedFitCfContent;
      if(fApplyMomResCorrection) tCorrectedFitCfContent = ApplyMomResCorrection(tFitCfContent, tKStarBinCenters, tMomResMatrix);
      else tCorrectedFitCfContent = tFitCfContent;

      if(fApplyNonFlatBackgroundCorrection)
      {
        TF1* tNonFlatBgd = tFitPartialAnalysis->GetNonFlatBackground(/*0.40,0.90*/);
        ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, tKStarBinCenters, tNonFlatBgd);
      }

      ApplyNormalization(tPar[5], tCorrectedFitCfContent);

      for(int ix=0; ix < tNbinsXToFit; ix++)
      {
        if(tRejectOmega && (tKStarBinCenters[ix] > tRejectOmegaLow) && (tKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tChi2 = 0.;
            if(fFitSharedAnalyses->GetFitType() == kChi2PML) tChi2 = GetPmlValue(tNumContent[ix],tDenContent[ix],tCorrectedFitCfContent[ix]);
            else if(fFitSharedAnalyses->GetFitType() == kChi2) tChi2 = GetChi2Value(ix+1,tCf,tPar);
            else tChi2 = 0.;

            fChi2Vec[iAnaly] += tChi2;
            fChi2 += tChi2;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }

      }

      delete[] tPar;
    }

  }

//  delete[] tCurrentFitPar;

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
/*
  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
*/
}

//________________________________________________________________________________________________________________
TF1* LednickyFitter::CreateFitFunction(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);

  int tNFitParams = tFitPairAnalysis->GetNFitParams(); //should be equal to 5
  TF1* ReturnFunction = new TF1(aName,LednickyEq,0.,0.5,tNFitParams+1);
  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    ReturnFunction->SetParameter(iPar,tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValue());
    ReturnFunction->SetParError(iPar,tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValueError());
  }

  ReturnFunction->SetParameter(5,1.);
  ReturnFunction->SetParError(5,0.);

  ReturnFunction->SetChisquare(fChi2);
  ReturnFunction->SetNDF(fNDF);

  ReturnFunction->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");
  fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber)->GetKStarCfHeavy()->GetHeavyCf()->GetListOfFunctions()->Add(ReturnFunction);
  fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber)->GetKStarCf()->GetListOfFunctions()->Add(ReturnFunction);

  return ReturnFunction;
}



//________________________________________________________________________________________________________________
void LednickyFitter::DoFit()
{
  cout << "*****************************************************************************" << endl;
  //cout << "Starting to fit " << fCfName << endl;
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
  arglist[1] = 0.001;
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

  int tNParams = fFitSharedAnalyses->GetNMinuitParams();

  fNDF = fNpFits-fNvpar;


  double *tPar = new double[tNParams];
  //get result
  for(int i=0; i<tNParams; i++)
  {
    double tempMinParam;
    double tempParError;
    fMinuit->GetParameter(i,tempMinParam,tempParError);
    
    fMinParams.push_back(tempMinParam);
    fParErrors.push_back(tempParError);

    tPar[i] = tempMinParam;
  }
  if(fErrFlg==0)
  {
    int tNpar;
    double tChi2;
    fReturnPrimaryWithResidualsToAnalyses = true;
    CalculateFitFunction(tNpar,tChi2,tPar);
    fReturnPrimaryWithResidualsToAnalyses = false;
  }
  delete[] tPar;

  fFitSharedAnalyses->SetMinuitMinParams(fMinParams);
  fFitSharedAnalyses->SetMinuitParErrors(fParErrors);
  fFitSharedAnalyses->ReturnFitParametersToAnalyses();

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetFit(CreateFitFunction("fit",iAnaly));
  }

}


//________________________________________________________________________________________________________________
vector<double> LednickyFitter::FindGoodInitialValues()
{
  fMinuit->SetPrintLevel(-1); //quiet mode

  double tChi2 = 100000;
  double testChi2;

  int tNParams = fMinuit->GetNumPars();

  vector<double> ReturnGoodValues(tNParams);
  for(int i=0; i<tNParams; i++) {ReturnGoodValues[i] = 0.;}

  vector<double> tParamErrors(tNParams);
  for(int i=0; i<tNParams; i++) {tParamErrors[i] = 0.;}

  //---------------------------------------------------------------
  const int nValuesLambda = 3;
  vector<double> tLambdaValues(nValuesLambda);
    tLambdaValues[0] = 0.1;
    tLambdaValues[1] = 0.2;
    tLambdaValues[2] = 0.5;
  
  const int nValuesRadius = 3;
  vector<double> tRadiusValues(nValuesRadius);
    tRadiusValues[0] = 3.;
    tRadiusValues[1] = 4.;
    tRadiusValues[2] = 5.;

  const int nValuesRef0 = 6;
  vector<double> tRef0Values(nValuesRef0);
    tRef0Values[0] = -1.;
    tRef0Values[1] = -0.5;
    tRef0Values[2] = -0.1;
    tRef0Values[3] = 0.1;
    tRef0Values[4] = 0.5;
    tRef0Values[5] = 1.;

  const int nValuesImf0 = 6;
  vector<double> tImf0Values(nValuesImf0);
    tImf0Values[0] = -1.;
    tImf0Values[1] = -0.5;
    tImf0Values[2] = -0.1;
    tImf0Values[3] = 0.1;
    tImf0Values[4] = 0.5;
    tImf0Values[5] = 1.;

  const int nValuesd0 = 1;
  vector<double> td0Values(nValuesd0);
//    td0Values[0] = -10.;
//    td0Values[1] = -1.;
//    td0Values[2] = -0.1;
    td0Values[0] = 0.;
//    td0Values[4] = 0.1;
//    td0Values[5] = 1.;
//    td0Values[6] = 10.;

  const int nValuesNorm = 1;
  vector<double> tNormValues(nValuesNorm);
    tNormValues[0] = 0.1;

  //---------------------------------------------------------------

  TString *tCpnam = fMinuit->fCpnam;

  vector<int> tIndices;
  tIndices.resize(tNParams);
  for(unsigned int i=0; i<tIndices.size(); i++) {tIndices[i] = 0;}

  vector<int> tMaxIndices;
  tMaxIndices.resize(tNParams);

  vector<vector<double> > tStartValuesMatrix(tNParams);

  for(int i=0; i<tNParams; i++)
  {
    if(tCpnam[i] == "Lambda") 
    {
      tStartValuesMatrix[i] = tLambdaValues;
      tMaxIndices[i] = nValuesLambda;
    }

    else if(tCpnam[i] == "Radius") 
    {
      tStartValuesMatrix[i] = tRadiusValues;
      tMaxIndices[i] = nValuesRadius;
    }

    else if(tCpnam[i] == "Ref0") 
    {
      tStartValuesMatrix[i] = tRef0Values;
      tMaxIndices[i] = nValuesRef0;
    }

    else if(tCpnam[i] == "Imf0") 
    {
      tStartValuesMatrix[i] = tImf0Values;
      tMaxIndices[i] = nValuesImf0;
    }

    else if(tCpnam[i] == "d0") 
    {
      tStartValuesMatrix[i] = td0Values;
      tMaxIndices[i] = nValuesd0;
    }

    else if(tCpnam[i] == "Norm") 
    {
      tStartValuesMatrix[i] = tNormValues;
      tMaxIndices[i] = nValuesNorm;
    }

    else{cout << "ERROR in LednickyFitter::FindGoodInitialValues(): Parameter has incorrect name!!!" << endl;}
  }

  //------------------------------------------------------------------------
  int tCounter = 0;

  double tArgList[2];
  int tErrFlg = 0;

  while(tIndices[tNParams-1] < tMaxIndices[tNParams-1])
  {
    for(int i=0; i<tNParams; i++)
    {
      tArgList[0] = i+1;  //because Minuit numbering starts at 1, not 0!
      tArgList[1] = tStartValuesMatrix[i][tIndices[i]];
      fMinuit->mnexcm("SET PAR",tArgList,2,tErrFlg);
    }

    DoFit();
    tCounter++;
    cout << "tCounter = " << tCounter << endl << endl << endl << endl;

    testChi2 = fChi2;
    if(testChi2 < tChi2)
    {
      if(fErrFlg == 0)
      {
        tChi2 = testChi2;
        for(int i=0; i<tNParams; i++) {fMinuit->GetParameter(i,ReturnGoodValues[i],tParamErrors[i]);}
      }
    }

    tIndices[0]++;
    for(unsigned int i=0; i<tIndices.size(); i++)
    {
      if(tIndices[i] == tMaxIndices[i])
      {
        if(i == tIndices.size()-1) {continue;}
        else
        {
          tIndices[i] = 0;
          tIndices[i+1]++;
        }
      }
    }

    cout << "tIndices = " << endl;
    for(unsigned int i=0; i<tIndices.size(); i++) {cout << tIndices[i] << endl;}

  }

  cout << "Chi2 from ideal initial values = " << tChi2 << endl;
  ReturnGoodValues.push_back(tChi2);

  cout << "tCounter = " << tCounter << endl << endl << endl << endl;

  return ReturnGoodValues;
}

