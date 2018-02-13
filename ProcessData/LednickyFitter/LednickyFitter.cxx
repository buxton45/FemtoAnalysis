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
  fVerbose(false),

  fSaveLocationBase(""),
  fSaveNameModifier(""),

  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),
  fCorrectedFitVecs(0),

  fMaxFitKStar(aMaxFitKStar),
  fNbinsXToFit(0),

  fMaxBuildKStar(0.0),
  fNbinsXToBuild(0),

  fKStarBinWidth(0.),
  fKStarBinCenters(0),
  fRejectOmega(false),
  fApplyNonFlatBackgroundCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fApplyMomResCorrection(false), //TODO change deault to true here AND in CoulombFitter

  fIncludeResidualsType(kIncludeNoResiduals), //TODO change deault to true here AND in CoulombFitter
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fResidualsInitiated(false),
  fReturnPrimaryWithResidualsToAnalyses(false),
  fNonFlatBgdFitType(kLinear),

  fUsemTScalingOfResidualRadii(false),
  fmTScalingPowerOfResidualRadii(-0.5),

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
  int tNFitPartialAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetNFitPartialAnalysis();
  fCorrectedFitVecs.resize(fNAnalyses, td2dVec(tNFitPartialAnalysis));
}

//________________________________________________________________________________________________________________
LednickyFitter::LednickyFitter(AnalysisType aAnalysisType, double aMaxBuildKStar, double aKStarBinWidth):
  fVerbose(false),

  fSaveLocationBase(""),
  fSaveNameModifier(""),

  fFitSharedAnalyses(nullptr),
  fMinuit(nullptr),
  fNAnalyses(0),
  fCorrectedFitVecs(0),

  fMaxFitKStar(aMaxBuildKStar),
  fNbinsXToFit(0),

  fMaxBuildKStar(aMaxBuildKStar),
  fNbinsXToBuild(0),

  fKStarBinWidth(aKStarBinWidth),
  fKStarBinCenters(0),
  fRejectOmega(false),
  fApplyNonFlatBackgroundCorrection(false), //TODO change deault to true here AND in CoulombFitter
  fApplyMomResCorrection(false), //TODO change deault to true here AND in CoulombFitter

  fIncludeResidualsType(kIncludeNoResiduals), //TODO change deault to true here AND in CoulombFitter
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fResidualsInitiated(false),
  fReturnPrimaryWithResidualsToAnalyses(false),
  fNonFlatBgdFitType(kLinear),

  fUsemTScalingOfResidualRadii(false),
  fmTScalingPowerOfResidualRadii(-0.5),

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
  fNAnalyses=1;

  fNbinsXToFit = std::round(fMaxFitKStar/fKStarBinWidth);
  fNbinsXToBuild = fNbinsXToFit;
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
void LednickyFitter::AppendFitInfo(TString &aSaveName, bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, 
                                          IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, ChargedResidualsType aChargedResidualsType, bool aFixD0)
{
  if(aApplyMomResCorrection) aSaveName += TString("_MomResCrctn");
  if(aApplyNonFlatBackgroundCorrection) aSaveName += TString("_NonFlatBgdCrctn");

  aSaveName += cIncludeResidualsTypeTags[aIncludeResidualsType];
  if(aIncludeResidualsType != kIncludeNoResiduals)
  {
    aSaveName += cResPrimMaxDecayTypeTags[aResPrimMaxDecayType];
    aSaveName += cChargedResidualsTypeTags[aChargedResidualsType];
  }
  if(aFixD0) aSaveName += TString("_FixedD0");
}


//________________________________________________________________________________________________________________
void LednickyFitter::PrintCurrentParamValues(int aNpar, double* aPar)
{
  for(int i=0; i<aNpar; i++) cout << "par[" << i << "] = " << aPar[i] << endl;
  cout << endl;
}

//________________________________________________________________________________________________________________
double LednickyFitter::GetChi2Value(int aKStarBin, TH1* aCfToFit, double* aPar)
{
//TODO this is dated, and only works if no corrections (i.e. no momentum resolution, non-flat bgd, residuals, etc.)
  double tKStar[1];
  tKStar[0] = aCfToFit->GetXaxis()->GetBinCenter(aKStarBin);
  double tChi = (aCfToFit->GetBinContent(aKStarBin) - LednickyEq(tKStar,aPar))/aCfToFit->GetBinError(aKStarBin);
  return tChi*tChi;
}

//________________________________________________________________________________________________________________
double LednickyFitter::GetChi2Value(int aKStarBin, TH1* aCfToFit, double aFitCfContent)
{
  double tChi = (aCfToFit->GetBinContent(aKStarBin) - aFitCfContent)/aCfToFit->GetBinError(aKStarBin);
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
void LednickyFitter::ApplyNormalization(double aNorm, td1dVec &aCf)
{
  for(unsigned int i=0; i<aCf.size(); i++) aCf[i] *= aNorm;
}



//________________________________________________________________________________________________________________
vector<double> LednickyFitter::GetFitCfIncludingResiduals(FitPairAnalysis* aFitPairAnalysis, vector<double> &aPrimaryFitCfContent, double *aParamSet)
{
  td1dVec tFitCfContent = aFitPairAnalysis->CombinePrimaryWithResiduals(aParamSet, aPrimaryFitCfContent);
  if(fReturnPrimaryWithResidualsToAnalyses) aFitPairAnalysis->SetPrimaryWithResiduals(tFitCfContent);

  return tFitCfContent;
}

//________________________________________________________________________________________________________________
void LednickyFitter::CalculateFitFunction(int &npar, double &chi2, double *par)
{
  if(fVerbose) PrintCurrentParamValues(fFitSharedAnalyses->GetNMinuitParams(),par);
  //---------------------------------------------------------
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  int tNFitParPerAnalysis = 5;

  vector<double> tPrimaryFitCfContent(fNbinsXToBuild,0.);
  vector<double> tNumContent(fNbinsXToBuild,0.);
  vector<double> tDenContent(fNbinsXToBuild,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    TH2* tMomResMatrix = NULL;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();
      TH1* tCf = tKStarCfLite->Cf();

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
      double *tParPrim = new double[tNFitParams];


//      if(fIncludeResidualsType != kIncludeNoResiduals) tParPrim[0] = cAnalysisLambdaFactors[tFitPairAnalysis->GetAnalysisType()]*par[tLambdaMinuitParamNumber];
      if(fIncludeResidualsType != kIncludeNoResiduals) tParPrim[0] = cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][tFitPairAnalysis->GetAnalysisType()]*par[tLambdaMinuitParamNumber];
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
      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();

      vector<double> tFitCfContent;
      vector<double> tCorrectedFitCfContent;

      for(int ix=1; ix <= fNbinsXToBuild; ix++)
      {
        x[0] = fKStarBinCenters[ix-1];

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

        tPrimaryFitCfContent[ix-1] = LednickyEq(x,tParPrim);
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
        tFitCfContent = GetFitCfIncludingResiduals(tFitPairAnalysis, tPrimaryFitCfContent, tParOverall);
        delete[] tParOverall;
      }
      else tFitCfContent = tPrimaryFitCfContent;


      if(fApplyMomResCorrection) tCorrectedFitCfContent = ApplyMomResCorrection(tFitCfContent, fKStarBinCenters, tMomResMatrix);
      else tCorrectedFitCfContent = tFitCfContent;

      bool tNormalizeBgdFitToCf=false;
      if(fApplyNonFlatBackgroundCorrection)
      {
        TF1* tNonFlatBgd;
        //I thought using PairAnalysis, when BgdFitType != kLinear, would help stabilize things, but it doesn't seem to help all that much.
        //  Things have been stabilized with other tweaks.
        tNonFlatBgd = tFitPartialAnalysis->GetNonFlatBackground(fNonFlatBgdFitType, fFitSharedAnalyses->GetFitType(), tNormalizeBgdFitToCf);
        ApplyNonFlatBackgroundCorrection(tCorrectedFitCfContent, fKStarBinCenters, tNonFlatBgd);
      }

      fCorrectedFitVecs[iAnaly][iPartAn] = tCorrectedFitCfContent;
      ApplyNormalization(tParPrim[5], tCorrectedFitCfContent);
      if(fApplyNonFlatBackgroundCorrection && fFitSharedAnalyses->GetFitType()==kChi2PML && !tNormalizeBgdFitToCf)
      {
        //In this case, ApplyNonFlatBackgroundCorrection essentially takes care of the normalization, since it fits raw Num and Den
        // ApplyNormalization applies a normalization that is very close to 1.  Therefore, for the plots in fCorrectedFitVecs to look pretty,
        // I must scale them back up to around unity...also need to apply normalization = tParPrim[5]
        //TODO make this more general (to be sure, inclusion of tParPrim here is almost certainly correct)
        //  Would simply putting ApplyNormalization(tParPrim[5], tCorrectedFitCfContent); before fCorrectedFitVecs[iAnaly][iPartAn] = tCorrectedFitCfContent; 
        //  solve this FOR ALL CASES?
        ApplyNormalization(tParPrim[5]*(tKStarCfLite->GetDenScale()/tKStarCfLite->GetNumScale()), fCorrectedFitVecs[iAnaly][iPartAn]);
      }

      for(int ix=0; ix < fNbinsXToFit; ix++)
      {
        if(tRejectOmega && (fKStarBinCenters[ix] > tRejectOmegaLow) && (fKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCorrectedFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tChi2 = 0.;
            if(fFitSharedAnalyses->GetFitType() == kChi2PML) tChi2 = GetPmlValue(tNumContent[ix],tDenContent[ix],tCorrectedFitCfContent[ix]);
            else if(fFitSharedAnalyses->GetFitType() == kChi2) tChi2 = GetChi2Value(ix+1,tCf,tCorrectedFitCfContent[ix]);
            else tChi2 = 0.;

            fChi2Vec[iAnaly] += tChi2;
            fChi2 += tChi2;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }

      }
      delete[] tParPrim;
    }
  }

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
void LednickyFitter::CalculateFitFunctionOnce(int &npar, double &chi2, double *par, double *parErr, double aChi2, int aNDF)
{
  InitializeFitter();
  LednickyFitter::CalculateFitFunction(npar, chi2, par);

  double tFitParams[5];
  double tFitParamErrs[5];

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    int tLambdaParamNumber = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(0)->GetFitParameter(kLambda)->GetMinuitParamNumber();
    int tRadiusParamNumber = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(0)->GetFitParameter(kRadius)->GetMinuitParamNumber();
    int tRef0ParamNumber = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(0)->GetFitParameter(kRef0)->GetMinuitParamNumber();
    int tImf0ParamNumber = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(0)->GetFitParameter(kImf0)->GetMinuitParamNumber();
    int td0ParamNumber = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(0)->GetFitParameter(kd0)->GetMinuitParamNumber();

    tFitParams[0] = par[tLambdaParamNumber];
    tFitParamErrs[0] = parErr[tLambdaParamNumber];

    tFitParams[1] = par[tRadiusParamNumber];
    tFitParamErrs[1] = parErr[tRadiusParamNumber];

    tFitParams[2] = par[tRef0ParamNumber];
    tFitParamErrs[2] = parErr[tRef0ParamNumber];

    tFitParams[3] = par[tImf0ParamNumber];
    tFitParamErrs[3] = parErr[tImf0ParamNumber];

    tFitParams[4] = par[td0ParamNumber];
    tFitParamErrs[4] = parErr[td0ParamNumber];

    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetPrimaryFit(CreateFitFunction(iAnaly, tFitParams, tFitParamErrs, aChi2, aNDF));
  }

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iPartAn=0; iPartAn<fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetNFitPartialAnalysis(); iPartAn++)
    {
      fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(iPartAn)->SetCorrectedFitVec(fCorrectedFitVecs[iAnaly][iPartAn]);
    }
  }
}

//________________________________________________________________________________________________________________
TF1* LednickyFitter::CreateFitFunction(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);

  int tNFitParams = tFitPairAnalysis->GetNFitParams(); //should be equal to 5
  TF1* ReturnFunction = new TF1(aName,LednickyEq,0.,0.5,tNFitParams+1);
  double tParamValue, tParamError;
  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    tParamValue = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValue();
    tParamError = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValueError();
    if(tParamType==kLambda && fIncludeResidualsType != kIncludeNoResiduals)
    {
//      tParamValue *= cAnalysisLambdaFactors[tFitPairAnalysis->GetAnalysisType()];
//      tParamError *= cAnalysisLambdaFactors[tFitPairAnalysis->GetAnalysisType()];
      tParamValue *= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][tFitPairAnalysis->GetAnalysisType()];
      tParamError *= cAnalysisLambdaFactorsArr[fIncludeResidualsType][fResPrimMaxDecayType][tFitPairAnalysis->GetAnalysisType()];
    }
    ReturnFunction->SetParameter(iPar,tParamValue);
    ReturnFunction->SetParError(iPar,tParamError);
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
TF1* LednickyFitter::CreateFitFunction(int aAnalysisNumber, double *par, double *parErr, double aChi2, int aNDF)
{
  int tNFitParams = 5;
  TF1* ReturnFunction = new TF1("fit",LednickyEq,0.,0.5,tNFitParams+1);
  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    ReturnFunction->SetParameter(iPar,par[iPar]);
    ReturnFunction->SetParError(iPar,parErr[iPar]);
  }

  ReturnFunction->SetParameter(5,1.);
  ReturnFunction->SetParError(5,0.);

  ReturnFunction->SetChisquare(aChi2);
  ReturnFunction->SetNDF(aNDF);

  ReturnFunction->SetParNames("Lambda","Radius","Ref0","Imf0","d0","Norm");
  fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber)->GetKStarCfHeavy()->GetHeavyCf()->GetListOfFunctions()->Add(ReturnFunction);
  fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber)->GetKStarCf()->GetListOfFunctions()->Add(ReturnFunction);

  return ReturnFunction;
}

//________________________________________________________________________________________________________________
void LednickyFitter::InitializeFitter()
{
  cout << "----- Initializing fitter -----" << endl;

  //First, make sure KStar fit region, KStar normalization region, and NonFlatBgd fit region do not overlap
  //TODO Is this absolutely necessary?
/*
  assert(fMaxFitKStar < fFitSharedAnalyses->GetKStarMinNorm());
  if(fApplyNonFlatBackgroundCorrection) assert(fFitSharedAnalyses->GetKStarMaxNorm() < fFitSharedAnalyses->GetMinBgdFit());
*/
  //Bare minimum, the KStar fit region and the NonFlatBgd fit region definitely should not overlap
  if(fApplyNonFlatBackgroundCorrection) assert(fMaxFitKStar < fFitSharedAnalyses->GetMinBgdFit());

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
  fNbinsXToFit = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(fNbinsXToFit) == fMaxFitKStar) fNbinsXToFit--;

  if(fApplyMomResCorrection) tNbinsXToBuildMomResCrctn = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetModelKStarTrueVsRecMixed()->GetNbinsX();
  if(fIncludeResidualsType != kIncludeNoResiduals) tNbinsXToBuildResiduals = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetTransformMatrix(fIncludeResidualsType, 0)->GetNbinsX();
  fNbinsXToBuild = std::max({tNbinsXToBuildMomResCrctn, tNbinsXToBuildResiduals, fNbinsXToFit});

  if(fKStarBinWidth==0.) fKStarBinWidth = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetXaxis()->GetBinWidth(1);
  else assert(fKStarBinWidth == fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetXaxis()->GetBinWidth(1));
  
  //-------------------------------------------------------------------------------------------

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    TH2* tMomResMatrix = NULL;
    if(fApplyMomResCorrection)
    {
      tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
      assert(tMomResMatrix);
      tTempNbinsXToBuild = tMomResMatrix->GetNbinsX();
      assert(tTempNbinsXToBuild == fNbinsXToBuild);
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
      assert(tNum->GetXaxis()->GetBinWidth(1) == fKStarBinWidth);
      //make sure tNum and tDen and tCf have same bin size as tMomResMatrix
      if(fApplyMomResCorrection)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetYaxis()->GetBinWidth(1));
      }
      //make sure tNum and tDen and tCf have same bin size as residuals
      if(fIncludeResidualsType != kIncludeNoResiduals)
      {
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrix(fIncludeResidualsType, 0)->GetXaxis()->GetBinWidth(1));
        assert(tNum->GetXaxis()->GetBinWidth(1) == tFitPairAnalysis->GetTransformMatrix(fIncludeResidualsType, 0)->GetYaxis()->GetBinWidth(1));
      }

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());
      assert(tNum->GetNbinsX() == tCf->GetNbinsX());

      tTempNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tTempNbinsXToFit) == fMaxFitKStar) tTempNbinsXToFit--;

      if(tTempNbinsXToFit > tNum->GetNbinsX()) {tTempNbinsXToFit = tNum->GetNbinsX();}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tTempNbinsXToFit == fNbinsXToFit);

      if(!fApplyMomResCorrection && fIncludeResidualsType==kIncludeNoResiduals) fNbinsXToBuild = fNbinsXToFit;

      if(iAnaly==0 && iPartAn==0)
      {
        fKStarBinCenters.resize(fNbinsXToBuild,0.);
        for(int ix=1; ix <= fNbinsXToBuild; ix++)
        {
          fKStarBinCenters[ix-1] = tNum->GetXaxis()->GetBinCenter(ix);
        }
      }
    }
  }
  fMaxBuildKStar = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetXaxis()->GetBinUpEdge(fNbinsXToBuild);

  if(fIncludeResidualsType != kIncludeNoResiduals)
  {
    for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
    {
      fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->InitiateResidualCollection(fKStarBinCenters, fIncludeResidualsType, fChargedResidualsType, fResPrimMaxDecayType);
      if(fUsemTScalingOfResidualRadii) fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetResidualCollection()->SetUsemTScalingOfRadii(fmTScalingPowerOfResidualRadii);
    }
  }



}


//________________________________________________________________________________________________________________
TString LednickyFitter::BuildParamCorrCoeffOutputFile(TString aFileBaseName, TString aFileType)
{
  ExistsSaveLocationBase();

  AnalysisType tAnType = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType();
  bool tConjIncluded = fNAnalyses % 2==0 ? true : false;
  TString tConjIncMod = "";
  if(tConjIncluded) tConjIncMod = TString("wConj");

  TString tOutputDir = TString::Format("%sParameterCorrelations/", fSaveLocationBase.Data());
  gSystem->mkdir(tOutputDir, true);

  TString tOutputName = TString::Format("%s%s_%s%s%s.%s", tOutputDir.Data(), aFileBaseName.Data(), cAnalysisBaseTags[tAnType], tConjIncMod.Data(), fSaveNameModifier.Data(), aFileType.Data());

  return tOutputName;
}




//________________________________________________________________________________________________________________
void LednickyFitter::DoFit()
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
  assert(fErrFlg==0);

  // Print results
  fMinuit->mnstat(fChi2,fEdm,fErrDef,fNvpar,fNparx,fIcstat);
  fMinuit->mnprin(3,fChi2);

  //---------------------------------
  TString tParamCorrOutputName = BuildParamCorrCoeffOutputFile("ParameterCorrelationCoefficients", "txt");
  gSystem->RedirectOutput(tParamCorrOutputName, "w");
  fMinuit->mnexcm("SHO COR", arglist ,2,fErrFlg);
  gSystem->RedirectOutput(0);
  //---------------------------------

  Finalize();
}


//________________________________________________________________________________________________________________
void LednickyFitter::Finalize()
{
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
/*
  if(fErrFlg==0)
  {
    int tNpar;
    double tChi2;
    fReturnPrimaryWithResidualsToAnalyses = true;
    CalculateFitFunction(tNpar,tChi2,tPar);
    fReturnPrimaryWithResidualsToAnalyses = false;
  }
  delete[] tPar;
*/
  fFitSharedAnalyses->SetMinuitMinParams(fMinParams);
  fFitSharedAnalyses->SetMinuitParErrors(fParErrors);
  fFitSharedAnalyses->ReturnFitParametersToAnalyses();

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetPrimaryFit(CreateFitFunction("fit",iAnaly));
  }

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iPartAn=0; iPartAn<fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetNFitPartialAnalysis(); iPartAn++)
    {
      fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetFitPartialAnalysis(iPartAn)->SetCorrectedFitVec(fCorrectedFitVecs[iAnaly][iPartAn]);
    }
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




//______________________________________________________________________________
td1dVec LednickyFitter::ReadLine(TString aLine)
{
  td1dVec tReturnVec(0);

  TObjArray* tValues = aLine.Tokenize("  ");

  double tValue;
  for(int i=0; i<tValues->GetEntries(); i++)
  {
    tValue = ((TObjString*)tValues->At(i))->String().Atof();
    tReturnVec.push_back(tValue);
  }
  return tReturnVec;
}


//______________________________________________________________________________
vector<int> LednickyFitter::GetNParamsAndRowWidth(ifstream &aStream)
{
  std::string tStdString;
  TString tLine;

  int tNParams = 0;
  int tRowWidth = 0;
  while(getline(aStream, tStdString))
  {
    tLine = TString(tStdString);
    if(tLine.Contains("*")) continue;
    if(tLine.Contains("PARAMETER")) continue;
    if(tLine.Contains("NO.")) continue;

    TObjArray* tValues = tLine.Tokenize("  ");
    if(tNParams==0) tRowWidth = tValues->GetEntries();
    if(tValues->GetEntries() == tRowWidth) tNParams++;
  }
  aStream.clear();
  aStream.seekg(0, ios::beg);

  return vector<int>{tNParams, tRowWidth};
}

//______________________________________________________________________________
void LednickyFitter::FinishMatrix(td2dVec &aMatrix, vector<int> &aNParamsAndRowWidth)
{
  //Due to how things are printed, and therefore, how I read them,
  // the first RowWidth rows only have RowWidth entries, whereas the
  // remaining (NParams-RowWidth) rows have full NParams entries
  int tNParams = aNParamsAndRowWidth[0];
  int tRowWidth = aNParamsAndRowWidth[1];

  for(int i=0; i<tRowWidth; i++)
  {
    for(int j=tRowWidth; j<tNParams; j++)
    {
      aMatrix[i].push_back(aMatrix[j][i]);
    }
  }

  //The matrix should now be symmetric about the diagonal, check this
  assert((int)aMatrix.size()==tNParams);
  for(unsigned int i=0; i<aMatrix.size(); i++)
  {
    assert((int)aMatrix[i].size()==tNParams);
    for(unsigned int j=0; j<aMatrix[i].size(); j++)
    {
      assert(aMatrix[i][j]==aMatrix[j][i]);
    }
  }

}


//______________________________________________________________________________
void LednickyFitter::PrintMatrix(td2dVec &aMatrix)
{
  cout << "Parameter Coefficient Matrix------------------------------" << endl;
  for(unsigned int i=0; i<aMatrix.size(); i++)
  {
    for(unsigned int j=0; j<aMatrix[i].size(); j++)
    {
      printf("% 05.3f  ", aMatrix[i][j]);
    }
    cout << endl;
  }
}


//________________________________________________________________________________________________________________
td2dVec LednickyFitter::GetParamCorrCoefMatrix(TString aFileLocation)
{
  ifstream tFileIn(aFileLocation);
  if(!tFileIn.is_open()) cout << "FAILURE - FILE NOT OPEN: " << aFileLocation << endl;
  assert(tFileIn.is_open());

  vector<int> tNParamsAndRowWidth = GetNParamsAndRowWidth(tFileIn);

  td2dVec tValuesMatrix;
  int tCounter = -1;

  std::string tStdString;
  TString tLine;
  while(getline(tFileIn, tStdString))
  {
    tLine = TString(tStdString);
    if(tLine.Contains("*")) continue;
    if(tLine.Contains("PARAMETER")) continue;
    if(tLine.Contains("NO.")) continue;

    td1dVec tValuesVec = ReadLine(tLine);

    if((int)tValuesVec.size() == tNParamsAndRowWidth[1])
    {
      tValuesVec.erase(tValuesVec.begin(), tValuesVec.begin()+2);  //First value is parameter number
                                                                   //Second value is global correlation value

      tValuesMatrix.push_back(tValuesVec);
      tCounter++;
    }
    else
    {
      tValuesMatrix[tCounter].insert(tValuesMatrix[tCounter].end(), tValuesVec.begin(), tValuesVec.end());
    }
  }
  tFileIn.close();

  //Due to the erase call above, RowWidth has decreased by 2
  tNParamsAndRowWidth[1] -= 2;
  FinishMatrix(tValuesMatrix, tNParamsAndRowWidth);
  PrintMatrix(tValuesMatrix);


  return tValuesMatrix;
}





//________________________________________________________________________________________________________________
vector<int> LednickyFitter::GetParamInfoFromMinuitParamNumber(int aMinuitParamNumber)
{
  vector<int> tReturnVec = {-1, -1, -1};  //[AnType, CentType, ParamType]

  AnalysisType tAnType, tPartAnType;
  CentralityType tCentType, tPartCentType;
  ParameterType tParamType;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    tAnType = tFitPairAnalysis->GetAnalysisType();
    tCentType = tFitPairAnalysis->GetCentralityType();
    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      tPartAnType = tFitPartialAnalysis->GetAnalysisType();
      tPartCentType = tFitPartialAnalysis->GetCentralityType();

      assert(tAnType==tPartAnType);
      assert(tCentType==tPartCentType);

      int tNFitParams = tFitPartialAnalysis->GetNFitParams();  //the +1 accounts for the normalization parameter
      assert(tNFitParams = 5);

      for(int iParam=0; iParam<tNFitParams; iParam++)
      {
        if(tFitPartialAnalysis->GetFitParameter(static_cast<ParameterType>(iParam))->GetMinuitParamNumber() == aMinuitParamNumber)
        {
          tParamType = static_cast<ParameterType>(iParam);
          tReturnVec = vector<int>{tAnType, tCentType, tParamType};
          return tReturnVec;
        }
      }
      if(tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber() == aMinuitParamNumber)
      {
        tParamType = kNorm;
        tReturnVec = vector<int>{tAnType, tCentType, tParamType};
        return tReturnVec;
      }
    }
  }
  assert(0);
  return tReturnVec;
}


//________________________________________________________________________________________________________________
TGraph* LednickyFitter::GetContourPlot(int aNPoints, int aParam1, int aParam2)
{
  //X-AXIS
  vector<int> tParamInfo1 = GetParamInfoFromMinuitParamNumber(aParam1);
  AnalysisType tAnType1 = static_cast<AnalysisType>(tParamInfo1[0]);
  CentralityType tCentType1 = static_cast<CentralityType>(tParamInfo1[1]);
  ParameterType tParamType1 = static_cast<ParameterType>(tParamInfo1[2]);
  TString tTextXax = TString::Format("%s %s %s", cAnalysisRootTags[tAnType1], cPrettyCentralityTags[tCentType1], cParameterNames[tParamType1]);

  //Y-AXIS
  vector<int> tParamInfo2 = GetParamInfoFromMinuitParamNumber(aParam2);
  AnalysisType tAnType2 = static_cast<AnalysisType>(tParamInfo2[0]);
  CentralityType tCentType2 = static_cast<CentralityType>(tParamInfo2[1]);
  ParameterType tParamType2 = static_cast<ParameterType>(tParamInfo2[2]);
  TString tTextYax = TString::Format("%s %s %s", cAnalysisRootTags[tAnType2], cPrettyCentralityTags[tCentType2], cParameterNames[tParamType2]);
  //-------------------
  TString tName = TString::Format("%s%s_%svs%s%s_%s", cAnalysisBaseTags[tAnType1], cCentralityTags[tCentType1], cParameterNames[tParamType1], 
                                                    cAnalysisBaseTags[tAnType2], cCentralityTags[tCentType2], cParameterNames[tParamType2]);

  TString tTitle = TString::Format("%s %s %s vs. %s %s %s", cAnalysisRootTags[tAnType1], cPrettyCentralityTags[tCentType1], cParameterNames[tParamType1], 
                                                            cAnalysisRootTags[tAnType2], cPrettyCentralityTags[tCentType2], cParameterNames[tParamType2]);

  cout << "GetContourPlot for " << tTitle << endl;

  //-------------------------------------------------------------------------
  TGraph* tReturnGr;

  tReturnGr = (TGraph*)fMinuit->Contour(aNPoints, aParam1, aParam2);
  int tStatus = fMinuit->GetStatus();

  //NOTE: TMinuit documentation is not accurate in describing what tStatus will be in various situations
  //      Look at code instead
  if(tStatus != 0)  //i.e., if failure
  {
    if(tStatus==-1) assert(0);  //if this is the case, I did something wrong
    else tReturnGr = new TGraph(0);
  }

  tReturnGr->SetName(tName);
  tReturnGr->SetTitle(tTitle);

  tReturnGr->GetXaxis()->SetTitle(tTextXax);
  tReturnGr->GetYaxis()->SetTitle(tTextYax);

  return tReturnGr;
}


//________________________________________________________________________________________________________________
void LednickyFitter::FixAllOtherParameters(int aParam1Exclude, int aParam2Exclude, vector<double> &aParamFitValues)
{
  int tNParams = fFitSharedAnalyses->GetNMinuitParams();
  assert(tNParams == (int)aParamFitValues.size());

  double arglist[2];
  int tErrFlg = 0;

  //First, free any previously fixed parameters
  fMinuit->mnexcm("RES",arglist,0,tErrFlg);
  for(int i=0; i<tNParams; i++)
  {
    //Next, make sure we have a fresh start, with all parameters set to their fit values
    arglist[0] = i+1;  //because Minuit numbering starts at 1, not 0!
    arglist[1] = aParamFitValues[i];
    fMinuit->mnexcm("SET PAR",arglist,2,tErrFlg);

    //Finally, fix appropriate parameters
    if(i != aParam1Exclude && i != aParam2Exclude) fMinuit->FixParameter(i);
  }

}

//________________________________________________________________________________________________________________
TCanvas* LednickyFitter::GenerateContourPlots(int aNPoints, const vector<double> &aParams, const vector<double> &aErrVals, TString aSaveNameModifier, bool aFixAllOthers)
{
  assert(aErrVals.size() <= 2);  //TODO for now, impose this restriction

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
  assert(fErrFlg==0);

  // Print results
  fMinuit->mnstat(fChi2,fEdm,fErrDef,fNvpar,fNparx,fIcstat);
  fMinuit->mnprin(3,fChi2);

  int tNParams = fFitSharedAnalyses->GetNMinuitParams();
  td1dVec tMinParams;
  for(int i=0; i<tNParams; i++)
  {
    double tempMinParam;
    double tempParError;
    fMinuit->GetParameter(i,tempMinParam,tempParError);
    
    tMinParams.push_back(tempMinParam);
  }

  //---------------------------------
  TString tParamCorrOutputName = BuildParamCorrCoeffOutputFile("ParameterCorrelationCoefficients", "txt");
  gSystem->RedirectOutput(tParamCorrOutputName, "w");
  fMinuit->mnexcm("SHO COR", arglist ,2,fErrFlg);
  gSystem->RedirectOutput(0);
  //---------------------------------

  TCanvas* tReturnCan = new TCanvas("tReturnCan", "tReturnCan");
  tReturnCan->Divide(aParams.size()-1,aParams.size()-1, 0.001, 0.001);  //-1 because I don't plot parameter correlation with itself
  tReturnCan->cd();

  int tPadNum = 0;
  TGraph* tGr;

  //-----
  TString tFixedAllOthersMod = "";
  if(aFixAllOthers) tFixedAllOthersMod = TString("_FixedAllOthers");

  TString tParamContoursFileName = BuildParamCorrCoeffOutputFile(TString::Format("ParameterContoursFile%s%s", aSaveNameModifier.Data(), tFixedAllOthersMod.Data()), "root");
  TFile *tSaveFile = new TFile(tParamContoursFileName, "RECREATE");
  TString tParamContoursFigureName = BuildParamCorrCoeffOutputFile(TString::Format("ParameterContoursFigure%s%s", aSaveNameModifier.Data(), tFixedAllOthersMod.Data()), "eps");

  td2dVec tParamCorrCoefMatrix = GetParamCorrCoefMatrix(tParamCorrOutputName);
  //-----

  double tErrVal = 0.;
  for(unsigned int iErrVal=0; iErrVal<aErrVals.size(); iErrVal++)
  {
    tErrVal = aErrVals[iErrVal];
    arglist[0] = tErrVal;
    fMinuit->mnexcm("SET ERR", arglist ,1,fErrFlg);

    for(int i=0; i< aParams.size(); i++)
    {
      for(int j=i+1; j< aParams.size(); j++)
      {
        tPadNum = i*(aParams.size()-1) + (j-1);  //-1's because I don't plot parameter correlation with itself
        tPadNum += 1;                            //+1 because pad numbering starts at 1, not 0
        tReturnCan->cd(tPadNum);

        if(aFixAllOthers) FixAllOtherParameters(aParams[i], aParams[j], tMinParams);

        tGr = GetContourPlot(aNPoints, aParams[i], aParams[j]);
        if(iErrVal==0)
        {
          tGr->SetFillColor(42);
          tGr->Draw("alf");

          TPaveText* tText = new TPaveText(0.75, 0.75, 0.90, 0.85, "NDC");
          tText->SetFillColor(0);
          tText->SetBorderSize(0);
          tText->SetTextAlign(22);
          tText->AddText(TString::Format("CorrCoeff = % 05.3f", tParamCorrCoefMatrix[aParams[i]][aParams[j]]));
          tText->Draw();
        }
        else
        {
          tGr->SetFillColor(38);
          tGr->Draw("lf");
        }
        tReturnCan->Update();
        tGr->Write(TString::Format("%s_tErrVal=%0.1f", tGr->GetName(), tErrVal));
      }
    }
  }

  tSaveFile->Close();
  tReturnCan->SaveAs(tParamContoursFigureName);
  return tReturnCan;
  //---------------------------------
//  Finalize();

}

//________________________________________________________________________________________________________________
TCanvas* LednickyFitter::GenerateContourPlots(int aNPoints, CentralityType aCentType, const vector<double> &aErrVals, bool aFixAllOthers)
{
  assert(fNAnalyses==6);  //This procedure assumes a full analysis with all centralities and pair/conj
  TString tSaveNameMod = cCentralityTags[aCentType];

  AnalysisType tAnType = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType();

  vector<double> aParams(0);
  if(aCentType==k0010)
  {
    if(tAnType != kLamK0) aParams = vector<double>{0, 1, 6, 9, 10, 11};
    else aParams = vector<double>{0, 1, 4, 5, 6};
  }
  else if(aCentType==k1030)
  {
    if(tAnType != kLamK0) aParams = vector<double>{2, 3, 7, 9, 10, 11};
    else aParams = vector<double>{0, 2, 4, 5, 6};
  }
  else if(aCentType==k3050)
  {
    if(tAnType != kLamK0) aParams = vector<double>{4, 5, 8, 9, 10, 11};
    else aParams = vector<double>{0, 3, 4, 5, 6};
  }
  else
  {
    tSaveNameMod = "_0010_1030_3050";
    for(int i=0; i<fFitSharedAnalyses->GetNMinuitParams(); i++)
    {
      aParams.push_back(i);
    }
  }

  return GenerateContourPlots(aNPoints, aParams, aErrVals, tSaveNameMod, aFixAllOthers);
}

//________________________________________________________________________________________________________________
void LednickyFitter::SetSaveLocationBase(TString aBase, TString aSaveNameModifier)
{
  fSaveLocationBase=aBase;
  if(!aSaveNameModifier.IsNull()) fSaveNameModifier = aSaveNameModifier;
}

//________________________________________________________________________________________________________________
void LednickyFitter::ExistsSaveLocationBase()
{
  if(!fSaveLocationBase.IsNull()) return;

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


