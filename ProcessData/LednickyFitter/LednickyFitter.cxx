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

//________________________________________________________________________
double GetLednickyF1(double z)
{
  double result = (1./z)*Faddeeva::Dawson(z);
  return result;
}

//________________________________________________________________________
double GetLednickyF2(double z)
{
  double result = (1./z)*(1.-exp(-z*z));
  return result;
}

//________________________________________________________________________
double LednickyEq(double *x, double *par)
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
  Cf *= par[5];

  return Cf;
  
}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
LednickyFitter::LednickyFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):

  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),
  fCfsToFit(fNAnalyses),
  fFits(fNAnalyses),
  fMaxFitKStar(aMaxFitKStar),
  fRejectOmega(false),
  fApplyMomResCorrection(false), //TODO change deault to true here AND in CoulombFitter
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
  for(int i=0; i<fNAnalyses; i++)
  {
    //fCfsToFit[i] = fFitSharedAnalyses->GetPairAnalysis(i)->GetCf();
  }

}


//________________________________________________________________________________________________________________
LednickyFitter::~LednickyFitter()
{
  cout << "LednickyFitter object is being deleted!!!" << endl;
}


/*
//________________________________________________________________________________________________________________
void LednickyFitter::CalculateChi2(int &npar, double &chi2, double *par)
{
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    double *tPar = new double[fFitSharedAnalyses->GetNParamsPerAnalysis()];
    //double *tPar = new double[6];

    TH1F* tCfToFit = fCfsToFit[iAnaly];

    TAxis *xaxis = tCfToFit->GetXaxis();
    //TAxis *yaxis = tCfToFit->GetYaxis();

    int nbinsX = tCfToFit->GetNbinsX();
    int nbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
    if(nbinsXToFit > nbinsX) {nbinsXToFit = nbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

    PairAnalysis* tempAnaly = fFitSharedAnalyses->GetPairAnalysis(iAnaly);

    int tLambdaMinuitParamNumber = tempAnaly->GetParameter(Parameter::kLambda)->GetMinuitParamNumber();
    int tRadiusMinuitParamNumber = tempAnaly->GetParameter(Parameter::kRadius)->GetMinuitParamNumber();
    int tRef0MinuitParamNumber = tempAnaly->GetParameter(Parameter::kRef0)->GetMinuitParamNumber();
    int tImf0MinuitParamNumber = tempAnaly->GetParameter(Parameter::kImf0)->GetMinuitParamNumber();
    int td0MinuitParamNumber = tempAnaly->GetParameter(Parameter::kd0)->GetMinuitParamNumber();
    int tNormMinuitParamNumber = tempAnaly->GetParameter(Parameter::kNorm)->GetMinuitParamNumber();

    tPar[0] = par[tLambdaMinuitParamNumber];
    tPar[1] = par[tRadiusMinuitParamNumber];
    tPar[2] = par[tRef0MinuitParamNumber];
    tPar[3] = par[tImf0MinuitParamNumber];
    tPar[4] = par[td0MinuitParamNumber];
    tPar[5] = par[tNormMinuitParamNumber];


    double tmp;
    double x[1];

    bool tRejectOmega = tempAnaly->RejectOmega();

    for(int ix=1; ix <= nbinsXToFit; ix++)
    {
      x[0] = xaxis->GetBinCenter(ix);
      if(tRejectOmega && (x[0] > tRejectOmegaLow) && (x[0] < tRejectOmegaHigh)) {fChi2+=0;}
      else
      {
        tmp = (tCfToFit->GetBinContent(ix) - LednickyEq(x,tPar))/tCfToFit->GetBinError(ix);

        fChi2Vec[iAnaly] += tmp*tmp;
        fChi2 += tmp*tmp;

        fNpFitsVec[iAnaly]++;
        fNpFits++;
      }

    }

  }

  chi2 = fChi2;

}
*/

//________________________________________________________________________________________________________________
double LednickyFitter::GetLednickyMomResCorrectedPoint(double aKStar, double* aPar, TH2* aMomResMatrix)
{
  double tKStar[1];
  int aKStarRecBin = aMomResMatrix->GetYaxis()->FindBin(aKStar);

  double tValue = 0.;
  for(int j=1; j<=aMomResMatrix->GetNbinsX(); j++)
  {
    tKStar[0] = aMomResMatrix->GetXaxis()->GetBinCenter(j);
    tValue += LednickyEq(tKStar,aPar)*aMomResMatrix->GetBinContent(j,aKStarRecBin);
  }
  tValue /= aMomResMatrix->Integral(1,aMomResMatrix->GetNbinsX(),aKStarRecBin,aKStarRecBin);
  return tValue;
}


//________________________________________________________________________________________________________________
double LednickyFitter::ApplyResidualCorrelationToPoint(double aKStar, double* aPar, TH2* aTransformMatrix)
{
  double tKStar[1];
  int tKStarBinOfInterest = aTransformMatrix->GetXaxis()->FindBin(aKStar);

  double tValue = 0.;
  for(int j=1; j<=aTransformMatrix->GetNbinsY(); j++)
  {
    tKStar[0] = aTransformMatrix->GetYaxis()->GetBinCenter(j);
    tValue += LednickyEq(tKStar,aPar)*aTransformMatrix->GetBinContent(tKStarBinOfInterest,j);
  }
  tValue /= aTransformMatrix->Integral(tKStarBinOfInterest,tKStarBinOfInterest,1,aTransformMatrix->GetNbinsY());
  return tValue;
}


//________________________________________________________________________________________________________________
void LednickyFitter::CalculateChi2PML(int &npar, double &chi2, double *par)
{
  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
  cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

  cout << "\t\t\tpar[2] = ReF0  = " << par[2] << endl;
  cout << "\t\t\tpar[3] = ImF0  = " << par[3] << endl;
  cout << "\t\t\tpar[4] = D0    = " << par[4] << endl;

  cout << "\t\t\tpar[8] = Norm1  = " << par[5] << endl;
  cout << "\t\t\tpar[9] = Norm2  = " << par[6] << endl;
  cout << "\t\t\tpar[10] = Norm3 = " << par[7] << endl;
  cout << "\t\t\tpar[11] = Norm4 = " << par[8] << endl;
  cout << "\t\t\tpar[12] = Norm5 = " << par[9] << endl;
  cout << "\t\t\tpar[13] = Norm6 = " << par[10] << endl;
  cout << "\t\t\tpar[14] = Norm7 = " << par[11] << endl;
  cout << "\t\t\tpar[15] = Norm8 = " << par[12] << endl;
  cout << "\t\t\tpar[16] = Norm9 = " << par[13] << endl;
  cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl << endl;

  //---------------------------------------------------------


  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
/*
    CfHeavy* tKStarCfHeavy = fFitSharedAnalyses->GetKStarCfHeavy(iAnaly);
    TObjArray* tNumCollection = tKStarCfHeavy->GetNumCollection();
    TObjArray *tDenCollection = tKStarCfHeavy->GetDenCollection();
*/
    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();
      TAxis* tXaxisDen = tDen->GetXaxis();

      //make sure tNum and tDen have to same bin width
      assert(tXaxisNum->GetBinWidth(1) == tXaxisDen->GetBinWidth(1));

      int tNbinsX = tNum->GetNbinsX();
      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams == 6);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;
      double x[1];

      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        x[0] = tXaxisNum->GetBinCenter(ix);
        if(tRejectOmega && (x[0] > tRejectOmegaLow) && (x[0] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          double tNumContent = tNum->GetBinContent(ix);
          double tDenContent = tDen->GetBinContent(ix);
          double tCfContent = LednickyEq(x,tPar);

          if(tNumContent!=0 && tDenContent!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tTerm1 = tNumContent*log(  (tCfContent*(tNumContent+tDenContent)) / (tNumContent*(tCfContent+1))  );
            double tTerm2 = tDenContent*log(  (tNumContent+tDenContent) / (tDenContent*(tCfContent+1))  );
            tmp = -2.0*(tTerm1+tTerm2);

            fChi2Vec[iAnaly] += tmp;
            fChi2 += tmp;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }
      }

      delete[] tPar;
    }

  }

  chi2 = fChi2;
}

//________________________________________________________________________________________________________________
void LednickyFitter::CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par)
{
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);

    TH2* tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
    assert(tMomResMatrix);
/*
    CfHeavy* tKStarCfHeavy = fFitSharedAnalyses->GetKStarCfHeavy(iAnaly);
    TObjArray* tNumCollection = tKStarCfHeavy->GetNumCollection();
    TObjArray *tDenCollection = tKStarCfHeavy->GetDenCollection();
*/
    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();
      TAxis* tXaxisDen = tDen->GetXaxis();

      //make sure tNum and tDen have to same bin width
      assert(tXaxisNum->GetBinWidth(1) == tXaxisDen->GetBinWidth(1));

      int tNbinsX = tNum->GetNbinsX();
      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams == 6);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;
      double x[1];

      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        x[0] = tXaxisNum->GetBinCenter(ix);
        if(tRejectOmega && (x[0] > tRejectOmegaLow) && (x[0] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          double tNumContent = tNum->GetBinContent(ix);
          double tDenContent = tDen->GetBinContent(ix);
//          double tCfContent = LednickyEq(x,tPar);
          double tCfContent = GetLednickyMomResCorrectedPoint(x[0],tPar,tMomResMatrix);


          if(tNumContent!=0 && tDenContent!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tTerm1 = tNumContent*log(  (tCfContent*(tNumContent+tDenContent)) / (tNumContent*(tCfContent+1))  );
            double tTerm2 = tDenContent*log(  (tNumContent+tDenContent) / (tDenContent*(tCfContent+1))  );
            tmp = -2.0*(tTerm1+tTerm2);

            fChi2Vec[iAnaly] += tmp;
            fChi2 += tmp;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }
      }

      delete[] tPar;
    }

  }

  chi2 = fChi2;
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
vector<double> LednickyFitter::GetResidualCorrelation(vector<double> &aParentCf, vector<double> &aKStarBinCenters, TH2* aTransformMatrix)
{
  unsigned int tDaughterPairKStarBin, tParentPairKStarBin;
  double tDaughterPairKStar, tParentPairKStar;
  assert(aParentCf.size() == aKStarBinCenters.size());
  assert(aParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsX());
  assert(aParentCf.size() == (unsigned int)aTransformMatrix->GetNbinsY());

  vector<double> tReturnResCf(aParentCf.size(),0.);
  vector<double> tNormVec(aParentCf.size(),0.);  //TODO once I match bin size, I should be able to call /= by integral, instead of tracking normVec

  for(unsigned int i=0; i<aParentCf.size(); i++)
  {
    tDaughterPairKStar = aKStarBinCenters[i];
    tDaughterPairKStarBin = aTransformMatrix->GetXaxis()->FindBin(tDaughterPairKStar);

    for(unsigned int j=0; j<aParentCf.size(); j++)
    {
      tParentPairKStar = aKStarBinCenters[j];
      tParentPairKStarBin = aTransformMatrix->GetYaxis()->FindBin(tParentPairKStar);

      tReturnResCf[i] += aParentCf[j]*aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
      tNormVec[i] += aTransformMatrix->GetBinContent(tDaughterPairKStarBin,tParentPairKStarBin);
    }
    tReturnResCf[i] /= tNormVec[i];
  }
  return tReturnResCf;
}


//________________________________________________________________________________________________________________
void LednickyFitter::CalculateChi2PMLwMomResCorrectionv2(int &npar, double &chi2, double *par)
{
  assert(fApplyMomResCorrection);

  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  int tNFitParPerAnalysis = 5;
//  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
//  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;

  int tNbinsXToBuildGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetModelKStarTrueVsRecMixed()->GetNbinsX();  // when applying momentum resolution corrections, many times
                                                                                                                     // you must go beyond fitting range to apply correction

//  vector<double> tCfContentUnNorm(tNbinsXToBuildGlobal,0.);
  vector<double> tCfContent(tNbinsXToBuildGlobal,0.);
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

    TH2* tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
    assert(tMomResMatrix);
    int tNbinsXToBuild = tMomResMatrix->GetNbinsX();
    assert(tNbinsXToBuild == tNbinsXToBuildGlobal);

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same bin size as tMomResMatrix
      assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetXaxis()->GetBinWidth(1));
      assert(tNum->GetXaxis()->GetBinWidth(1) == tMomResMatrix->GetYaxis()->GetBinWidth(1));

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
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;
      double x[1];
      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();
      //bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);

      for(int ix=1; ix <= tNbinsXToBuild; ix++)
      {
        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);
        x[0] = tKStarBinCenters[ix-1];

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

        tCfContent[ix-1] = LednickyEq(x,tPar);
      }

      vector<double> tCfContentwMomResCorrection = ApplyMomResCorrection(tCfContent, tKStarBinCenters, tMomResMatrix);
      for(int ix=0; ix < tNbinsXToFit; ix++)
      {
        if(tRejectOmega && (tKStarBinCenters[ix] > tRejectOmegaLow) && (tKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tCfContentwMomResCorrection[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tTerm1 = tNumContent[ix]*log(  (tCfContentwMomResCorrection[ix]*(tNumContent[ix]+tDenContent[ix])) / (tNumContent[ix]*(tCfContentwMomResCorrection[ix]+1))  );
            double tTerm2 = tDenContent[ix]*log(  (tNumContent[ix]+tDenContent[ix]) / (tDenContent[ix]*(tCfContentwMomResCorrection[ix]+1))  );
            tmp = -2.0*(tTerm1+tTerm2);
  
            fChi2Vec[iAnaly] += tmp;
            fChi2 += tmp;

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

cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;
/*
  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
*/
}



//________________________________________________________________________________________________________________
void LednickyFitter::CalculateChi2PMLwCorrectedCfs(int &npar, double &chi2, double *par)
{
  double tRejectOmegaLow = 0.19;
  double tRejectOmegaHigh = 0.23;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
/*
    CfHeavy* tKStarCfHeavy = fFitSharedAnalyses->GetKStarCfHeavy(iAnaly);
    TObjArray* tNumCollection = tKStarCfHeavy->GetNumCollection();
    TObjArray *tDenCollection = tKStarCfHeavy->GetDenCollection();
*/

    TH1* tCorrectionHist = tFitPairAnalysis->GetModelCfFakeIdealCfFakeRatio();
    assert(tCorrectionHist);

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();

      TH1* tNum = tKStarCfLite->Num();
      TH1* tDen = tKStarCfLite->Den();

      //make sure tNum and tDen have same number of bins
      assert(tNum->GetNbinsX() == tDen->GetNbinsX());

      TAxis* tXaxisNum = tNum->GetXaxis();
      TAxis* tXaxisDen = tDen->GetXaxis();

      //make sure tNum and tDen have to same bin width
      assert(tXaxisNum->GetBinWidth(1) == tXaxisDen->GetBinWidth(1));

      int tNbinsX = tNum->GetNbinsX();
      int tNbinsXToFit = tNum->FindBin(fMaxFitKStar);
      if(tNum->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() +1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams == 6);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;
      double x[1];

      bool tRejectOmega = tFitPartialAnalysis->RejectOmega();

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        x[0] = tXaxisNum->GetBinCenter(ix);
        if(tRejectOmega && (x[0] > tRejectOmegaLow) && (x[0] < tRejectOmegaHigh)) {fChi2+=0;}
        else
        {
          double tNumContent = tNum->GetBinContent(ix);
          //tNumContent *= tCorrectionHist->GetBinContent(tCorrectionHist->FindBin(x[0]));
          double tDenContent = tDen->GetBinContent(ix);
          double tCfContent = LednickyEq(x,tPar);
          tCfContent /= tCorrectionHist->GetBinContent(tCorrectionHist->FindBin(x[0]));

          if(tNumContent!=0 && tDenContent!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
          {
            double tTerm1 = tNumContent*log(  (tCfContent*(tNumContent+tDenContent)) / (tNumContent*(tCfContent+1))  );
            double tTerm2 = tDenContent*log(  (tNumContent+tDenContent) / (tDenContent*(tCfContent+1))  );
            tmp = -2.0*(tTerm1+tTerm2);

            fChi2Vec[iAnaly] += tmp;
            fChi2 += tmp;

            fNpFitsVec[iAnaly]++;
            fNpFits++;
          }
        }
      }

      delete[] tPar;
    }

  }

  chi2 = fChi2;
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
//  fCfsToFit[aAnalysisNumber]->GetListOfFunctions()->Add(ReturnFunction);


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
  arglist[0] = 1;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 5000;
  arglist[1] = 0.01;
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

  //get result
  for(int i=0; i<tNParams; i++)
  {
    double tempMinParam;
    double tempParError;
    fMinuit->GetParameter(i,tempMinParam,tempParError);
    
    fMinParams.push_back(tempMinParam);
    fParErrors.push_back(tempParError);
  }

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

  const int nValuesImf0 = 4;
  vector<double> tImf0Values(nValuesImf0);
    tImf0Values[0] = 0.1;
    tImf0Values[1] = 0.3;
    tImf0Values[2] = 0.5;
    tImf0Values[3] = 1.;

  const int nValuesd0 = 7;
  vector<double> td0Values(nValuesd0);
    td0Values[0] = -10.;
    td0Values[1] = -1.;
    td0Values[2] = -0.1;
    td0Values[3] = 0.;
    td0Values[4] = 0.1;
    td0Values[5] = 1.;
    td0Values[6] = 10.;

  const int nValuesNorm = 1;
  vector<double> tNormValues(nValuesNorm);
    tNormValues[0] = 1.;

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

