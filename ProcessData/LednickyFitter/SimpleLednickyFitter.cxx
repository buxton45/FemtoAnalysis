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
  fParErrors(0)

{
  fMinuit = new TMinuit(50);
  CreateMinuitParameters();
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
  double aMinNorm = 0.32, aMaxNorm = 0.40;
  TH1D* tNum = Get1dHisto(aFileLocation, TString("Num")+aBaseName+cAnalysisBaseTags[fAnalysisType]);
  TH1D* tDen = Get1dHisto(aFileLocation, TString("Den")+aBaseName+cAnalysisBaseTags[fAnalysisType]);

  fCfLite = new CfLite(TString("CfLite")+aBaseName+cAnalysisBaseTags[fAnalysisType], 
                       TString("CfLite")+aBaseName+cAnalysisBaseTags[fAnalysisType],
                       tNum, tDen, aMinNorm, aMaxNorm);

  fMinuit = new TMinuit(50);
  CreateMinuitParameters();
}


//________________________________________________________________________________________________________________
SimpleLednickyFitter::~SimpleLednickyFitter()
{
  cout << "LednickyFitter object is being deleted!!!" << endl;
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

  double tStartVal_Lambda = 0.5;
  double tStepSize_Lambda = 0.001;
  double tLowerBound_Lambda = 0.;
  double tUpperBound_Lambda = 0.;
  fMinuit->mnparm(0, TString("Lambda"), tStartVal_Lambda, tStepSize_Lambda, tLowerBound_Lambda, tUpperBound_Lambda, tErrFlg);

  double tStartVal_Radius = 5.0;
  double tStepSize_Radius = 0.001;
  double tLowerBound_Radius = 0.;
  double tUpperBound_Radius = 0.;
  fMinuit->mnparm(1, TString("Radius"), tStartVal_Radius, tStepSize_Radius, tLowerBound_Radius, tUpperBound_Radius, tErrFlg);

  double tStartVal_ReF0 = -0.5;
  double tStepSize_ReF0 = 0.001;
  double tLowerBound_ReF0 = 0.;
  double tUpperBound_ReF0 = 0.;
  fMinuit->mnparm(2, TString("ReF0"), tStartVal_ReF0, tStepSize_ReF0, tLowerBound_ReF0, tUpperBound_ReF0, tErrFlg);

  double tStartVal_ImF0 = 0.5;
  double tStepSize_ImF0 = 0.001;
  double tLowerBound_ImF0 = 0.;
  double tUpperBound_ImF0 = 0.;
  fMinuit->mnparm(3, TString("ImF0"), tStartVal_ImF0, tStepSize_ImF0, tLowerBound_ImF0, tUpperBound_ImF0, tErrFlg);

  double tStartVal_D0 = 0.0;
  double tStepSize_D0 = 0.001;
  double tLowerBound_D0 = 0.;
  double tUpperBound_D0 = 0.;
  fMinuit->mnparm(4, TString("D0"), tStartVal_D0, tStepSize_D0, tLowerBound_D0, tUpperBound_D0, tErrFlg);

  double tStartVal_Norm = (double)fCfLite->GetNumScale()/(double)fCfLite->GetDenScale();
  double tStepSize_Norm = 0.001;
  double tLowerBound_Norm = 0.;
  double tUpperBound_Norm = 0.;
  fMinuit->mnparm(5, TString("Norm"), tStartVal_Norm, tStepSize_Norm, tLowerBound_Norm, tUpperBound_Norm, tErrFlg);
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

  tParPrim[0] = par[tLambdaMinuitParamNumber];
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

  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    x[0] = fKStarBinCenters[ix-1];

    tNumContent[ix-1] = tNum->GetBinContent(ix);
    tDenContent[ix-1] = tDen->GetBinContent(ix);

    tPrimaryFitCfContent[ix-1] = LednickyFitter::LednickyEq(x,tParPrim);
  }

  LednickyFitter::ApplyNormalization(tParPrim[5], tPrimaryFitCfContent);

  for(int ix=0; ix < fNbinsXToFit; ix++)
  {
    if(tRejectOmega && (fKStarBinCenters[ix] > tRejectOmegaLow) && (fKStarBinCenters[ix] < tRejectOmegaHigh)) {fChi2+=0;}
    else
    {
      if(tNumContent[ix]!=0 && tDenContent[ix]!=0 && tPrimaryFitCfContent[ix]!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
      {
        double tChi2 = 0.;
        if(fFitType == kChi2PML) tChi2 = LednickyFitter::GetPmlValue(tNumContent[ix],tDenContent[ix],tPrimaryFitCfContent[ix]);
        else if(fFitType == kChi2) tChi2 = LednickyFitter::GetChi2Value(ix+1,tCf,tParPrim);
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

  fNbinsXToBuild = std::max({tNbinsXToBuildMomResCrctn, tNbinsXToBuildResiduals, fNbinsXToFit});

  fKStarBinWidth = fCfLite->Num()->GetXaxis()->GetBinWidth(1);
  //-------------------------------------------------------------------------------------------


  TH1* tNum = fCfLite->Num();
  TH1* tDen = fCfLite->Den();
  TH1* tCf = fCfLite->Cf();

  assert(tNum->GetXaxis()->GetBinWidth(1) == tDen->GetXaxis()->GetBinWidth(1));
  assert(tNum->GetXaxis()->GetBinWidth(1) == tCf->GetXaxis()->GetBinWidth(1));
  assert(tNum->GetXaxis()->GetBinWidth(1) == fKStarBinWidth);

  //make sure tNum and tDen have same number of bins
  assert(tNum->GetNbinsX() == tDen->GetNbinsX());
  assert(tNum->GetNbinsX() == tCf->GetNbinsX());

  tTempNbinsXToFit = tNum->FindBin(fMaxFitKStar);
  if(tNum->GetBinLowEdge(tTempNbinsXToFit) == fMaxFitKStar) tTempNbinsXToFit--;

  if(tTempNbinsXToFit > tNum->GetNbinsX()) {tTempNbinsXToFit = tNum->GetNbinsX();}  //in case I accidentally include an overflow bin in nbinsXToFit
  assert(tTempNbinsXToFit == fNbinsXToFit);

  fKStarBinCenters.resize(fNbinsXToBuild,0.);
  for(int ix=1; ix <= fNbinsXToBuild; ix++)
  {
    fKStarBinCenters[ix-1] = tNum->GetXaxis()->GetBinCenter(ix);
  }


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

}


//________________________________________________________________________________________________________________
void SimpleLednickyFitter::DrawCfWithFit(TPad *aPad)
{
  aPad->cd();

  TH1D* tCf = (TH1D*)fCfLite->Cf();
  TF1* tFit = CreateFitFunction(TString("Fit"));


  tCf->Draw();
  tFit->Draw("same");
}





