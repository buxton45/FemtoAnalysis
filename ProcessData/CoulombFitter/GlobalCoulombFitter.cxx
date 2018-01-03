// GlobalCoulombFitter

#include "GlobalCoulombFitter.h"

#ifdef __ROOT__
ClassImp(GlobalCoulombFitter)
#endif

//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
GlobalCoulombFitter::GlobalCoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar):
  CoulombFitter(aFitSharedAnalyses, aMaxFitKStar),

  fInterpHistsLoadedOppSign(false),
  fCoulombTypeOppSign(kAttractive),
  
  fInterpHistFileOppSign(0),
  fGTildeRealHistOppSign(0),
  fGTildeImagHistOppSign(0),
  fHyperGeo1F1RealHistOppSign(0),
  fHyperGeo1F1ImagHistOppSign(0)
{

}


//________________________________________________________________________________________________________________
GlobalCoulombFitter::~GlobalCoulombFitter()
{
  cout << "GlobalCoulombFitter object is being deleted!!!" << endl;

  //---Clean up
  if(fInterpHistsLoadedOppSign)
  {
    delete fHyperGeo1F1RealHistOppSign;
    delete fHyperGeo1F1ImagHistOppSign;

    delete fGTildeRealHistOppSign;
    delete fGTildeImagHistOppSign;

    fInterpHistFileOppSign->Close();
    delete fInterpHistFileOppSign;
  }

}


//________________________________________________________________________________________________________________
void GlobalCoulombFitter::LoadInterpHistFileOppSign(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting LoadInterpHistFileOppSign" << endl;


  TString aFileName = aFileBaseName+".root";
  fInterpHistFileOppSign = TFile::Open(aFileName);
//--------------------------------------------------------------
  fHyperGeo1F1RealHistOppSign = (TH3D*)fInterpHistFileOppSign->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHistOppSign = (TH3D*)fInterpHistFileOppSign->Get("HyperGeo1F1Imag");
    fHyperGeo1F1RealHistOppSign->SetDirectory(0);
    fHyperGeo1F1ImagHistOppSign->SetDirectory(0);

  fGTildeRealHistOppSign = (TH2D*)fInterpHistFileOppSign->Get("GTildeReal");
  fGTildeImagHistOppSign = (TH2D*)fInterpHistFileOppSign->Get("GTildeImag");
    fGTildeRealHistOppSign->SetDirectory(0);
    fGTildeImagHistOppSign->SetDirectory(0);

  fInterpHistFileOppSign->Close();


  assert(fMinInterpKStar == fHyperGeo1F1RealHistOppSign->GetXaxis()->GetBinCenter(1));
  assert(fMaxInterpKStar == fHyperGeo1F1RealHistOppSign->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHistOppSign->GetNbinsX()));

  assert(fMinInterpRStar == fHyperGeo1F1RealHistOppSign->GetYaxis()->GetBinCenter(1));
  assert(fMaxInterpRStar == fHyperGeo1F1RealHistOppSign->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHistOppSign->GetNbinsY()));

  assert(fMinInterpTheta == fHyperGeo1F1RealHistOppSign->GetZaxis()->GetBinCenter(1));
  assert(fMaxInterpTheta == fHyperGeo1F1RealHistOppSign->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHistOppSign->GetNbinsZ()));

//--------------------------------------------------------------
  cout << "Interpolation histograms LOADED!" << endl;

tTimer.Stop();
cout << "LoadInterpHistFileOppSign: ";
tTimer.PrintInterval();

  fInterpHistsLoadedOppSign = true;
}


//________________________________________________________________________________________________________________
complex<double> GlobalCoulombFitter::BuildScatteringLength(AnalysisType aAnalysisType, double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;
  double tLednickyHFunction = Interpolator::LinearInterpolate(fLednickyHFunctionHist,aKStar);

  double tImagChi = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
  complex<double> tLednickyChi (tLednickyHFunction,tImagChi);

  tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);

  return tScattAmp;
}


//________________________________________________________________________________________________________________
double GlobalCoulombFitter::InterpolateWfSquared(AnalysisType aAnalysisType, double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

  //TODO put check to make sure file is open, not sure if assert(fInterpHistFile->IsOpen works);
  assert(fInterpHistsLoaded);
  assert(fInterpHistsLoadedOppSign);
  //assert(fInterpHistFile->IsOpen());

  double tGamow = GetGamowFactor(aKStarMag);
  complex<double> tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag;
  //-------------------------------------
  if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM)
  {
    tHyperGeo1F1Real = Interpolator::TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
    tHyperGeo1F1Imag = Interpolator::TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

    tGTildeReal = Interpolator::BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
    tGTildeImag = Interpolator::BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);
  }
  else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP)
  {
    tHyperGeo1F1Real = Interpolator::TrilinearInterpolate(fHyperGeo1F1RealHistOppSign,aKStarMag,aRStarMag,aTheta);
    tHyperGeo1F1Imag = Interpolator::TrilinearInterpolate(fHyperGeo1F1ImagHistOppSign,aKStarMag,aRStarMag,aTheta);

    tGTildeReal = Interpolator::BilinearInterpolate(fGTildeRealHistOppSign,aKStarMag,aRStarMag);
    tGTildeImag = Interpolator::BilinearInterpolate(fGTildeImagHistOppSign,aKStarMag,aRStarMag);
  }
  else assert(0);

  tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
  tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);
  //-------------------------------------

  complex<double> tScattLenComplex = BuildScatteringLength(aAnalysisType,aKStarMag,aReF0,aImF0,aD0);
  tScattLenComplexConj = std::conj(tScattLenComplex);

  //-------------------------------------------

  complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitter::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}





//________________________________________________________________________________________________________________
double GlobalCoulombFitter::GetFitCfContent(AnalysisType aAnalysisType, double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  double tBinSize = 0.01;
  int tBin = std::round(aKStarMagMin/tBinSize);

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;

  tMaxKStarCalls = 100000;

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*par[1]);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*par[1]);

  std::uniform_int_distribution<int> tRandomKStarElement;
  if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[aAnalysisNumber][tBin].size()-1);

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  int tNInterpolate = 0;
  int tNMathematica = 0;

  for(int i=0; i<tMaxKStarCalls; i++)
  {
    if(!fUseRandomKStarVectors)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
    }
    else SetRandomKStar3Vec(tKStar3Vec,aKStarMagMin,aKStarMagMax);

    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric

    tTheta = tKStar3Vec->Angle(*tSource3Vec);
    tKStarMag = tKStar3Vec->Mag();

    tRStarMag = tSource3Vec->Mag();

    if(fTurnOffCoulomb || CanInterpAll(tKStarMag,tRStarMag,tTheta)) 
    {
      tWaveFunctionSq = InterpolateWfSquared(aAnalysisType,tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tNInterpolate++;
    }
    else
    {
      tWaveFunction = fWaveFunction->GetWaveFunction(tKStar3Vec,tSource3Vec,par[2],par[3],par[4]);
      tWaveFunctionSq = norm(tWaveFunction);
      tNMathematica++;
    }

    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }

  delete tKStar3Vec;
  delete tSource3Vec;

  tReturnCfContent /= tCounter;
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));

  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double GlobalCoulombFitter::GetFitCfContentwStaticPairs(AnalysisType aAnalysisType, double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  UpdatePairRadiusParameter(par[1], aAnalysisNumber);

  double tBinSize = aKStarMagMax-aKStarMagMin;
  int tBin = std::round(aKStarMagMin/tBinSize);


  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample4dVec[aAnalysisNumber][tBin].size();

  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  int tNInterpolate = 0;
  int tNMathematica = 0;

  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0];
    tRStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1];
    tTheta = fPairSample4dVec[aAnalysisNumber][tBin][i][2];

    if(fTurnOffCoulomb || CanInterpAll(tKStarMag,tRStarMag,tTheta)) 
    {
      tWaveFunctionSq = InterpolateWfSquared(aAnalysisType, tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tNInterpolate++;
    }
    else
    {
      tWaveFunction = fWaveFunction->GetWaveFunction(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tWaveFunctionSq = norm(tWaveFunction);
      tNMathematica++;
    }
    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }

  tReturnCfContent /= tCounter;

  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
void GlobalCoulombFitter::CalculateChi2PML(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;
  PrintCurrentParamValues(fFitSharedAnalyses->GetNMinuitParams(),par);

  int tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;

  vector<double> tCfContentUnNorm(tNbinsXToFitGlobal,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
    if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
    else fBohrRadius = 1000000000;
    fWaveFunction->SetCurrentAnalysisType(tAnalysisType);

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
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams = tNFitParPerAnalysis+1);

      int tLambdaMinuitParamNumber, tRadiusMinuitParamNumber, tRef0MinuitParamNumber, tImf0MinuitParamNumber, td0MinuitParamNumber, tNormMinuitParamNumber;
      double *tPar;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      assert(tNFitParams == 6);
      tPar = new double[tNFitParams];

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

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        double tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        double tNumContent = tNum->GetBinContent(ix);
        double tDenContent = tDen->GetBinContent(ix);


        if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tAnalysisType,tKStarMin,tKStarMax,tPar,iAnaly);
        else tCfContentUnNorm[ix-1] = GetFitCfContent(tAnalysisType,tKStarMin,tKStarMax,tPar,iAnaly);


        double tCfContent;
        tCfContent = tPar[5]*tCfContentUnNorm[ix-1];

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

      delete[] tPar;
    } 

  }

  delete[] tCurrentFitPar;

  if(std::isnan(fChi2) || std::isinf(fChi2))
  {
    cout << "WARNING: fChi2 = nan, setting it equal to 10^9-----------------------------------------" << endl << endl;
    fChi2 = pow(10,9);
  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;

tTotalTimer.Stop();
cout << "tTotalTimer: ";
tTotalTimer.PrintInterval();


  double *tParamsForHistograms = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tParamsForHistograms[i] = par[i];
  fFitSharedAnalyses->GetFitChi2Histograms()->FillHistograms(fChi2,tParamsForHistograms);
  delete[] tParamsForHistograms;
}

