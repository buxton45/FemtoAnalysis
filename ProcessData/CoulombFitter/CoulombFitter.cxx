///////////////////////////////////////////////////////////////////////////
// CoulombFitter:                                                        //
///////////////////////////////////////////////////////////////////////////


#include "CoulombFitter.h"

#ifdef __ROOT__
ClassImp(CoulombFitter)
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


//________________________________________________________________________________________________________________
CoulombFitter::CoulombFitter():
  fCreateInterpVectors(false),
  fUseScattLenHists(false),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(true),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),
  fApplyMomResCorrection(false), //TODO change default to true

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(0),
  fMinuit(0),
  fNAnalyses(0),
  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar4dVec(0),
  fNPairsPerKStarBin(16384),
  fCurrentRadiusParameter(1.),
  fPairSample4dVec(0),
  fInterpHistFile(0), fInterpHistFileScatLenReal1(0), fInterpHistFileScatLenImag1(0), fInterpHistFileScatLenReal2(0), fInterpHistFileScatLenImag2(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fCoulombScatteringLengthRealHist1(0),
  fCoulombScatteringLengthImagHist1(0),

  fCoulombScatteringLengthRealHist2(0),
  fCoulombScatteringLengthImagHist2(0),

  fMinInterpKStar1(0), fMinInterpKStar2(0), fMinInterpRStar(0), fMinInterpTheta(0), fMinInterpReF0(0), fMinInterpImF0(0), fMinInterpD0(0),
  fMaxInterpKStar1(0), fMaxInterpKStar2(0), fMaxInterpRStar(0), fMaxInterpTheta(0), fMaxInterpReF0(0), fMaxInterpImF0(0), fMaxInterpD0(0),

  fLednickyHFunction(0),

  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fCoulombScatteringLengthReal(0),
  fCoulombScatteringLengthImag(0),

  fCoulombScatteringLengthRealSub(0),
  fCoulombScatteringLengthImagSub(0),

  fCfsToFit(0),
  fFits(0),
  fMaxFitKStar(0),
  fRejectOmega(false),
  fChi2(0),
  fChi2GlobalMin(1000000000),
  fChi2Vec(0),
  fNpFits(0),
  fNpFitsVec(0),
  fNDF(0),
  fErrFlg(0),
  fMinParams(0),
  fParErrors(0)

{
  gRandom->SetSeed();

  fWaveFunction = new WaveFunction();

  omp_set_num_threads(3);

}



//________________________________________________________________________________________________________________
CoulombFitter::CoulombFitter(FitSharedAnalyses* aFitSharedAnalyses, double aMaxFitKStar, bool aCreateInterpVectors, bool aUseScattLenHists):
  fCreateInterpVectors(aCreateInterpVectors),
  fUseScattLenHists(aUseScattLenHists),
  fTurnOffCoulomb(false),
  fInterpHistsLoaded(false),
  fIncludeSingletAndTriplet(true),
  fUseRandomKStarVectors(false),
  fUseStaticPairs(false),
  fApplyMomResCorrection(false), //TODO change default to true

  fNCalls(0),
  fFakeCf(0),

  fFitSharedAnalyses(aFitSharedAnalyses),
  fMinuit(fFitSharedAnalyses->GetMinuitObject()),
  fNAnalyses(fFitSharedAnalyses->GetNFitPairAnalysis()),
  fAllOfSameCoulombType(false),
  fCoulombType(kRepulsive),
  fWaveFunction(0),
  fBohrRadius(gBohrRadiusXiK),

  fPairKStar4dVec(0),
  fNPairsPerKStarBin(16384),
  fCurrentRadiusParameter(1.),
  fPairSample4dVec(0),
  fInterpHistFile(0), fInterpHistFileScatLenReal1(0), fInterpHistFileScatLenImag1(0), fInterpHistFileScatLenReal2(0), fInterpHistFileScatLenImag2(0), fInterpHistFileLednickyHFunction(0),

  fLednickyHFunctionHist(0),

  fGTildeRealHist(0),
  fGTildeImagHist(0),

  fHyperGeo1F1RealHist(0),
  fHyperGeo1F1ImagHist(0),

  fCoulombScatteringLengthRealHist1(0),
  fCoulombScatteringLengthImagHist1(0),

  fCoulombScatteringLengthRealHist2(0),
  fCoulombScatteringLengthImagHist2(0),

  fMinInterpKStar1(0), fMinInterpKStar2(0), fMinInterpRStar(0), fMinInterpTheta(0), fMinInterpReF0(0), fMinInterpImF0(0), fMinInterpD0(0),
  fMaxInterpKStar1(0), fMaxInterpKStar2(0), fMaxInterpRStar(0), fMaxInterpTheta(0), fMaxInterpReF0(0), fMaxInterpImF0(0), fMaxInterpD0(0),

  fLednickyHFunction(0),

  fGTildeReal(0),
  fGTildeImag(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fCoulombScatteringLengthReal(0),
  fCoulombScatteringLengthImag(0),

  fCoulombScatteringLengthRealSub(0),
  fCoulombScatteringLengthImagSub(0),

  fCfsToFit(fNAnalyses),
  fFits(fNAnalyses),
  fMaxFitKStar(aMaxFitKStar),
  fRejectOmega(false),
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
  gRandom->SetSeed();

  fWaveFunction = new WaveFunction();
  CheckIfAllOfSameCoulombType();

  for(int i=0; i<fNAnalyses; i++)
  {
    //fCfsToFit[i] = fFitSharedAnalyses->GetPairAnalysis(i)->GetCf();
  }

  omp_set_num_threads(3);

}


//________________________________________________________________________________________________________________
CoulombFitter::~CoulombFitter()
{
  cout << "CoulombFitter object is being deleted!!!" << endl;

  //---Clean up
  if(!fCreateInterpVectors && fInterpHistsLoaded)
  {
    delete fLednickyHFunctionHist;

    fInterpHistFileLednickyHFunction->Close();
    delete fInterpHistFileLednickyHFunction;

    delete fHyperGeo1F1RealHist;
    delete fHyperGeo1F1ImagHist;

    delete fGTildeRealHist;
    delete fGTildeImagHist;

    fInterpHistFile->Close();
    delete fInterpHistFile;

    delete fCoulombScatteringLengthRealHist1;
    delete fCoulombScatteringLengthImagHist1;

    delete fCoulombScatteringLengthRealHist2;
    delete fCoulombScatteringLengthImagHist2;

    if(fUseScattLenHists)
    {
      fInterpHistFileScatLenReal1->Close();
      delete fInterpHistFileScatLenReal1;

      fInterpHistFileScatLenImag1->Close();
      delete fInterpHistFileScatLenImag1;

      fInterpHistFileScatLenReal2->Close();
      delete fInterpHistFileScatLenReal2;

      fInterpHistFileScatLenImag2->Close();
      delete fInterpHistFileScatLenImag2;
    }
  }

}



//________________________________________________________________________________________________________________
CoulombType CoulombFitter::GetCoulombType(AnalysisType aAnalysisType)
{
//  assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
  if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) return kAttractive; //attractive
  else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP) return kRepulsive; //repulsive
  else return kNeutral;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetBohrRadius(CoulombType aCoulombType)
{
  double tBohrRadius;
  if(aCoulombType == kAttractive) tBohrRadius = -gBohrRadiusXiK;
  else if(aCoulombType == kRepulsive) tBohrRadius = gBohrRadiusXiK;
  else if(aCoulombType == kNeutral) tBohrRadius = 1000000000;
  else
  {
    cout << "ERROR in GetBohrRadius:  Invalid fCoulombType selected" << endl;
    assert(0);
  }

  return tBohrRadius;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetBohrRadius(AnalysisType aAnalysisType)
{
  CoulombType tCoulombType = GetCoulombType(aAnalysisType);
  double tBohrRadius = GetBohrRadius(tCoulombType);
  return tBohrRadius;
}

//________________________________________________________________________________________________________________
void CoulombFitter::CheckIfAllOfSameCoulombType()
{
  CoulombType tCoulombType0, tCoulombType1;

  bool tAllSame = true;
  for(int iAnaly1=1; iAnaly1<fNAnalyses; iAnaly1++)
  {
    tCoulombType0 = GetCoulombType(fFitSharedAnalyses->GetFitPairAnalysis(iAnaly1-1)->GetAnalysisType());
    tCoulombType1 = GetCoulombType(fFitSharedAnalyses->GetFitPairAnalysis(iAnaly1)->GetAnalysisType());

    if(tCoulombType0 != tCoulombType1) tAllSame = false;
  }

  fAllOfSameCoulombType = tAllSame;
  if(fAllOfSameCoulombType)
  {
    fCoulombType = tCoulombType1;
    fBohrRadius = GetBohrRadius(fCoulombType);
    fWaveFunction->SetCurrentAnalysisType(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetAnalysisType());
  }

}


//________________________________________________________________________________________________________________
void CoulombFitter::LoadLednickyHFunctionFile(TString aFileBaseName)
{
  TString tFileName = aFileBaseName+".root";
  fInterpHistFileLednickyHFunction = TFile::Open(tFileName);

  fLednickyHFunctionHist = (TH1D*)fInterpHistFileLednickyHFunction->Get("LednickyHFunction");
//    fLednickyHFunctionHist->SetDirectory(0);

//  fInterpHistFileLednickyHFunction->Close();
}

//________________________________________________________________________________________________________________
void CoulombFitter::MakeNiceScattLenVectors()
{
ChronoTimer tTimer(kSec);
tTimer.Start();
  cout << "Starting MakeNiceScattLenVectors" << endl;
  //--------------------------------------------------------------
  fScattLenInfo.nBinsK = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins();
  fScattLenInfo.binWidthK = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinWidth(1);
  fScattLenInfo.minK = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinLowEdge(1);
  fScattLenInfo.minInterpK = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(1);
  fScattLenInfo.maxK = fCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinUpEdge(fCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());
  fScattLenInfo.maxInterpK = fCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(fCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());

  fScattLenInfo.nBinsReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins();
  fScattLenInfo.binWidthReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinWidth(1);
  fScattLenInfo.minReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinLowEdge(1);
  fScattLenInfo.minInterpReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(1);
  fScattLenInfo.maxReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinUpEdge(fCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());
  fScattLenInfo.maxInterpReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());

  fScattLenInfo.nBinsImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins();
  fScattLenInfo.binWidthImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinWidth(1);
  fScattLenInfo.minImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinLowEdge(1);
  fScattLenInfo.minInterpImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(1);
  fScattLenInfo.maxImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinUpEdge(fCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());
  fScattLenInfo.maxInterpImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());

  fScattLenInfo.nBinsD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins();
  fScattLenInfo.binWidthD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinWidth(1);
  fScattLenInfo.minD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinLowEdge(1);
  fScattLenInfo.minInterpD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(1);
  fScattLenInfo.maxD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinUpEdge(fCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());
  fScattLenInfo.maxInterpD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());
  //--------------------------------------------------------------

  int tNbinsReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins();
  int tNbinsImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins();
  int tNbinsD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins();
  int tNbinsK = 2*fCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins();

  fCoulombScatteringLengthReal.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));
  fCoulombScatteringLengthImag.resize(tNbinsReF0, td3dVec(tNbinsImF0, td2dVec(tNbinsD0, td1dVec(tNbinsK,0))));

cout << "fCoulombScatteringLengthReal.size() = " << fCoulombScatteringLengthReal.size() << endl;
cout << "fCoulombScatteringLengthReal[0].size() = " << fCoulombScatteringLengthReal[0].size() << endl;
cout << "fCoulombScatteringLengthReal[0][0].size() = " << fCoulombScatteringLengthReal[0][0].size() << endl;
cout << "fCoulombScatteringLengthReal[0][0][0].size() = " << fCoulombScatteringLengthReal[0][0][0].size() << endl;

  int tBin[4];
  int iK, iReF0, iImF0, iD0;
  #pragma omp parallel for private(iK,iReF0,iImF0,iD0) firstprivate(tBin)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iReF0=1; iReF0<=tNbinsReF0; iReF0++)
    {
      for(iImF0=1; iImF0<=tNbinsImF0; iImF0++)
      {
        for(iD0=1; iD0<=tNbinsD0; iD0++)
        {
          tBin[0] = iK;
          tBin[1] = iReF0;
          tBin[2] = iImF0;
          tBin[3] = iD0;

          if(iK <= tNbinsK/2)
          {
            fCoulombScatteringLengthReal[iReF0-1][iImF0-1][iD0-1][iK-1] = fCoulombScatteringLengthRealHist1->GetBinContent(tBin);
            fCoulombScatteringLengthImag[iReF0-1][iImF0-1][iD0-1][iK-1] = fCoulombScatteringLengthImagHist1->GetBinContent(tBin);
          }
          else
          {
	    tBin[0] = iK-tNbinsK/2;
            fCoulombScatteringLengthReal[iReF0-1][iImF0-1][iD0-1][iK-1] = fCoulombScatteringLengthRealHist2->GetBinContent(tBin);
            fCoulombScatteringLengthImag[iReF0-1][iImF0-1][iD0-1][iK-1] = fCoulombScatteringLengthImagHist2->GetBinContent(tBin);
          }
        }
      }
    }
  }
//  delete[] tBin;  //TODO NO!!!!!!!!!!!!!!! This causes "double free or corruption"!!!!!!!!!

//--------------------------------------------------------------
tTimer.Stop();
cout << "MakeNiceScattLenVectors: ";
tTimer.PrintInterval();
}


//________________________________________________________________________________________________________________
void CoulombFitter::MakeOtherVectors()
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting MakeOtherVectors" << endl;

//--------------------------------------------------------------
  fHyperGeo1F1Info.nBinsK = fHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthK = fHyperGeo1F1RealHist->GetXaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minK = fHyperGeo1F1RealHist->GetXaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpK = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxK = fHyperGeo1F1RealHist->GetXaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsX());
  fHyperGeo1F1Info.maxInterpK = fHyperGeo1F1RealHist->GetXaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsX());

  fHyperGeo1F1Info.nBinsR = fHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthR = fHyperGeo1F1RealHist->GetYaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minR = fHyperGeo1F1RealHist->GetYaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpR = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxR = fHyperGeo1F1RealHist->GetYaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsY());
  fHyperGeo1F1Info.maxInterpR = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fHyperGeo1F1Info.nBinsTheta = fHyperGeo1F1RealHist->GetZaxis()->GetNbins();
  fHyperGeo1F1Info.binWidthTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinWidth(1);
  fHyperGeo1F1Info.minTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinLowEdge(1);
  fHyperGeo1F1Info.minInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fHyperGeo1F1Info.maxTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinUpEdge(fHyperGeo1F1RealHist->GetNbinsZ());
  fHyperGeo1F1Info.maxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());

  //-----

  fGTildeInfo.nBinsK = fGTildeRealHist->GetXaxis()->GetNbins();
  fGTildeInfo.binWidthK = fGTildeRealHist->GetXaxis()->GetBinWidth(1);
  fGTildeInfo.minK = fGTildeRealHist->GetXaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpK = fGTildeRealHist->GetXaxis()->GetBinCenter(1);
  fGTildeInfo.maxK = fGTildeRealHist->GetXaxis()->GetBinUpEdge(fGTildeRealHist->GetNbinsX());
  fGTildeInfo.maxInterpK = fGTildeRealHist->GetXaxis()->GetBinCenter(fGTildeRealHist->GetNbinsX());

  fGTildeInfo.nBinsR = fGTildeRealHist->GetYaxis()->GetNbins();
  fGTildeInfo.binWidthR = fGTildeRealHist->GetYaxis()->GetBinWidth(1);
  fGTildeInfo.minR = fGTildeRealHist->GetYaxis()->GetBinLowEdge(1);
  fGTildeInfo.minInterpR = fGTildeRealHist->GetYaxis()->GetBinCenter(1);
  fGTildeInfo.maxR = fGTildeRealHist->GetYaxis()->GetBinUpEdge(fGTildeRealHist->GetNbinsY());
  fGTildeInfo.maxInterpR = fGTildeRealHist->GetYaxis()->GetBinCenter(fGTildeRealHist->GetNbinsY());

//--------------------------------------------------------------

  int tNbinsK = fHyperGeo1F1RealHist->GetXaxis()->GetNbins();
  int tNbinsR = fHyperGeo1F1RealHist->GetYaxis()->GetNbins();
  int tNbinsTheta = fHyperGeo1F1RealHist->GetZaxis()->GetNbins();

  fHyperGeo1F1Real.resize(tNbinsK, td2dVec(tNbinsR, td1dVec(tNbinsTheta,0)));
  fHyperGeo1F1Imag.resize(tNbinsK, td2dVec(tNbinsR, td1dVec(tNbinsTheta,0)));

  int iK, iR, iTheta;
  #pragma omp parallel for private(iK,iR,iTheta)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iR=1; iR<=tNbinsR; iR++)
    {
      for(iTheta=1; iTheta<=tNbinsTheta; iTheta++)
      {
        fHyperGeo1F1Real[iK-1][iR-1][iTheta-1] = fHyperGeo1F1RealHist->GetBinContent(iK,iR,iTheta);
        fHyperGeo1F1Imag[iK-1][iR-1][iTheta-1] = fHyperGeo1F1ImagHist->GetBinContent(iK,iR,iTheta);
      }
    }
  }

  fGTildeReal.resize(tNbinsK, td1dVec(tNbinsR,0));
  fGTildeImag.resize(tNbinsK, td1dVec(tNbinsR,0));

  #pragma omp parallel for private(iK,iR)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    for(iR=1; iR<=tNbinsR; iR++)
    {
      fGTildeReal[iK-1][iR-1] = fGTildeRealHist->GetBinContent(iK,iR);
      fGTildeImag[iK-1][iR-1] = fGTildeImagHist->GetBinContent(iK,iR);
    }
  }

  fLednickyHFunction.resize(tNbinsK,0.);
  #pragma omp parallel for private(iK)
  for(iK=1; iK<=tNbinsK; iK++)
  {
    fLednickyHFunction[iK-1] = fLednickyHFunctionHist->GetBinContent(iK);
  }

//--------------------------------------------------------------
tTimer.Stop();
cout << "MakeOtherVectors: ";
tTimer.PrintInterval();
}


//________________________________________________________________________________________________________________
void CoulombFitter::LoadInterpHistFile(TString aFileBaseName)
{
ChronoTimer tTimer(kSec);
tTimer.Start();
cout << "Starting LoadInterpHistFile" << endl;
cout << "\t fCreateInterpVectors = " << fCreateInterpVectors << endl;
cout << "\t fUseScattLenHists    = " << fUseScattLenHists << endl;

//--------------------------------------------------------------

  LoadLednickyHFunctionFile();

  TString aFileName = aFileBaseName+".root";
  fInterpHistFile = TFile::Open(aFileName);
//--------------------------------------------------------------
  fHyperGeo1F1RealHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Real");
  fHyperGeo1F1ImagHist = (TH3D*)fInterpHistFile->Get("HyperGeo1F1Imag");
//    fHyperGeo1F1RealHist->SetDirectory(0);
//    fHyperGeo1F1ImagHist->SetDirectory(0);

  fGTildeRealHist = (TH2D*)fInterpHistFile->Get("GTildeReal");
  fGTildeImagHist = (TH2D*)fInterpHistFile->Get("GTildeImag");
//    fGTildeRealHist->SetDirectory(0);
//    fGTildeImagHist->SetDirectory(0);

//  fInterpHistFile->Close();

  fMinInterpKStar1 = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(1);
  fMaxInterpKStar2 = fLednickyHFunctionHist->GetXaxis()->GetBinCenter(fLednickyHFunctionHist->GetNbinsX());

  fMinInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(1);
  fMaxInterpRStar = fHyperGeo1F1RealHist->GetYaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsY());

  fMinInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(1);
  fMaxInterpTheta = fHyperGeo1F1RealHist->GetZaxis()->GetBinCenter(fHyperGeo1F1RealHist->GetNbinsZ());

//--------------------------------------------------------------


  //--------------------------------------------------------------
  if(fUseScattLenHists)
  {
    //No SetDirectory call for THn, so unfortunately these files must remain open 

    TString aFileNameScatLenReal1 = aFileBaseName+"ScatLenReal1.root";  // 0. < k* < 0.2
    TString aFileNameScatLenImag1 = aFileBaseName+"ScatLenImag1.root";

    TString aFileNameScatLenReal2 = aFileBaseName+"ScatLenReal2.root";  // 0.2 < k* < 0.4
    TString aFileNameScatLenImag2 = aFileBaseName+"ScatLenImag2.root";

    fInterpHistFileScatLenReal1 = TFile::Open(aFileNameScatLenReal1);
    fCoulombScatteringLengthRealHist1 = (THnD*)fInterpHistFileScatLenReal1->Get("CoulombScatteringLengthReal");

    fInterpHistFileScatLenImag1 = TFile::Open(aFileNameScatLenImag1);
    fCoulombScatteringLengthImagHist1 = (THnD*)fInterpHistFileScatLenImag1->Get("CoulombScatteringLengthImag");

    fInterpHistFileScatLenReal2 = TFile::Open(aFileNameScatLenReal2);
    fCoulombScatteringLengthRealHist2 = (THnD*)fInterpHistFileScatLenReal2->Get("CoulombScatteringLengthReal");

    fInterpHistFileScatLenImag2 = TFile::Open(aFileNameScatLenImag2);
    fCoulombScatteringLengthImagHist2 = (THnD*)fInterpHistFileScatLenImag2->Get("CoulombScatteringLengthImag");

    //--------------------------------------------------------------

    fMinInterpKStar1 = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(1);
    fMaxInterpKStar1 = fCoulombScatteringLengthRealHist1->GetAxis(0)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(0)->GetNbins());

    fMinInterpKStar2 = fCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(1);
    fMaxInterpKStar2 = fCoulombScatteringLengthRealHist2->GetAxis(0)->GetBinCenter(fCoulombScatteringLengthRealHist2->GetAxis(0)->GetNbins());

    fMinInterpReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(1);
    fMaxInterpReF0 = fCoulombScatteringLengthRealHist1->GetAxis(1)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(1)->GetNbins());

    fMinInterpImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(1);
    fMaxInterpImF0 = fCoulombScatteringLengthRealHist1->GetAxis(2)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(2)->GetNbins());

    fMinInterpD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(1);
    fMaxInterpD0 = fCoulombScatteringLengthRealHist1->GetAxis(3)->GetBinCenter(fCoulombScatteringLengthRealHist1->GetAxis(3)->GetNbins());
 
    //--------------------------------------------------------------
  }

  if(fCreateInterpVectors)
  {
    MakeOtherVectors();

    delete fHyperGeo1F1RealHist;
    delete fHyperGeo1F1ImagHist;

    delete fGTildeRealHist;
    delete fGTildeImagHist;

    fInterpHistFile->Close();
    delete fInterpHistFile;
    
    if(fUseScattLenHists)
    {
      MakeNiceScattLenVectors();

      delete fCoulombScatteringLengthRealHist1;
      delete fCoulombScatteringLengthImagHist1;

      delete fCoulombScatteringLengthRealHist2;
      delete fCoulombScatteringLengthImagHist2;

      fInterpHistFileScatLenReal1->Close();
      delete fInterpHistFileScatLenReal1;

      fInterpHistFileScatLenImag1->Close();
      delete fInterpHistFileScatLenImag1;

      fInterpHistFileScatLenReal2->Close();
      delete fInterpHistFileScatLenReal2;

      fInterpHistFileScatLenImag2->Close();
      delete fInterpHistFileScatLenImag2;
    }
  }


//--------------------------------------------------------------
  cout << "Interpolation histograms LOADED!" << endl;

tTimer.Stop();
cout << "LoadInterpHistFile: ";
tTimer.PrintInterval();

  fInterpHistsLoaded = true;
}


//________________________________________________________________________________________________________________
int CoulombFitter::GetBinNumber(double aBinSize, int aNbins, double aValue)
{
//TODO check the accuracy of this
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*aBinSize;
    tBinKStarMax = (i+1)*aBinSize;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int CoulombFitter::GetBinNumber(int aNbins, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  double tBinSize = (aMax-aMin)/aNbins;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<aNbins; i++)
  {
    tBinKStarMin = i*tBinSize + aMin;
    tBinKStarMax = (i+1)*tBinSize + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
int CoulombFitter::GetBinNumber(double aBinWidth, double aMin, double aMax, double aValue)
{
//TODO check the accuracy of this
  int tNbins = (aMax-aMin)/aBinWidth;
  double tBinKStarMin, tBinKStarMax;

  for(int i=0; i<tNbins; i++)
  {
    tBinKStarMin = i*aBinWidth + aMin;
    tBinKStarMax = (i+1)*aBinWidth + aMin;

    if(aValue>=tBinKStarMin && aValue<tBinKStarMax) return i;
  }

  return -1;  //i.e. failure
}

//________________________________________________________________________________________________________________
void CoulombFitter::ExtractPairKStar3dVecFromSingleFile(TString aFileLocation, TString aArrayName, TString aNtupleName, double aBinSizeKStar, double aNbinsKStar, td3dVec &aVecToFill)
{
  vector<double> tTempEntry;
  int tKStarBinNumber;
  //---------------------------------------------------

  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtoList = (TList*)tFile->Get("femtolist");
  TObjArray *tArray = (TObjArray*)tFemtoList->FindObject(aArrayName)->Clone();
    tArray->SetOwner();
  TNtuple *tPairKStarNtuple = (TNtuple*)tArray->FindObject(aNtupleName);
  //---------------------------------------------------

  float tTupleKStarMag, tTupleKStarOut, tTupleKStarSide, tTupleKStarLong;

  tPairKStarNtuple->SetBranchAddress("KStarMag", &tTupleKStarMag);
  tPairKStarNtuple->SetBranchAddress("KStarOut", &tTupleKStarOut);
  tPairKStarNtuple->SetBranchAddress("KStarSide", &tTupleKStarSide);
  tPairKStarNtuple->SetBranchAddress("KStarLong", &tTupleKStarLong);

  //--------------------------------------
  assert(aVecToFill.size() == aNbinsKStar);

  for(int i=0; i<tPairKStarNtuple->GetEntries(); i++)
  {
    tPairKStarNtuple->GetEntry(i);

    tKStarBinNumber = GetBinNumber(aBinSizeKStar, aNbinsKStar, tTupleKStarMag);
    if(tKStarBinNumber>=0)  //i.e, the KStarMag value is within my bins of interest
    {
      tTempEntry.clear();
        tTempEntry.push_back(tTupleKStarMag);
        tTempEntry.push_back(tTupleKStarOut);
        tTempEntry.push_back(tTupleKStarSide);
        tTempEntry.push_back(tTupleKStarLong);

      aVecToFill[tKStarBinNumber].push_back(tTempEntry);
    }
  }


  //Clean up----------------------------------------------------
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtoList object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtoList);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  tFemtoList->Delete();
  delete tFemtoList;

  tArray->Delete();
  delete tArray;

  //NOTE:  Apparently, tPairKStarNtuple is deleted when file is closed
  //       In fact, trying to delete it manually causes errors!

  tFile->Close();
  delete tFile;
  //------------------------------------------------------------
}

//________________________________________________________________________________________________________________
td3dVec CoulombFitter::BuildPairKStar3dVecFull(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  cout << "Beginning FULL conversion of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory

  //---------------------------------------------------
  assert(aNFiles==27);

  vector<TString> tRomanNumerals(17);
    tRomanNumerals[0] = "I";
    tRomanNumerals[1] = "II";
    tRomanNumerals[2] = "III";
    tRomanNumerals[3] = "IV";
    tRomanNumerals[4] = "V";
    tRomanNumerals[5] = "VI";
    tRomanNumerals[6] = "VII";
    tRomanNumerals[7] = "VIII";
    tRomanNumerals[8] = "IX";
    tRomanNumerals[9] = "X";
    tRomanNumerals[10] = "XI";
    tRomanNumerals[11] = "XII";
    tRomanNumerals[12] = "XIII";
    tRomanNumerals[13] = "XIV";
    tRomanNumerals[14] = "XV";
    tRomanNumerals[15] = "XVI";
    tRomanNumerals[16] = "XVII";
  //---------------------------------------------------

  double tBinSize = (aKStarMax-aKStarMin)/aNbinsKStar;
cout << "tBinSize = " << tBinSize << endl;

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();
  tPairKStar3dVec.resize(aNbinsKStar, td2dVec(0,td1dVec(0)));

cout << "Pre: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;

  //---------------------------------------------------
  TString tFileLocation;

  TString tArrayName = TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]);
  TString tNtupleName = "PairKStarKStarCf_"+TString(cAnalysisBaseTags[aAnalysisType]);

  for(int iFile=0; iFile<17; iFile++) //Bm files
  {
    cout << "\t iFile = Bm" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bm") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,tBinSize,aNbinsKStar,tPairKStar3dVec);
  }

  for(int iFile=0; iFile<10; iFile++) //Bp files
  {
    cout << "\t iFile = Bp" << iFile << endl;
    tFileLocation = aPairKStarNtupleDirName + TString("/") + aFileBaseName + TString("_Bp") + tRomanNumerals[iFile] + TString(".root");
    ExtractPairKStar3dVecFromSingleFile(tFileLocation,tArrayName,tNtupleName,tBinSize,aNbinsKStar,tPairKStar3dVec);
  }

  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Conversion complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
void CoulombFitter::WriteRow(ostream &aOutput, vector<double> &aRow)
{
  for(int i = 0; i < (int)aRow.size(); i++)
  {
    if( i < (int)aRow.size()-1) aOutput << aRow[i] << " ";
    else if(i == (int)aRow.size()-1) aOutput << aRow[i] << endl;
    else cout << "SOMETHING IS WRONG!!!!!\n";
  }
}



//________________________________________________________________________________________________________________
void CoulombFitter::WritePairKStar3dVecFile(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  TString tOutputName = aOutputBaseName + TString(cAnalysisBaseTags[aAnalysisType])+TString(cCentralityTags[aCentralityType]) + TString(".txt");
  ofstream tFileOut(tOutputName);

  cout << "Beginning FULL writing of TNtuple to 3dVec" << endl;
  std::clock_t start = std::clock();

  //----****----****----****----****----****----****
  //TODO make this automated
  //TODO add centrality selection to this
  //Also, this is basically FitPartialAnalysis::ConnectAnalysisDirectory

  td3dVec tPairKStar3dVec = BuildPairKStar3dVecFull(aPairKStarNtupleDirName,aFileBaseName,aNFiles,aAnalysisType,aCentralityType,aNbinsKStar,aKStarMin,aKStarMax);

  //---------------------------------------------------
  tFileOut << aNbinsKStar << " " << aKStarMin << " " << aKStarMax << endl;

  vector<double> tTempPair;
  for(int iBin=0; iBin<(int)tPairKStar3dVec.size(); iBin++)
  {
    tFileOut << iBin << " " << tPairKStar3dVec[iBin].size() << endl;
    for(int iPair=0; iPair<(int)tPairKStar3dVec[iBin].size(); iPair++)
    {
      WriteRow(tFileOut,tPairKStar3dVec[iBin][iPair]);
    }
  }

  tFileOut.close();
  //---------------------------------------------------
  double duration = (std::clock() - start)/(double) CLOCKS_PER_SEC;
  cout << "Writing complete in " << duration << " seconds" << endl;


  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<aNbinsKStar; i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;
}

//________________________________________________________________________________________________________________
void CoulombFitter::WriteAllPairKStar3dVecFiles(TString aOutputBaseName, TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();
    WritePairKStar3dVecFile(aOutputBaseName,aPairKStarNtupleDirName,aFileBaseName,aNFiles,tAnalysisType,tCentralityType,aNbinsKStar,aKStarMin,aKStarMax);
  }
}

//________________________________________________________________________________________________________________
td3dVec CoulombFitter::BuildPairKStar3dVecFromTxt(TString aFileName)
{
ChronoTimer tTimer;
tTimer.Start();

  ifstream tFileIn(aFileName);

  td3dVec tPairKStar3dVec;
    tPairKStar3dVec.clear();

  vector<vector<double> > tTempBin2dVec;
  vector<double> tTempPair1dVec;

  int aNbinsKStar;
  double aKStarMin, aKStarMax, aBinWidth;

  int tNbinsKStarNeeded = 200; //This will be set to correct value below

  string tString;
  int tCount = 0;
  while(getline(tFileIn, tString) && tCount <= tNbinsKStarNeeded)
  {
    tTempPair1dVec.clear();
    istringstream tStream(tString);
    string tElement;
    while(tStream >> tElement)
    {
      stringstream ss (tElement);
      double dbl;
      ss >> dbl;
      tTempPair1dVec.push_back(dbl);
    }

    if(tTempPair1dVec.size() == 2)  //bin header
    {           
      tCount++;
      if(tCount==1) continue;
      else
      {
        tPairKStar3dVec.push_back(tTempBin2dVec);
        tTempBin2dVec.clear();
      }
    }
    else if(tTempPair1dVec.size() == 4)  //pair
    {
      tTempBin2dVec.push_back(tTempPair1dVec);
    }
    else if(tTempPair1dVec.size() == 3) //File header
    {
      aNbinsKStar = tTempPair1dVec[0];
      aKStarMin = tTempPair1dVec[1];
      aKStarMax = tTempPair1dVec[2];
      aBinWidth = (aKStarMax-aKStarMin)/aNbinsKStar;
      tNbinsKStarNeeded = fMaxFitKStar/aBinWidth;
      tNbinsKStarNeeded ++;  //include one more bin than really needed, to be safe
      assert(tNbinsKStarNeeded <= aNbinsKStar); //cannot need more than we have
    }
    else
    {
      cout << "ERROR: Incorrect row size in BuildPairKStar3dVecFromTxt" << endl;
      assert(0);
    }
  }
  if(tNbinsKStarNeeded == aNbinsKStar) tPairKStar3dVec.push_back(tTempBin2dVec);  //if reading the entire file, need this final push_back
  tTempBin2dVec.clear();

tTimer.Stop();
cout << "BuildPairKStar3dVecFromTxt finished: ";
tTimer.PrintInterval();

  assert(tNbinsKStarNeeded == (int)tPairKStar3dVec.size());
  cout << "Final: tPairKStar3dVec.size() = " << tPairKStar3dVec.size() << endl;
  for(int i=0; i<(int)tPairKStar3dVec.size(); i++) cout << "i = " << i << "and tPairKStar3dVec[i].size() = " << tPairKStar3dVec[i].size() << endl;

  //Add the binning information for use by CoulombFitterParallel::BuildPairKStar3dVecFromTxt
  tTempPair1dVec.clear();
  tTempPair1dVec.push_back(aNbinsKStar);
  tTempPair1dVec.push_back(aKStarMin);
  tTempPair1dVec.push_back(aKStarMax);
  tTempBin2dVec.push_back(tTempPair1dVec);
  tPairKStar3dVec.push_back(tTempBin2dVec);

  tTempPair1dVec.clear();
  tTempBin2dVec.clear();

  return tPairKStar3dVec;
}

//________________________________________________________________________________________________________________
void CoulombFitter::BuildPairKStar4dVecFromTxt(TString aFileBaseName)
{
  fPairKStar4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  TString tFileName;

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();
    tFileName = aFileBaseName + TString(cAnalysisBaseTags[tAnalysisType])+TString(cCentralityTags[tCentralityType]) + TString(".txt");

    td3dVec tPairKStar3dVec = BuildPairKStar3dVecFromTxt(tFileName);
      tPairKStar3dVec.pop_back();  //strip off the binning information

    fPairKStar4dVec[iAnaly] = tPairKStar3dVec;
  }

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    cout << "iAnaly = " << iAnaly << endl;
    cout << "fPairKStar4dVec[iAnaly].size() = " << fPairKStar4dVec[iAnaly].size() << endl;
    for(int i=0; i<(int)fPairKStar4dVec[iAnaly].size(); i++)
    {
      cout << "\t i = " << i << endl;
      cout << "\t\tfPairKStar4dVec[iAnaly][i].size() = " << fPairKStar4dVec[iAnaly][i].size() << endl;
    }
  }
}



//________________________________________________________________________________________________________________
void CoulombFitter::BuildPairKStar4dVecOnFly(TString aPairKStarNtupleDirName, TString aFileBaseName, int aNFiles, int aNbinsKStar, double aKStarMin, double aKStarMax)
{
  fPairKStar4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  AnalysisType tAnalysisType;
  CentralityType tCentralityType;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tAnalysisType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetAnalysisType();
    tCentralityType = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->GetCentralityType();

    td3dVec tPairKStar3dVec = BuildPairKStar3dVecFull(aPairKStarNtupleDirName,aFileBaseName,aNFiles,tAnalysisType,tCentralityType,aNbinsKStar,aKStarMin,aKStarMax);
    fPairKStar4dVec[iAnaly] = tPairKStar3dVec;
  }

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    cout << "iAnaly = " << iAnaly << endl;
    cout << "fPairKStar4dVec[iAnaly].size() = " << fPairKStar4dVec[iAnaly].size() << endl;
    for(int i=0; i<(int)fPairKStar4dVec[iAnaly].size(); i++)
    {
      cout << "\t i = " << i << endl;
      cout << "\t\tfPairKStar4dVec[iAnaly][i].size() = " << fPairKStar4dVec[iAnaly][i].size() << endl;
    }
  }

}


//________________________________________________________________________________________________________________
void CoulombFitter::BuildPairSample4dVec(int aNPairsPerKStarBin)
{
ChronoTimer tTimer(kSec);
tTimer.Start();

  fNPairsPerKStarBin = aNPairsPerKStarBin;
  fPairSample4dVec.resize(fNAnalyses, td3dVec(0, td2dVec(0, td1dVec(0))));

  double tBinSize = 0.01;  //TODO make this automated
  int tNBinsKStar = fMaxFitKStar/tBinSize;  //TODO make this general, ie subtract 1 if fMaxFitKStar is on bin edge (maybe, maybe not bx of iKStarBin<tNBinsKStar)

  //Create the source Gaussians
  double tRoot2 = sqrt(2.);  //need this scale to get 4 on denominator of exp in normal dist instead of 2
  double tRadius = 1.0;
  fCurrentRadiusParameter = tRadius;
  std::default_random_engine generator (std::clock());  //std::clock() is seed
  std::normal_distribution<double> tROutSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRSideSource(0.,tRoot2*tRadius);
  std::normal_distribution<double> tRLongSource(0.,tRoot2*tRadius);

  std::uniform_int_distribution<int> tRandomKStarElement;

  TVector3* tKStar3Vec = new TVector3(0.,0.,0.);
  TVector3* tSource3Vec = new TVector3(0.,0.,0.);

  int tI;
  double tTheta, tKStarMag, tRStarMag;
  double tKStarMagMin, tKStarMagMax;
  td1dVec tTempPair(3);
  td2dVec tTemp2dVec;
  td3dVec tTemp3dVec;

  //------------------------------------
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    tTemp3dVec.clear();
    for(int iKStarBin=0; iKStarBin<tNBinsKStar; iKStarBin++)
    {
      if(!fUseRandomKStarVectors) tRandomKStarElement = std::uniform_int_distribution<int>(0.0, fPairKStar4dVec[iAnaly][iKStarBin].size()-1);
      tKStarMagMin = iKStarBin*tBinSize;
      tKStarMagMax = (iKStarBin+1)*tBinSize;
      tTemp2dVec.clear();
      for(int iPair=0; iPair<fNPairsPerKStarBin; iPair++)
      {
        if(!fUseRandomKStarVectors)
        {
          tI = tRandomKStarElement(generator);
          tKStar3Vec->SetXYZ(fPairKStar4dVec[iAnaly][iKStarBin][tI][1],fPairKStar4dVec[iAnaly][iKStarBin][tI][2],fPairKStar4dVec[iAnaly][iKStarBin][tI][3]);
        }
        else SetRandomKStar3Vec(tKStar3Vec,tKStarMagMin,tKStarMagMax);

        tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric

        tTheta = tKStar3Vec->Angle(*tSource3Vec);
        tKStarMag = tKStar3Vec->Mag();
        tRStarMag = tSource3Vec->Mag();

        tTempPair[0] = tKStarMag;
        tTempPair[1] = tRStarMag;
        tTempPair[2] = tTheta;

        tTemp2dVec.push_back(tTempPair);
      }
      tTemp3dVec.push_back(tTemp2dVec);
    }
    fPairSample4dVec[iAnaly] = tTemp3dVec;
  }

  delete tKStar3Vec;
  delete tSource3Vec;


  //----Set the bin info
  fSamplePairsBinInfo.nAnalyses = fNAnalyses;
  fSamplePairsBinInfo.nBinsK = tNBinsKStar;
  fSamplePairsBinInfo.nPairsPerBin = fNPairsPerKStarBin;

  fSamplePairsBinInfo.minK = 0.; //TODO
  fSamplePairsBinInfo.maxK = fMaxFitKStar;
  fSamplePairsBinInfo.binWidthK = tBinSize;
  fSamplePairsBinInfo.nElementsPerPair = fPairSample4dVec[0][0][0].size();
  //--------------------


tTimer.Stop();
cout << "BuildPairSample4dVec finished: ";
tTimer.PrintInterval();

}

//________________________________________________________________________________________________________________
void CoulombFitter::UpdatePairRadiusParameters(double aNewRadius)
{
  //TODO allow for change of MU also, not just SIGMA!
  double tScaleFactor = aNewRadius/fCurrentRadiusParameter;
  fCurrentRadiusParameter = aNewRadius;

//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    for(int iKStarBin=0; iKStarBin<(int)fPairSample4dVec[iAnaly].size(); iKStarBin++)
    {
      for(int iPair=0; iPair<(int)fPairSample4dVec[iAnaly][iKStarBin].size(); iPair++)
      {
        fPairSample4dVec[iAnaly][iKStarBin][iPair][1] *= tScaleFactor;
      }
    }
  }

}

//________________________________________________________________________________________________________________
void CoulombFitter::SetUseStaticPairs(bool aUseStaticPairs, int aNPairsPerKStarBin)
{
  fUseStaticPairs = aUseStaticPairs;
  BuildPairSample4dVec(aNPairsPerKStarBin);
}


//________________________________________________________________________________________________________________
int CoulombFitter::GetInterpLowBin(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fGTildeInfo.nBinsK;
          tBinWidth = fGTildeInfo.binWidthK;
          tMin = fGTildeInfo.minK;
          tMax = fGTildeInfo.maxK;
          break;

        case kRaxis:
          tNbins = fGTildeInfo.nBinsR;
          tBinWidth = fGTildeInfo.binWidthR;
          tMin = fGTildeInfo.minR;
          tMax = fGTildeInfo.maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fHyperGeo1F1Info.nBinsK;
          tBinWidth = fHyperGeo1F1Info.binWidthK;
          tMin = fHyperGeo1F1Info.minK;
          tMax = fHyperGeo1F1Info.maxK;
          break;

        case kRaxis:
          tNbins = fHyperGeo1F1Info.nBinsR;
          tBinWidth = fHyperGeo1F1Info.binWidthR;
          tMin = fHyperGeo1F1Info.minR;
          tMax = fHyperGeo1F1Info.maxR;
          break;

        case kThetaaxis:
          tNbins = fHyperGeo1F1Info.nBinsTheta;
          tBinWidth = fHyperGeo1F1Info.binWidthTheta;
          tMin = fHyperGeo1F1Info.minTheta;
          tMax = fHyperGeo1F1Info.maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = fScattLenInfo.nBinsReF0;
          tBinWidth = fScattLenInfo.binWidthReF0;
          tMin = fScattLenInfo.minReF0;
          tMax = fScattLenInfo.maxReF0;
          break;

        case kImF0axis:
          tNbins = fScattLenInfo.nBinsImF0;
          tBinWidth = fScattLenInfo.binWidthImF0;
          tMin = fScattLenInfo.minImF0;
          tMax = fScattLenInfo.maxImF0;
          break;

        case kD0axis:
          tNbins = fScattLenInfo.nBinsD0;
          tBinWidth = fScattLenInfo.binWidthD0;
          tMin = fScattLenInfo.minD0;
          tMax = fScattLenInfo.maxD0;
          break;

        case kKaxis:
          tNbins = fScattLenInfo.nBinsK;
          tBinWidth = fScattLenInfo.binWidthK;
          tMin = fScattLenInfo.minK;
          tMax = fScattLenInfo.maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;
  else return tReturnBin;

}

//________________________________________________________________________________________________________________
double CoulombFitter::GetInterpLowBinCenter(InterpType aInterpType, InterpAxisType aAxisType, double aVal)
{
  double tReturnValue;
  int tReturnBin = -2;

  int tNbins, tBin;
  double tMin, tMax, tBinWidth, tBinCenter;

  bool tErrorFlag = false;

  switch(aInterpType)
  {
    case kGTilde:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fGTildeInfo.nBinsK;
          tBinWidth = fGTildeInfo.binWidthK;
          tMin = fGTildeInfo.minK;
          tMax = fGTildeInfo.maxK;
          break;

        case kRaxis:
          tNbins = fGTildeInfo.nBinsR;
          tBinWidth = fGTildeInfo.binWidthR;
          tMin = fGTildeInfo.minR;
          tMax = fGTildeInfo.maxR;
          break;

        //Invalid axis selection
        case kThetaaxis:
          tErrorFlag = true;
          break;
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kHyperGeo1F1:
      switch(aAxisType)
      {
        case kKaxis:
          tNbins = fHyperGeo1F1Info.nBinsK;
          tBinWidth = fHyperGeo1F1Info.binWidthK;
          tMin = fHyperGeo1F1Info.minK;
          tMax = fHyperGeo1F1Info.maxK;
          break;

        case kRaxis:
          tNbins = fHyperGeo1F1Info.nBinsR;
          tBinWidth = fHyperGeo1F1Info.binWidthR;
          tMin = fHyperGeo1F1Info.minR;
          tMax = fHyperGeo1F1Info.maxR;
          break;

        case kThetaaxis:
          tNbins = fHyperGeo1F1Info.nBinsTheta;
          tBinWidth = fHyperGeo1F1Info.binWidthTheta;
          tMin = fHyperGeo1F1Info.minTheta;
          tMax = fHyperGeo1F1Info.maxTheta;
          break;

        //Invalid axis selection
        case kReF0axis:
          tErrorFlag = true;
          break;
        case kImF0axis:
          tErrorFlag = true;
          break;
        case kD0axis:
          tErrorFlag = true;
          break;
      }
      break;

    case kScattLen:
      switch(aAxisType)
      {
        case kReF0axis:
          tNbins = fScattLenInfo.nBinsReF0;
          tBinWidth = fScattLenInfo.binWidthReF0;
          tMin = fScattLenInfo.minReF0;
          tMax = fScattLenInfo.maxReF0;
          break;

        case kImF0axis:
          tNbins = fScattLenInfo.nBinsImF0;
          tBinWidth = fScattLenInfo.binWidthImF0;
          tMin = fScattLenInfo.minImF0;
          tMax = fScattLenInfo.maxImF0;
          break;

        case kD0axis:
          tNbins = fScattLenInfo.nBinsD0;
          tBinWidth = fScattLenInfo.binWidthD0;
          tMin = fScattLenInfo.minD0;
          tMax = fScattLenInfo.maxD0;
          break;

        case kKaxis:
          tNbins = fScattLenInfo.nBinsK;
          tBinWidth = fScattLenInfo.binWidthK;
          tMin = fScattLenInfo.minK;
          tMax = fScattLenInfo.maxK;
          break;


        //Invalid axis selection
        case kRaxis:
          tErrorFlag = true;
          break;
        case kThetaaxis:
          tErrorFlag = true;
          break;

      }
      break;
  }

  //Check error
  if(tErrorFlag) return -2;

  //---------------------------------
  tBin = GetBinNumber(tNbins,tMin,tMax,aVal);
  tBinCenter = tMin + (tBin+0.5)*tBinWidth;
  if(aVal < tBinCenter) tReturnBin = tBin-1;
  else tReturnBin = tBin;

  if(tReturnBin<0 || tReturnBin >= tNbins) return -2;

  tReturnValue = tMin + (tReturnBin+0.5)*tBinWidth;
  return tReturnValue;
}


/*
//________________________________________________________________________________________________________________
//TODO fix the entire functions
vector<int> CoulombFitter::GetRelevantKStarBinNumbers(double aKStarMagMin, double aKStarMagMax)
{
  int tMinBin = GetInterpLowBin(kScattLen,kKaxis,aKStarMagMin);
  int tMaxBin = GetInterpLowBin(kScattLen,kKaxis,aKStarMagMax);

  if(tMaxBin<(fScattLenInfo.nBinsK-1)) tMaxBin++;  //must include first bin over range
                                                   //Note:  GetInterpLowBin already return first underneath bin (if exists)
  //TODO fix this shit
  if(tMinBin<0) tMinBin=0;

  int tNbins = (tMaxBin-tMinBin)+1;  //+1 because inclusive, i.e. if bins 3,4,5,6,7,8 -> (8-3)+1 = 6 bins 
  vector<int> tReturnVector(tNbins);
  for(int i=0; i<tNbins; i++)
  {
    tReturnVector[i] = tMinBin+i;
  }
  return tReturnVector;
}
*/

//________________________________________________________________________________________________________________
//TODO fix the entire functions
vector<int> CoulombFitter::GetRelevantKStarBinNumbers(double aKStarMagMin, double aKStarMagMax)
{
  int tMinBin = GetBinNumber(fScattLenInfo.binWidthK,fScattLenInfo.minK,fScattLenInfo.maxK,aKStarMagMin);
  int tMaxBin = GetBinNumber(fScattLenInfo.binWidthK,fScattLenInfo.minK,fScattLenInfo.maxK,aKStarMagMax);

  if(tMaxBin<(fScattLenInfo.nBinsK-1)) tMaxBin++;  //must include first bin over range
                                                   //Note:  GetInterpLowBin already return first underneath bin (if exists)
  //TODO fix this shit
  if(tMinBin>0) tMinBin--;

  int tNbins = (tMaxBin-tMinBin)+1;  //+1 because inclusive, i.e. if bins 3,4,5,6,7,8 -> (8-3)+1 = 6 bins 
  vector<int> tReturnVector(tNbins);
  for(int i=0; i<tNbins; i++)
  {
    tReturnVector[i] = tMinBin+i;
  }
  return tReturnVector;
}




//________________________________________________________________________________________________________________
double CoulombFitter::LinearInterpolate(TH1* a1dHisto, double aX)
{
  if(a1dHisto->GetBuffer()) a1dHisto->BufferEmpty();  //not sure what this is all about

  int tXbin = a1dHisto->FindBin(aX);
  double tX0, tX1, tY0, tY1;

  //TODO: These allows evaluation in underflow and overflow bins, not sure I like this
  if(aX <= a1dHisto->GetBinCenter(1)) return a1dHisto->GetBinContent(1);
  else if( aX >= a1dHisto->GetBinCenter(a1dHisto->GetNbinsX()) ) return a1dHisto->GetBinContent(a1dHisto->GetNbinsX()); 

  else
  {
    if(aX <= a1dHisto->GetBinCenter(tXbin))
    {
      tY0 = a1dHisto->GetBinContent(tXbin-1);
      tX0 = a1dHisto->GetBinCenter(tXbin-1);
      tY1 = a1dHisto->GetBinContent(tXbin);
      tX1 = a1dHisto->GetBinCenter(tXbin);
    }
    else
    {
      tY0 = a1dHisto->GetBinContent(tXbin);
      tX0 = a1dHisto->GetBinCenter(tXbin);
      tY1 = a1dHisto->GetBinContent(tXbin+1);
      tX1 = a1dHisto->GetBinCenter(tXbin+1);
    }
    return tY0 + (aX-tX0)*((tY1-tY0)/(tX1-tX0));
  }
}

//________________________________________________________________________________________________________________
double CoulombFitter::BilinearInterpolate(TH2* a2dHisto, double aX, double aY)
{
  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  TAxis* tXaxis = a2dHisto->GetXaxis();
  TAxis* tYaxis = a2dHisto->GetYaxis();

  int tXbin = tXaxis->FindBin(aX);
  int tYbin = tYaxis->FindBin(aY);

  //---------------------------------
  if(tXbin<1 || tXbin>a2dHisto->GetNbinsX() || tYbin<1 || tYbin>a2dHisto->GetNbinsY()) 
  {
    cout << "Error in CoulombFitter::BilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin>0);
  assert(tXbin<=a2dHisto->GetNbinsX());
  assert(tYbin>0);
  assert(tYbin<=a2dHisto->GetNbinsY());
  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tXaxis->GetBinUpEdge(tXbin) - aX;
  tdY = tYaxis->GetBinUpEdge(tYbin) - aY;

  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 1; //upper right
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY<=tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 2; //upper left
  if(tdX>tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 3; //lower left
  if(tdX<=tXaxis->GetBinWidth(tXbin)/2 && tdY>tYaxis->GetBinWidth(tYbin)/2) tQuadrant = 4; //lower right

  switch(tQuadrant)
  {
    case 1:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 2:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin+1);
      break;
    case 3:
      tX1 = tXaxis->GetBinCenter(tXbin-1);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
    case 4:
      tX1 = tXaxis->GetBinCenter(tXbin);
      tY1 = tYaxis->GetBinCenter(tYbin-1);
      tX2 = tXaxis->GetBinCenter(tXbin+1);
      tY2 = tYaxis->GetBinCenter(tYbin);
      break;
  }

  int tBinX1 = tXaxis->FindBin(tX1);
  if(tBinX1<1) tBinX1 = 1;

  int tBinX2 = tXaxis->FindBin(tX2);
  if(tBinX2>a2dHisto->GetNbinsX()) tBinX2=a2dHisto->GetNbinsX();

  int tBinY1 = tYaxis->FindBin(tY1);
  if(tBinY1<1) tBinY1 = 1;

  int tBinY2 = tYaxis->FindBin(tY2);
  if(tBinY2>a2dHisto->GetNbinsY()) tBinY2=a2dHisto->GetNbinsY();

  int tBinQ22 = a2dHisto->GetBin(tBinX2,tBinY2);
  int tBinQ12 = a2dHisto->GetBin(tBinX1,tBinY2);
  int tBinQ11 = a2dHisto->GetBin(tBinX1,tBinY1);
  int tBinQ21 = a2dHisto->GetBin(tBinX2,tBinY1);

  double tQ11 = a2dHisto->GetBinContent(tBinQ11);
  double tQ12 = a2dHisto->GetBinContent(tBinQ12);
  double tQ21 = a2dHisto->GetBinContent(tBinQ21);
  double tQ22 = a2dHisto->GetBinContent(tBinQ22);

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}


//________________________________________________________________________________________________________________
double CoulombFitter::BilinearInterpolateVector(vector<vector<double> > &a2dVec, double aX, int aNbinsX, double aMinX, double aMaxX, double aY, int aNbinsY, double aMinY, double aMaxY)
{
  //NOTE: THIS IS SLOWER THAN BilinearInterpolate, but a method like this may be necessary for parallelization

  double tF = 0.;
  double tX1=0., tX2=0., tY1=0., tY2=0.;
  double tdX, tdY;

  int tXbin = GetBinNumber(aNbinsX,aMinX,aMaxX,aX);
  int tYbin = GetBinNumber(aNbinsY,aMinY,aMaxY,aY);

  double tBinWidthX = (aMaxX-aMinX)/aNbinsX;
  double tBinMinX = aMinX + tXbin*tBinWidthX;
  double tBinMaxX = aMinX + (tXbin+1)*tBinWidthX;

  double tBinWidthY = (aMaxY-aMinY)/aNbinsY;
  double tBinMinY = aMinY + tYbin*tBinWidthY;
  double tBinMaxY = aMinY + (tYbin+1)*tBinWidthY;

  //---------------------------------
  if(tXbin<0 || tYbin<0) 
  {
    cout << "Error in CoulombFitter::BilinearInterpolateVector, cannot interpolate outside histogram domain" << endl;
  }
  assert(tXbin >= 0);
  assert(tYbin >= 0);

  //---------------------------------

  int tQuadrant = 0; //CCW from UR 1,2,3,4
  // which quadrant of the bin (bin_P) are we in?
  tdX = tBinMaxX - aX;
  tdY = tBinMaxY - aY;

  int tBinX1, tBinX2, tBinY1, tBinY2;

  if(tdX<=tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 1; //upper right
  else if(tdX>tBinWidthX/2 && tdY<=tBinWidthY/2) tQuadrant = 2; //upper left
  else if(tdX>tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 3; //lower left
  else if(tdX<=tBinWidthX/2 && tdY>tBinWidthY/2) tQuadrant = 4; //lower right
  else cout << "ERROR IN BilinearInterpolateVector!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


  switch(tQuadrant)
  {
    case 1:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 2:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY + tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMaxY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin;
      tBinY2 = tYbin+1;

      break;
    case 3:
      tX1 = tBinMinX - tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMinX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin-1;
      tBinX2 = tXbin;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
    case 4:
      tX1 = tBinMinX + tBinWidthX/2;
      tY1 = tBinMinY - tBinWidthY/2;
      tX2 = tBinMaxX + tBinWidthX/2;
      tY2 = tBinMinY + tBinWidthY/2;

      tBinX1 = tXbin;
      tBinX2 = tXbin+1;
      tBinY1 = tYbin-1;
      tBinY2 = tYbin;

      break;
  }

  if(tBinX1<1) tBinX1 = 1;
  if(tBinX2>aNbinsX) tBinX2=aNbinsX;
  if(tBinY1<1) tBinY1 = 1;
  if(tBinY2>aNbinsY) tBinY2=aNbinsY;

  double tQ11 = a2dVec[tBinX1][tBinY1];
  double tQ12 = a2dVec[tBinX1][tBinY2];
  double tQ21 = a2dVec[tBinX2][tBinY1];
  double tQ22 = a2dVec[tBinX2][tBinY2];

  double tD = 1.0*(tX2-tX1)*(tY2-tY1);

  tF = (1.0/tD)*(tQ11*(tX2-aX)*(tY2-aY) + tQ21*(aX-tX1)*(tY2-aY) + tQ12*(tX2-aX)*(aY-tY1) + tQ22*(aX-tX1)*(aY-tY1));

  return tF;
}

//________________________________________________________________________________________________________________
double CoulombFitter::TrilinearInterpolate(TH3* a3dHisto, double aX, double aY, double aZ)
{
  TAxis* tXaxis = a3dHisto->GetXaxis();
  TAxis* tYaxis = a3dHisto->GetYaxis();
  TAxis* tZaxis = a3dHisto->GetZaxis();

  //--------------------------------
  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aX,aY,aZ) is within the limits, so I can interpolate
  if(ubx<=0 || uby<=0 || ubz<=0 || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in CoulombFitter::TrilinearInterpolate, cannot interpolate outside histogram domain" << endl;

    cout << "aX = " << aX << "\taY = " << aY << "\taZ = " << aZ << endl;
    cout << "ubx = " << ubx << "\tuby = " << uby << "\tubz = " << ubz << endl;
    cout << "obx = " << obx << "\toby = " << oby << "\tobz = " << obz << endl;
  }
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  double v[] = { a3dHisto->GetBinContent(ubx, uby, ubz), a3dHisto->GetBinContent(ubx, uby, obz),
                 a3dHisto->GetBinContent(ubx, oby, ubz), a3dHisto->GetBinContent(ubx, oby, obz),
                 a3dHisto->GetBinContent(obx, uby, ubz), a3dHisto->GetBinContent(obx, uby, obz),
                 a3dHisto->GetBinContent(obx, oby, ubz), a3dHisto->GetBinContent(obx, oby, obz) };

  double i1 = v[0]*(1.-zd) + v[1]*zd;
  double i2 = v[2]*(1.-zd) + v[3]*zd;
  double j1 = v[4]*(1.-zd) + v[5]*zd;
  double j2 = v[6]*(1.-zd) + v[7]*zd;

  double w1 = i1*(1.-yd) + i2*yd;
  double w2 = j1*(1.-yd) + j2*yd;

  double tResult = w1*(1.-xd) + w2*xd;

  return tResult;
}



//________________________________________________________________________________________________________________
double CoulombFitter::QuadrilinearInterpolate(THn* a4dHisto, double aT, double aX, double aY, double aZ)
{
  TAxis* tTaxis = a4dHisto->GetAxis(0);
  TAxis* tXaxis = a4dHisto->GetAxis(1);
  TAxis* tYaxis = a4dHisto->GetAxis(2);
  TAxis* tZaxis = a4dHisto->GetAxis(3);

  //--------------------------------

  int ubt = tTaxis->FindBin(aT);
  if( aT < tTaxis->GetBinCenter(ubt) ) ubt -= 1;
  int obt = ubt + 1;

  int ubx = tXaxis->FindBin(aX);
  if( aX < tXaxis->GetBinCenter(ubx) ) ubx -= 1;
  int obx = ubx + 1;

  int uby = tYaxis->FindBin(aY);
  if( aY < tYaxis->GetBinCenter(uby) ) uby -= 1;
  int oby = uby + 1;

  int ubz = tZaxis->FindBin(aZ);
  if( aZ < tZaxis->GetBinCenter(ubz) ) ubz -= 1;
  int obz = ubz + 1;

  //--------------------------------
  //make sure (aT,aX,aY,aZ) is within the limits, so I can interpolate
  if(ubt<=0 || ubx<=0 || uby<=0 || ubz<=0 || obt>tTaxis->GetNbins() || obx>tXaxis->GetNbins() || oby>tYaxis->GetNbins() || obz>tZaxis->GetNbins())
  {
    cout << "Error in CoulombFitter::QuadrilinearInterpolate, cannot interpolate outside histogram domain" << endl;
  }
  assert(ubt>0);
  assert(ubx>0);
  assert(uby>0);
  assert(ubz>0);
  assert(obt<=tTaxis->GetNbins());
  assert(obx<=tXaxis->GetNbins());
  assert(oby<=tYaxis->GetNbins());
  assert(obz<=tZaxis->GetNbins());
  //--------------------------------

  double tw = tTaxis->GetBinCenter(obt) - tTaxis->GetBinCenter(ubt);
  double xw = tXaxis->GetBinCenter(obx) - tXaxis->GetBinCenter(ubx);
  double yw = tYaxis->GetBinCenter(oby) - tYaxis->GetBinCenter(uby);
  double zw = tZaxis->GetBinCenter(obz) - tZaxis->GetBinCenter(ubz);

  double td = (aT - tTaxis->GetBinCenter(ubt))/tw;
  double xd = (aX - tXaxis->GetBinCenter(ubx))/xw;
  double yd = (aY - tYaxis->GetBinCenter(uby))/yw;
  double zd = (aZ - tZaxis->GetBinCenter(ubz))/zw;

  //--------------------------------
  //TODO these probably don't need to all be made at the same time
  // i.e., I can have a general int tBin, and put the appropriate numbers in there we needed
  int tBin0000[4] = {ubt,ubx,uby,ubz};
  int tBin1000[4] = {obt,ubx,uby,ubz};

  int tBin0100[4] = {ubt,obx,uby,ubz};
  int tBin1100[4] = {obt,obx,uby,ubz};

  int tBin0010[4] = {ubt,ubx,oby,ubz};
  int tBin1010[4] = {obt,ubx,oby,ubz};

  int tBin0110[4] = {ubt,obx,oby,ubz};
  int tBin1110[4] = {obt,obx,oby,ubz};

  int tBin0001[4] = {ubt,ubx,uby,obz};
  int tBin1001[4] = {obt,ubx,uby,obz};

  int tBin0101[4] = {ubt,obx,uby,obz};
  int tBin1101[4] = {obt,obx,uby,obz};

  int tBin0011[4] = {ubt,ubx,oby,obz};
  int tBin1011[4] = {obt,ubx,oby,obz};

  int tBin0111[4] = {ubt,obx,oby,obz};
  int tBin1111[4] = {obt,obx,oby,obz};

  //--------------------------------

  //Interpolate along t
  double tC000 = a4dHisto->GetBinContent(tBin0000)*(1.-td) + a4dHisto->GetBinContent(tBin1000)*td;
  double tC100 = a4dHisto->GetBinContent(tBin0100)*(1.-td) + a4dHisto->GetBinContent(tBin1100)*td;

  double tC010 = a4dHisto->GetBinContent(tBin0010)*(1.-td) + a4dHisto->GetBinContent(tBin1010)*td;
  double tC110 = a4dHisto->GetBinContent(tBin0110)*(1.-td) + a4dHisto->GetBinContent(tBin1110)*td;

  double tC001 = a4dHisto->GetBinContent(tBin0001)*(1.-td) + a4dHisto->GetBinContent(tBin1001)*td;
  double tC101 = a4dHisto->GetBinContent(tBin0101)*(1.-td) + a4dHisto->GetBinContent(tBin1101)*td;

  double tC011 = a4dHisto->GetBinContent(tBin0011)*(1.-td) + a4dHisto->GetBinContent(tBin1011)*td;
  double tC111 = a4dHisto->GetBinContent(tBin0111)*(1.-td) + a4dHisto->GetBinContent(tBin1111)*td;

  //--------------------------------

  //Interpolate along x
  double tC00 = tC000*(1.-xd) + tC100*xd;
  double tC10 = tC010*(1.-xd) + tC110*xd;
  double tC01 = tC001*(1.-xd) + tC101*xd;
  double tC11 = tC011*(1.-xd) + tC111*xd;

  //--------------------------------

  //Interpolate along y
  double tC0 = tC00*(1.-yd) + tC10*yd;
  double tC1 = tC01*(1.-yd) + tC11*yd;

  //--------------------------------

  //Interpolate along z
  double tC = tC0*(1.-zd) + tC1*zd;

  return tC;
}


//________________________________________________________________________________________________________________
double CoulombFitter::ScattLenInterpolate(vector<vector<vector<vector<double> > > > &aScatLen4dSubVec, double aReF0, double aImF0, double aD0, double aKStarMin, double aKStarMax, double aKStarVal)
{
  double tBinWidthK = fScattLenInfo.binWidthK;
  int tLocalBinNumberK = GetBinNumber(tBinWidthK,aKStarMin,aKStarMax,aKStarVal);
  double tBinLowCenterK = GetInterpLowBinCenter(kScattLen,kKaxis,aKStarVal);

  double tBinWidthReF0 = fScattLenInfo.binWidthReF0;
  double tBinLowCenterReF0 = GetInterpLowBinCenter(kScattLen,kReF0axis,aReF0);

  double tBinWidthImF0 = fScattLenInfo.binWidthImF0;
  double tBinLowCenterImF0 = GetInterpLowBinCenter(kScattLen,kImF0axis,aImF0);

  double tBinWidthD0 = fScattLenInfo.binWidthD0;
  double tBinLowCenterD0 = GetInterpLowBinCenter(kScattLen,kD0axis,aD0);

  //--------------------------------
  if(aKStarVal < tBinLowCenterK) tLocalBinNumberK--;
  int tLocalBinNumbersK[2] = {tLocalBinNumberK, tLocalBinNumberK+1};
  

  //--------------------------------
  double tDiffReF0 = (aReF0 - tBinLowCenterReF0)/tBinWidthReF0;
  double tDiffImF0 = (aImF0 - tBinLowCenterImF0)/tBinWidthImF0;
  double tDiffD0 = (aD0 - tBinLowCenterD0)/tBinWidthD0;
  double tDiffK = (aKStarVal - tBinLowCenterK)/tBinWidthK;

  //--------------------------------

  //Interpolate along ReF0
  double tC000 = aScatLen4dSubVec[0][0][0][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][0][tLocalBinNumbersK[0]]*tDiffReF0;
  double tC100 = aScatLen4dSubVec[0][1][0][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][0][tLocalBinNumbersK[0]]*tDiffReF0;

  double tC010 = aScatLen4dSubVec[0][0][1][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][1][tLocalBinNumbersK[0]]*tDiffReF0;
  double tC110 = aScatLen4dSubVec[0][1][1][tLocalBinNumbersK[0]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][1][tLocalBinNumbersK[0]]*tDiffReF0;

  double tC001 = aScatLen4dSubVec[0][0][0][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][0][tLocalBinNumbersK[1]]*tDiffReF0;
  double tC101 = aScatLen4dSubVec[0][1][0][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][0][tLocalBinNumbersK[1]]*tDiffReF0;

  double tC011 = aScatLen4dSubVec[0][0][1][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][0][1][tLocalBinNumbersK[1]]*tDiffReF0;
  double tC111 = aScatLen4dSubVec[0][1][1][tLocalBinNumbersK[1]]*(1.- tDiffReF0) + aScatLen4dSubVec[1][1][1][tLocalBinNumbersK[1]]*tDiffReF0;


  //--------------------------------

  //Interpolate along ImF0
  double tC00 = tC000*(1.-tDiffImF0) + tC100*tDiffImF0;
  double tC10 = tC010*(1.-tDiffImF0) + tC110*tDiffImF0;
  double tC01 = tC001*(1.-tDiffImF0) + tC101*tDiffImF0;
  double tC11 = tC011*(1.-tDiffImF0) + tC111*tDiffImF0;

  //--------------------------------

  //Interpolate along D0
  double tC0 = tC00*(1.-tDiffD0) + tC10*tDiffD0;
  double tC1 = tC01*(1.-tDiffD0) + tC11*tDiffD0;

  //--------------------------------

  //Interpolate along KStar
  double tC = tC0*(1.-tDiffK) + tC1*tDiffK;

  return tC;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetEta(double aKStar)
{
  if(fTurnOffCoulomb) return 0.;
  else return pow(((aKStar/hbarc)*fBohrRadius),-1);
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetGamowFactor(double aKStar)
{
  if(fTurnOffCoulomb) return 1.;
  else
  {
    double tEta = GetEta(aKStar);
    tEta *= TMath::TwoPi();  //eta always comes with 2Pi here
    double tGamow = tEta*pow((TMath::Exp(tEta)-1),-1);
    return tGamow;
  }
}


//________________________________________________________________________________________________________________
complex<double> CoulombFitter::GetExpTerm(double aKStar, double aRStar, double aTheta)
{
  complex<double> tImI (0.,1.);
  complex<double> tReturnValue = exp(-tImI*(aKStar/hbarc)*aRStar*cos(aTheta));
  return tReturnValue;
}

//________________________________________________________________________________________________________________
complex<double> CoulombFitter::BuildScatteringLength(double aKStar, double aReF0, double aImF0, double aD0)
{
  complex<double> tF0 (aReF0,aImF0);
  double tKStar = aKStar/hbarc;
  complex<double> tScattAmp;

  if(fTurnOffCoulomb)
  {
    complex<double> tLastTerm (0.,tKStar);
    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - tLastTerm,-1);
  }
  else
  {
    double tLednickyHFunction = LinearInterpolate(fLednickyHFunctionHist,aKStar);
    double tImagChi = GetGamowFactor(aKStar)/(2.*GetEta(aKStar));
    complex<double> tLednickyChi (tLednickyHFunction,tImagChi);

    tScattAmp = pow((1./tF0) + 0.5*aD0*tKStar*tKStar - 2.*tLednickyChi/fBohrRadius,-1);
  }

  return tScattAmp;
}



//________________________________________________________________________________________________________________
double CoulombFitter::InterpolateWfSquared(double aKStarMag, double aRStarMag, double aTheta, double aReF0, double aImF0, double aD0)
{

  bool tDebug = true; //debug means use personal interpolation methods, instead of standard root ones

  //TODO put check to make sure file is open, not sure if assert(fInterpHistFile->IsOpen works);
    if(!fCreateInterpVectors && fInterpHistsLoaded) assert(fInterpHistFile->IsOpen());

  double tGamow = GetGamowFactor(aKStarMag);
  complex<double> tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  if(!fTurnOffCoulomb)
  {
    double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag;
    //-------------------------------------
    if(tDebug)
    {
      tHyperGeo1F1Real = TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);
    }

    else
    {
      tHyperGeo1F1Real = fHyperGeo1F1RealHist->Interpolate(aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = fHyperGeo1F1ImagHist->Interpolate(aKStarMag,aRStarMag,aTheta);

      tGTildeReal = fGTildeRealHist->Interpolate(aKStarMag,aRStarMag);
      tGTildeImag = fGTildeImagHist->Interpolate(aKStarMag,aRStarMag);
    }

    tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
    tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);
    //-------------------------------------
    if(fUseScattLenHists)
    {
      double tScattLenReal, tScattLenImag;
      if(aKStarMag >= 0. && aKStarMag < 0.2)
      {
//cout << "In 0-0.2" << endl;
//cout << "\taKStarMag = " << aKStarMag << "\taReF0 = " << aReF0 << "\taImF0 = " << aImF0 << "\taD0 = " << aD0 << endl;
        tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist1,aKStarMag,aReF0,aImF0,aD0);
        tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist1,aKStarMag,aReF0,aImF0,aD0);
      }
      else if(aKStarMag >= 0.2 && aKStarMag < 0.4)
      {
//cout << "In 0.2-0.4" << endl;
//cout << "\taKStarMag = " << aKStarMag << "\taReF0 = " << aReF0 << "\taImF0 = " << aImF0 << "\taD0 = " << aD0 << endl;
        tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist2,aKStarMag,aReF0,aImF0,aD0);
        tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist2,aKStarMag,aReF0,aImF0,aD0);
      }
      else
      {
        cout << "In CoulombFitter::InterpolateWfSquared, aKStarMag is outside of limits!!!!!!!!!!" << endl;
        assert(0);
      }

      tScattLenComplexConj = complex<double>(tScattLenReal,-tScattLenImag);
    }

    else
    {
      complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
      tScattLenComplexConj = std::conj(tScattLenComplex);
    }
  }

  else
  {
    tHyperGeo1F1Complex = complex<double> (1.,0.);
    complex<double> tImI (0.,1.);
    tGTildeComplexConj = exp(-tImI*(aKStarMag/hbarc)*aRStarMag);

    complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
    tScattLenComplexConj = std::conj(tScattLenComplex);
  }

  //-------------------------------------------

  complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

  if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitter::InterpolateWfSquared !!!!!" << endl;
  assert(imag(tResultComplex) < std::numeric_limits< double >::min());

  return real(tResultComplex);

}

//________________________________________________________________________________________________________________
vector<double> CoulombFitter::InterpolateWfSquaredSerialv2(vector<vector<double> > &aPairs, double aKStarMagMin, double aKStarMagMax, double aReF0, double aImF0, double aD0)
{
  vector<double> tReturnVector;
    tReturnVector.clear();

  vector<vector<vector<vector<double> > > > tRelevantScattLengthReal;
  vector<vector<vector<vector<double> > > > tRelevantScattLengthImag;

  if(fUseScattLenHists && fCreateInterpVectors)
  {
    vector<int> tRelevantKBins = GetRelevantKStarBinNumbers(aKStarMagMin,aKStarMagMax);

    int tLowBinReF0 = GetInterpLowBin(kScattLen,kReF0axis,aReF0);
    int tLowBinImF0 = GetInterpLowBin(kScattLen,kImF0axis,aImF0);
    int tLowBinD0 = GetInterpLowBin(kScattLen,kD0axis,aD0);

    int tNbinsReF0 = 2;
    int tNbinsImF0 = 2;
    int tNbinsD0 = 2;
    int tNbinsK = tRelevantKBins.size();

    vector<int> tRelevantReF0Bins(2);
      tRelevantReF0Bins[0] = tLowBinReF0;
      tRelevantReF0Bins[1] = tLowBinReF0+1;

    vector<int> tRelevantImF0Bins(2);
      tRelevantImF0Bins[0] = tLowBinImF0;
      tRelevantImF0Bins[1] = tLowBinImF0+1;

    vector<int> tRelevantD0Bins(2);
      tRelevantD0Bins[0] = tLowBinD0;
      tRelevantD0Bins[1] = tLowBinD0+1;

    tRelevantScattLengthReal.resize(tNbinsReF0, vector<vector<vector<double> > > (tNbinsImF0, vector<vector<double> > (tNbinsD0, vector<double> (tNbinsK,0))));
    tRelevantScattLengthImag.resize(tNbinsReF0, vector<vector<vector<double> > > (tNbinsImF0, vector<vector<double> > (tNbinsD0, vector<double> (tNbinsK,0))));

    for(int iReF0=0; iReF0<2; iReF0++)
    {
      for(int iImF0=0; iImF0<2; iImF0++)
      {
        for(int iD0=0; iD0<2; iD0++)
        {
          for(int iK=0; iK<(int)tRelevantKBins.size(); iK++)
          {
            tRelevantScattLengthReal[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthReal[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][tRelevantKBins[iK]];
            tRelevantScattLengthImag[iReF0][iImF0][iD0][iK] = fCoulombScatteringLengthImag[tRelevantReF0Bins[iReF0]][tRelevantImF0Bins[iImF0]][tRelevantD0Bins[iD0]][tRelevantKBins[iK]];
          }
        }
      }
    }

  }

  //-----------------------------------------------

  double tGamow;
  complex<double> tExpTermComplex;
  
  complex<double> tScattLenComplexConj;

  complex<double> tHyperGeo1F1Complex;
  complex<double> tGTildeComplexConj;

  double aKStarMag, aRStarMag, aTheta;

  for(int i=0; i<(int)aPairs.size(); i++)
  {
    aKStarMag = aPairs[i][0];
    aRStarMag = aPairs[i][1];
    aTheta = aPairs[i][2];

    tGamow = GetGamowFactor(aKStarMag);
    tExpTermComplex = GetExpTerm(aKStarMag,aRStarMag,aTheta);

    if(!fTurnOffCoulomb)
    {
      double tHyperGeo1F1Real, tHyperGeo1F1Imag, tGTildeReal, tGTildeImag, tScattLenReal, tScattLenImag;

      tHyperGeo1F1Real = TrilinearInterpolate(fHyperGeo1F1RealHist,aKStarMag,aRStarMag,aTheta);
      tHyperGeo1F1Imag = TrilinearInterpolate(fHyperGeo1F1ImagHist,aKStarMag,aRStarMag,aTheta);

      tGTildeReal = BilinearInterpolate(fGTildeRealHist,aKStarMag,aRStarMag);
      tGTildeImag = BilinearInterpolate(fGTildeImagHist,aKStarMag,aRStarMag);

      tHyperGeo1F1Complex = complex<double> (tHyperGeo1F1Real,tHyperGeo1F1Imag);
      tGTildeComplexConj = complex<double> (tGTildeReal,-tGTildeImag);

      if(fUseScattLenHists && fCreateInterpVectors)
      {
        tScattLenReal = ScattLenInterpolate(tRelevantScattLengthReal,aReF0,aImF0,aD0,aKStarMagMin,aKStarMagMax,aKStarMag);
        tScattLenImag = ScattLenInterpolate(tRelevantScattLengthImag,aReF0,aImF0,aD0,aKStarMagMin,aKStarMagMax,aKStarMag);

        tScattLenComplexConj = complex<double>(tScattLenReal,-tScattLenImag);
      }
      else if(fUseScattLenHists)
      {
        if(aKStarMag >= 0. && aKStarMag < 0.2)
        {
          tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist1,aKStarMag,aReF0,aImF0,aD0);
          tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist1,aKStarMag,aReF0,aImF0,aD0);
        }
        else if(aKStarMag >= 0.2 && aKStarMag < 0.4)
        {
          tScattLenReal = QuadrilinearInterpolate(fCoulombScatteringLengthRealHist2,aKStarMag,aReF0,aImF0,aD0);
          tScattLenImag = QuadrilinearInterpolate(fCoulombScatteringLengthImagHist2,aKStarMag,aReF0,aImF0,aD0);
        }
        else
        {
          cout << "In CoulombFitter::InterpolateWfSquaredSerialv2, aKStarMag is outside of limits!!!!!!!!!!" << endl;
          assert(0);
        }
        tScattLenComplexConj = complex<double>(tScattLenReal,-tScattLenImag);
      }
      else
      {
        complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
        tScattLenComplexConj = std::conj(tScattLenComplex);
      }

    }
    else
    {
      tHyperGeo1F1Complex = complex<double> (1.,0.);
      complex<double> tImI (0.,1.);
      tGTildeComplexConj = exp(-tImI*(aKStarMag/hbarc)*aRStarMag);

      complex<double> tScattLenComplex = BuildScatteringLength(aKStarMag,aReF0,aImF0,aD0);
      tScattLenComplexConj = std::conj(tScattLenComplex);
    }

    complex<double> tResultComplex = tGamow*( norm(tHyperGeo1F1Complex) + norm(tScattLenComplexConj)*norm(tGTildeComplexConj)/(aRStarMag*aRStarMag) + 2.*real(tExpTermComplex*tHyperGeo1F1Complex*tScattLenComplexConj*tGTildeComplexConj/aRStarMag) );

    if(imag(tResultComplex) > std::numeric_limits< double >::min()) cout << "\t\t\t !!!!!!!!! Imaginary value in CoulombFitter::InterpolateWfSquaredSerialv2 !!!!!" << endl;
    assert(imag(tResultComplex) < std::numeric_limits< double >::min());

    tReturnVector.push_back(real(tResultComplex));
  }

  return tReturnVector;
}


//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpKStar(double aKStar)
{
  if(aKStar < fMinInterpKStar1) return false;
  if(aKStar > fMaxInterpKStar2) return false;
  if(fUseScattLenHists && aKStar > fMaxInterpKStar1 && aKStar < fMinInterpKStar2) return false;  //awkward non-overlapping scenarios  TODO fix this in InterpolationHistograms.cxx
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpRStar(double aRStar)
{
  if(aRStar < fMinInterpRStar || aRStar > fMaxInterpRStar) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpTheta(double aTheta)
{
  if(aTheta < fMinInterpTheta || aTheta > fMaxInterpTheta) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpReF0(double aReF0)
{
  if(aReF0 < fMinInterpReF0 || aReF0 > fMaxInterpReF0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpImF0(double aImF0)
{
  if(aImF0 < fMinInterpImF0 || aImF0 > fMaxInterpImF0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpD0(double aD0)
{
  if(aD0 < fMinInterpD0 || aD0 > fMaxInterpD0) return false;
  return true;
}

//________________________________________________________________________________________________________________
bool CoulombFitter::CanInterpAll(double aKStar, double aRStar, double aTheta, double aReF0, double aImF0, double aD0)
{
  if(fUseScattLenHists)
  {
    if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta) && CanInterpReF0(aReF0) && CanInterpImF0(aImF0) && CanInterpD0(aD0)) return true;
  }
  else
  {
    if(CanInterpKStar(aKStar) && CanInterpRStar(aRStar) && CanInterpTheta(aTheta)) return true;
  }

  return false;
}

//________________________________________________________________________________________________________________
void CoulombFitter::SetRandomKStar3Vec(TVector3* aKStar3Vec, double aKStarMagMin, double aKStarMagMax)
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
double CoulombFitter::GetFitCfContent(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
/*
cout << "In CoulombFitter::GetFitCfContent" << endl;
cout << "\t aKStarMagMin = " << aKStarMagMin << endl;
cout << "\t aKStarMagMax = " << aKStarMagMax << endl;
*/
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;
/*
cout << "In GetFitCfContent....." << endl;
cout << "\t tBin = " << tBin << endl;
cout << "\t aKStarMagMin = " << aKStarMagMin << "\t aKStarMagMax = " << aKStarMagMax << endl;
*/


  //KStarMag = fPairKStar4dVec[aAnalysisNumber][tBin][i][0]
  //KStarOut = fPairKStar4dVec[aAnalysisNumber][tBin][i][1]
  //KStarSide = fPairKStar4dVec[aAnalysisNumber][tBin][i][2]
  //KStarLong = fPairKStar4dVec[aAnalysisNumber][tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;
//cout << "fPairKStar4dVec[aAnalysisNumber][tBin].size() = " << fPairKStar4dVec[aAnalysisNumber][tBin].size() << endl << endl;

  int tMaxKStarCalls;
/*
  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < 100) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();
  else tMaxKStarCalls = 100;
*/
  //definitely oversampling by commenting out the above
  //Currently, lowest bin only have 6 entries!
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

//  for(int i=0; i<fPairKStar4dVec[aAnalysisNumber][tBin].size(); i++)
  {
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things

    if(!fUseRandomKStarVectors)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
    }
    else SetRandomKStar3Vec(tKStar3Vec,aKStarMagMin,aKStarMagMax);

    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric

    tTheta = tKStar3Vec->Angle(*tSource3Vec);
    tKStarMag = tKStar3Vec->Mag();

    //TODO: take out this restriction
//    while (tSource3Vec->Mag() > 20.) tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator));

    tRStarMag = tSource3Vec->Mag();


//TODO make 1 and 2 histograms overlap by a bin or two!
//TODO
//TODO
//TODO!!!!!!!!!!!!! Must have separate InterpolateWfSquaredXiKchP and InterpolateWfSquaredXiKchM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Note:  Running CanInterpAll is faster than checking CanInterpScatParams outside of for loop and comparing to CanInterpKRTh.....strange but true
    if(fTurnOffCoulomb || CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4])) 
    {
      tWaveFunctionSq = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tNInterpolate++;
    }
    else
    {
      tWaveFunction = fWaveFunction->GetWaveFunction(tKStar3Vec,tSource3Vec,par[2],par[3],par[4]);
      tWaveFunctionSq = norm(tWaveFunction);
      tNMathematica++;
    }
/*
    complex<double> tWaveFunction = fWaveFunction->GetWaveFunction(tKStar3Vec,tSource3Vec,par[2],par[3],par[4]);
    tWaveFunctionSq = norm(tWaveFunction);
*/
    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }

  delete tKStar3Vec;
  delete tSource3Vec;

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentwStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  UpdatePairRadiusParameters(par[1]);

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0]
  //RStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1]
  //Theta    = fPairSample4dVec[aAnalysisNumber][tBin][i][2]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample4dVec[aAnalysisNumber][tBin].size();
//cout << "In GetFitCfContentwStaticPairs: " << endl;
//cout << "\taKStarMagMin = " << aKStarMagMin << " and tBin = " << tBin << endl;
//cout << "\ttMaxKStarCalls = " << tMaxKStarCalls << endl;

  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSq;
  complex<double> tWaveFunction;

  int tNInterpolate = 0;
  int tNMathematica = 0;

  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0];
    tRStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1];
    tTheta = fPairSample4dVec[aAnalysisNumber][tBin][i][2];

    if(fTurnOffCoulomb || CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4])) 
    {
      tWaveFunctionSq = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
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

//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentComplete(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  assert(fIncludeSingletAndTriplet == true);

  omp_set_num_threads(6);
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairKStar4dVec[aAnalysisNumber][tBin][i][0]
  //KStarOut = fPairKStar4dVec[aAnalysisNumber][tBin][i][1]
  //KStarSide = fPairKStar4dVec[aAnalysisNumber][tBin][i][2]
  //KStarLong = fPairKStar4dVec[aAnalysisNumber][tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;
//cout << "fPairKStar4dVec[aAnalysisNumber][tBin].size() = " << fPairKStar4dVec[aAnalysisNumber][tBin].size() << endl << endl;

  int tMaxKStarCalls;

//  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < 100) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();
//  else tMaxKStarCalls = 100;

  //definitely oversampling by commenting out the above
  //Currently, lowest bin only have 6 entries!
  tMaxKStarCalls = 16384;
//  if(fPairKStar4dVec[aAnalysisNumber][tBin].size() < tMaxKStarCalls) tMaxKStarCalls = fPairKStar4dVec[aAnalysisNumber][tBin].size();

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
  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq;
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;

  vector<vector<double> > tMathematicaPairs;
  vector<double> tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

//ChronoTimer tIntTimer;
//tIntTimer.Start();

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tI, tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq) firstprivate(tKStar3Vec, tSource3Vec, tTempPair)
  for(int i=0; i<tMaxKStarCalls; i++)

//  for(int i=0; i<fPairKStar4dVec[aAnalysisNumber][tBin].size(); i++)
  {
//TODO Check randomization
//TODO Make sure I am grabbing from correct tBin.  Must work even when I rebin things
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


//TODO make 1 and 2 histograms overlap by a bin or two!
//TODO
//TODO
//TODO!!!!!!!!!!!!! Must have separate InterpolateWfSquaredXiKchP and InterpolateWfSquaredXiKchM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Note:  Running CanInterpAll is faster than checking CanInterpScatParams outside of for loop and comparing to CanInterpKRTh.....strange but true
    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
    if(fTurnOffCoulomb || tCanInterp)
    {
      tWaveFunctionSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tWaveFunctionSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
      tReturnCfContent += tWaveFunctionSq;
      tCounter ++;
    }

    #pragma omp critical
    if(!tCanInterp)
    {
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;
      tMathematicaPairs.push_back(tTempPair);
    }
  }

//tIntTimer.Stop();
//cout << "Interpolation in GetFitCfContentComplete ";
//tIntTimer.PrintInterval();

  tNInterpolate = tCounter;
  tNMathematica = tMathematicaPairs.size();

//ChronoTimer tMathTimer;
//tMathTimer.Start();

  for(int i=0; i<(int)tMathematicaPairs.size(); i++)
  {
    tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
    tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

    tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[5],par[6],par[7]);
    tWaveFunctionSqTriplet = norm(tWaveFunctionTriplet);

    tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }
//tMathTimer.Stop();
//cout << "Mathematica calls in GetFitCfContentComplete ";
//tMathTimer.PrintInterval();

  delete tKStar3Vec;
  delete tSource3Vec;

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[8]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}

//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentCompletewStaticPairs(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  assert(fIncludeSingletAndTriplet == true);

  omp_set_num_threads(6);
  //par[0] = kLambda
  //par[1] = kRadius
  //par[2] = kRef0
  //par[3] = kImf0
  //par[4] = kd0
  //par[5] = kRef02
  //par[6] = kImf02
  //par[7] = kd02
  //par[8] = kNorm

  UpdatePairRadiusParameters(par[1]);

  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0]
  //RStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1]
  //Theta    = fPairSample4dVec[aAnalysisNumber][tBin][i][2]


  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tMaxKStarCalls;
  tMaxKStarCalls = fPairSample4dVec[aAnalysisNumber][tBin].size();
//cout << "In GetFitCfContentCompletewStaticPairs: " << endl;
//cout << "\taKStarMagMin = " << aKStarMagMin << " and tBin = " << tBin << endl;
//cout << "\ttMaxKStarCalls = " << tMaxKStarCalls << endl;


  bool tCanInterp;
  double tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq;
  complex<double> tWaveFunctionSinglet, tWaveFunctionTriplet;

  vector<vector<double> > tMathematicaPairs;
  vector<double> tTempPair(3);

  int tNInterpolate = 0;
  int tNMathematica = 0;

//ChronoTimer tIntTimer;
//tIntTimer.Start();

  #pragma omp parallel for reduction(+: tCounter) reduction(+: tReturnCfContent) private(tCanInterp, tTheta, tKStarMag, tRStarMag, tWaveFunctionSqSinglet, tWaveFunctionSqTriplet, tWaveFunctionSq) firstprivate(tTempPair)
  for(int i=0; i<tMaxKStarCalls; i++)
  {
    tKStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][0];
    tRStarMag = fPairSample4dVec[aAnalysisNumber][tBin][i][1];
    tTheta = fPairSample4dVec[aAnalysisNumber][tBin][i][2];

    tCanInterp = CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
    if(fTurnOffCoulomb || tCanInterp)
    {
      tWaveFunctionSqSinglet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]);
      tWaveFunctionSqTriplet = InterpolateWfSquared(tKStarMag,tRStarMag,tTheta,par[5],par[6],par[7]);

      tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
      tReturnCfContent += tWaveFunctionSq;
      tCounter ++;
    }

    #pragma omp critical
    if(!tCanInterp)
    {
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;
      tMathematicaPairs.push_back(tTempPair);
    }
  }

//tIntTimer.Stop();
//cout << "Interpolation in GetFitCfContentComplete ";
//tIntTimer.PrintInterval();

  tNInterpolate = tCounter;
  tNMathematica = tMathematicaPairs.size();

//ChronoTimer tMathTimer;
//tMathTimer.Start();

  for(int i=0; i<(int)tMathematicaPairs.size(); i++)
  {
    tWaveFunctionSinglet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[2],par[3],par[4]);
    tWaveFunctionSqSinglet = norm(tWaveFunctionSinglet);

    tWaveFunctionTriplet = fWaveFunction->GetWaveFunction(tMathematicaPairs[i][0],tMathematicaPairs[i][1],tMathematicaPairs[i][2],par[5],par[6],par[7]);
    tWaveFunctionSqTriplet = norm(tWaveFunctionTriplet);

    tWaveFunctionSq = 0.25*tWaveFunctionSqSinglet + 0.75*tWaveFunctionSqTriplet;
    tReturnCfContent += tWaveFunctionSq;
    tCounter ++;
  }
//tMathTimer.Stop();
//cout << "Mathematica calls in GetFitCfContentComplete ";
//tMathTimer.PrintInterval();

  tReturnCfContent /= tCounter;

//  tReturnCfContent = par[8]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));


  if(tNMathematica > 0.2*tMaxKStarCalls)
  {
    cout << "\t\t\tWARNING:: tNMathematica > 1/5 of pairs!!!!!!!!!!!!!!!!!!" << endl;
  }

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetFitCfContentSerialv2(double aKStarMagMin, double aKStarMagMax, double *par, int aAnalysisNumber)
{
  // par[0] = Lambda 
  // par[1] = Radius
  // par[2] = Ref0
  // par[3] = Imf0
  // par[4] = d0
  // par[5] = Norm

  //should probably do x[0] /= hbarc, but let me test first


  //TODO make this general
  //This is messing up around aKStarMagMin = 0.29, or bin 57/58
  double tBinSize = 0.01;
  int tBin = aKStarMagMin/tBinSize;

  //KStarMag = fPairKStar4dVec[aAnalysisNumber][tBin][i][0]
  //KStarOut = fPairKStar4dVec[aAnalysisNumber][tBin][i][1]
  //KStarSide = fPairKStar4dVec[aAnalysisNumber][tBin][i][2]
  //KStarLong = fPairKStar4dVec[aAnalysisNumber][tBin][i][3]

  int tCounter = 0;
  double tReturnCfContent = 0.;

  int tCounterGPU = 0;
  double tCfContentGPU = 0.;

  int tCounterCPU = 0;
  double tCfContentCPU = 0.;

//  int tMaxKStarCalls = 10000;
  int tMaxKStarCalls = 16384; //==2^14

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

  vector<vector<double> > tPairsGPU;
  vector<vector<double> > tPairsCPU;
  vector<double> tTempPair(3);

  int tNGood=0;
  while(tNGood<tMaxKStarCalls)
//  for(int i=0; i<tMaxKStarCalls; i++)
  {
    if(!fUseRandomKStarVectors)
    {
      tI = tRandomKStarElement(generator);
      tKStar3Vec->SetXYZ(fPairKStar4dVec[aAnalysisNumber][tBin][tI][1],fPairKStar4dVec[aAnalysisNumber][tBin][tI][2],fPairKStar4dVec[aAnalysisNumber][tBin][tI][3]);
    }
    else SetRandomKStar3Vec(tKStar3Vec,aKStarMagMin,aKStarMagMax);
    tKStarMag = tKStar3Vec->Mag();

    tSource3Vec->SetXYZ(tROutSource(generator),tRSideSource(generator),tRLongSource(generator)); //TODO: for now, spherically symmetric
      tRStarMag = tSource3Vec->Mag();
    tTheta = tKStar3Vec->Angle(*tSource3Vec);

//    tTempPair.clear();  //TODO NEVER CLEAR A VECTOR IF YOU WANT IT TO MAINTAIN ITS SIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      tTempPair[0] = tKStarMag;
      tTempPair[1] = tRStarMag;
      tTempPair[2] = tTheta;


    if(CanInterpAll(tKStarMag,tRStarMag,tTheta,par[2],par[3],par[4]))
    {
      tPairsGPU.push_back(tTempPair);
      tNGood++;
    }
    else tPairsCPU.push_back(tTempPair);
  }
  delete tKStar3Vec;
  delete tSource3Vec;

  //---------Do serial calculations------------
  for(int i=0; i<(int)tPairsCPU.size(); i++)
  {
    tWaveFunction = fWaveFunction->GetWaveFunction(tPairsCPU[i][0],tPairsCPU[i][1],tPairsCPU[i][2],par[2],par[3],par[4]);
    tWaveFunctionSq = norm(tWaveFunction);

    tCfContentCPU += tWaveFunctionSq;
    tCounterCPU++;
  }

  //--------Do parallel calculations!----------
  vector<double> tGPUResults = InterpolateWfSquaredSerialv2(tPairsGPU,aKStarMagMin,aKStarMagMax,par[2],par[3],par[4]);
  for(int i=0; i<(int)tGPUResults.size(); i++)
  {
    tCfContentGPU += tGPUResults[i];
    tCounterGPU++;
  }

  tReturnCfContent = (tCfContentCPU + tCfContentGPU)/(tCounterCPU+tCounterGPU);
//  tReturnCfContent = par[5]*(par[0]*tReturnCfContent + (1.-par[0]));  //C = Norm*(Lam*C_gen + (1-Lam));
  tReturnCfContent = (par[0]*tReturnCfContent + (1.-par[0]));  //C = (Lam*C_gen + (1-Lam));

  return tReturnCfContent;
}


//________________________________________________________________________________________________________________
bool CoulombFitter::AreParamsSame(double *aCurrent, double *aNew, int aNEntries)
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
void CoulombFitter::CalculateChi2PML(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  if(fIncludeSingletAndTriplet)
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0s  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0s  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0s    = " << par[4] << endl;

    cout << "\t\t\tpar[5] = ReF0t  = " << par[5] << endl;
    cout << "\t\t\tpar[6] = ImF0t  = " << par[6] << endl;
    cout << "\t\t\tpar[7] = D0t    = " << par[7] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[8] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[9] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[10] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[11] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[12] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[13] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[14] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[15] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[16] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[17] << endl;
  }
  else
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
    cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl;
  }

  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

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

    if(!fAllOfSameCoulombType)
    {
      //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
      else fBohrRadius = 1000000000;
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
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
      int tRef02MinuitParamNumber, tImf02MinuitParamNumber, td02MinuitParamNumber;
      double *tPar;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      if(!fIncludeSingletAndTriplet)
      {
        assert(tNFitParams == 6);
        tPar = new double[tNFitParams];

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tNormMinuitParamNumber];
      }

      else
      {
        assert(tNFitParams == 9);
        tPar = new double[tNFitParams];

        tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
        tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
        td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tRef02MinuitParamNumber];
        tPar[6] = par[tImf02MinuitParamNumber];
        tPar[7] = par[td02MinuitParamNumber];
        tPar[8] = par[tNormMinuitParamNumber];
      }

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        double tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        double tNumContent = tNum->GetBinContent(ix);
        double tDenContent = tDen->GetBinContent(ix);

//        if(!tAreParamsSame) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
//        tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tCfContentUnNorm[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }

        double tCfContent;
        if(fIncludeSingletAndTriplet) tCfContent = tPar[8]*tCfContentUnNorm[ix-1];
        else tCfContent = tPar[5]*tCfContentUnNorm[ix-1];

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


//________________________________________________________________________________________________________________
vector<double> CoulombFitter::ApplyMomResCorrection(vector<double> &aCf, vector<double> &aKStarBinCenters, TH2* aMomResMatrix)
{
  //TODO probably rebin aMomResMatrix to match bin size of aCf

  unsigned int tKStarRecBin, tKStarTrueBin;
  double tKStarRec, tKStarTrue;
  assert(aCf.size() == aKStarBinCenters.size());

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
void CoulombFitter::CalculateChi2PMLwMomResCorrection(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  assert(fApplyMomResCorrection);

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  if(fIncludeSingletAndTriplet)
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0s  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0s  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0s    = " << par[4] << endl;

    cout << "\t\t\tpar[5] = ReF0t  = " << par[5] << endl;
    cout << "\t\t\tpar[6] = ImF0t  = " << par[6] << endl;
    cout << "\t\t\tpar[7] = D0t    = " << par[7] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[8] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[9] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[10] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[11] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[12] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[13] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[14] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[15] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[16] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[17] << endl;
  }
  else
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
    cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl;
  }

  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

  double *tCurrentFitPar = new double[tNFitParPerAnalysis];
  for(int i=0; i<tNFitParPerAnalysis; i++) tCurrentFitPar[i] = 0.;

  int tNbinsXToFitGlobal = fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->FindBin(fMaxFitKStar);
  if(fFitSharedAnalyses->GetFitPairAnalysis(0)->GetFitPartialAnalysis(0)->GetKStarCfLite()->Num()->GetBinLowEdge(tNbinsXToFitGlobal) == fMaxFitKStar) tNbinsXToFitGlobal--;

  vector<double> tCfContentUnNorm(tNbinsXToFitGlobal,0.);
  vector<double> tCfContent(tNbinsXToFitGlobal,0.);
  vector<double> tNumContent(tNbinsXToFitGlobal,0.);
  vector<double> tDenContent(tNbinsXToFitGlobal,0.);
  vector<double> tKStarBinCenters(tNbinsXToFitGlobal,0.);

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    TH2* tMomResMatrix = tFitPairAnalysis->GetModelKStarTrueVsRecMixed();
    assert(tMomResMatrix);

    if(!fAllOfSameCoulombType)
    {
      //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
      else fBohrRadius = 1000000000;
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
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
      int tRef02MinuitParamNumber, tImf02MinuitParamNumber, td02MinuitParamNumber;
      double *tPar;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      if(!fIncludeSingletAndTriplet)
      {
        assert(tNFitParams == 6);
        tPar = new double[tNFitParams];

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tNormMinuitParamNumber];
      }

      else
      {
        assert(tNFitParams == 9);
        tPar = new double[tNFitParams];

        tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
        tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
        td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tRef02MinuitParamNumber];
        tPar[6] = par[tImf02MinuitParamNumber];
        tPar[7] = par[td02MinuitParamNumber];
        tPar[8] = par[tNormMinuitParamNumber];
      }

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2PML, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxisNum->GetBinLowEdge(ix);
        double tKStarMax = tXaxisNum->GetBinLowEdge(ix+1);

        tKStarBinCenters[ix-1] = tXaxisNum->GetBinCenter(ix);

        tNumContent[ix-1] = tNum->GetBinContent(ix);
        tDenContent[ix-1] = tDen->GetBinContent(ix);

//        if(!tAreParamsSame) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
//        tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tCfContentUnNorm[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }

        if(fIncludeSingletAndTriplet) tCfContent[ix-1] = tPar[8]*tCfContentUnNorm[ix-1];
        else tCfContent[ix-1] = tPar[5]*tCfContentUnNorm[ix-1];
      }

      vector<double> tCfContentwMomResCorrection = ApplyMomResCorrection(tCfContent, tKStarBinCenters, tMomResMatrix);
      for(int ix=0; ix < tNbinsXToFit; ix++)
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



//________________________________________________________________________________________________________________
void CoulombFitter::CalculateChi2(int &npar, double &chi2, double *par)
{
ChronoTimer tTotalTimer;
tTotalTimer.Start();

  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  if(fIncludeSingletAndTriplet)
  {
    cout << "\t\tParameter update: " << endl;
    cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
    cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

    cout << "\t\t\tpar[2] = ReF0s  = " << par[2] << endl;
    cout << "\t\t\tpar[3] = ImF0s  = " << par[3] << endl;
    cout << "\t\t\tpar[4] = D0s    = " << par[4] << endl;

    cout << "\t\t\tpar[5] = ReF0t  = " << par[5] << endl;
    cout << "\t\t\tpar[6] = ImF0t  = " << par[6] << endl;
    cout << "\t\t\tpar[7] = D0t    = " << par[7] << endl;

    cout << "\t\t\tpar[8] = Norm1  = " << par[8] << endl;
    cout << "\t\t\tpar[9] = Norm2  = " << par[9] << endl;
    cout << "\t\t\tpar[10] = Norm3 = " << par[10] << endl;
    cout << "\t\t\tpar[11] = Norm4 = " << par[11] << endl;
    cout << "\t\t\tpar[12] = Norm5 = " << par[12] << endl;
    cout << "\t\t\tpar[13] = Norm6 = " << par[13] << endl;
    cout << "\t\t\tpar[14] = Norm7 = " << par[14] << endl;
    cout << "\t\t\tpar[15] = Norm8 = " << par[15] << endl;
    cout << "\t\t\tpar[16] = Norm9 = " << par[16] << endl;
    cout << "\t\t\tpar[17] = Norm10= " << par[17] << endl;
  }
  else
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
    cout << "\t\t\tpar[17] = Norm10= " << par[14] << endl;
  }

  //--------------------------------------------------------------

  int tNFitParPerAnalysis;
  if(fIncludeSingletAndTriplet) tNFitParPerAnalysis = 8;
  else tNFitParPerAnalysis = 5;

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

    if(!fAllOfSameCoulombType)
    {
      //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
      else fBohrRadius = 1000000000;
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();
      TH1* tCfToFit = tKStarCfLite->Cf();

      TAxis* tXaxis = tCfToFit->GetXaxis();
      int tNbinsX = tCfToFit->GetNbinsX();

      int tNbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
      if(tCfToFit->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;


      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXToFitGlobal);
      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams = tNFitParPerAnalysis+1);

      int tLambdaMinuitParamNumber, tRadiusMinuitParamNumber, tRef0MinuitParamNumber, tImf0MinuitParamNumber, td0MinuitParamNumber, tNormMinuitParamNumber;
      int tRef02MinuitParamNumber, tImf02MinuitParamNumber, td02MinuitParamNumber;
      double *tPar;

      tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();
      tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      if(!fIncludeSingletAndTriplet)
      {
        assert(tNFitParams == 6);
        tPar = new double[tNFitParams];

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tNormMinuitParamNumber];
      }

      else
      {
        assert(tNFitParams == 9);
        tPar = new double[tNFitParams];

        tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
        tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
        td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();

        tPar[0] = par[tLambdaMinuitParamNumber];
        tPar[1] = par[tRadiusMinuitParamNumber];
        tPar[2] = par[tRef0MinuitParamNumber];
        tPar[3] = par[tImf0MinuitParamNumber];
        tPar[4] = par[td0MinuitParamNumber];
        tPar[5] = par[tRef02MinuitParamNumber];
        tPar[6] = par[tImf02MinuitParamNumber];
        tPar[7] = par[td02MinuitParamNumber];
        tPar[8] = par[tNormMinuitParamNumber];
      }

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      bool tAreParamsSame = AreParamsSame(tCurrentFitPar,tPar,tNFitParPerAnalysis);
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxis->GetBinLowEdge(ix);
        double tKStarMax = tXaxis->GetBinLowEdge(ix+1);

//        if(!tAreParamsSame) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
//        tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        if(!tAreParamsSame)
        {
          if(fUseStaticPairs && fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fIncludeSingletAndTriplet) tCfContentUnNorm[ix-1] = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
          else if(fUseStaticPairs) tCfContentUnNorm[ix-1] = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,iAnaly);
          else tCfContentUnNorm[ix-1] = GetFitCfContent(tKStarMin,tKStarMax,tPar,iAnaly);
        }

        double tCfContent;
        if(fIncludeSingletAndTriplet) tCfContent = tPar[8]*tCfContentUnNorm[ix-1];
        else tCfContent = tPar[5]*tCfContentUnNorm[ix-1];

        if(tCfToFit->GetBinContent(ix)!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          tmp = (tCfToFit->GetBinContent(ix) - tCfContent)/tCfToFit->GetBinError(ix);

          fChi2Vec[iAnaly] += tmp*tmp;
          fChi2 += tmp*tmp;

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

/*
//________________________________________________________________________________________________________________
void CoulombFitter::CalculateChi2(int &npar, double &chi2, double *par)
{
  fNCalls++;

  cout << "\tfNCalls = " << fNCalls << endl;

  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = " << par[0] << endl;
  cout << "\t\t\tpar[1] = " << par[1] << endl;
  cout << "\t\t\tpar[2] = " << par[2] << endl;
  cout << "\t\t\tpar[3] = " << par[3] << endl;
  cout << "\t\t\tpar[4] = " << par[4] << endl;
  cout << "\t\t\tpar[5] = " << par[5] << endl;
  cout << "\t\t\tpar[6] = " << par[6] << endl;
  cout << "\t\t\tpar[7] = " << par[7] << endl;
  cout << "\t\t\tpar[8] = " << par[8] << endl;
  cout << "\t\t\tpar[9] = " << par[9] << endl;
  cout << "\t\t\tpar[10] = " << par[10] << endl;

  fChi2 = 0.;
  for(unsigned int i=0; i<fChi2Vec.size(); i++) {fChi2Vec[i] = 0.;}

  fNpFits = 0.;
  fNpFitsVec.resize(fNAnalyses);
  for(unsigned int i=0; i<fNpFitsVec.size(); i++) {fNpFitsVec[i] = 0.;}

  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    if(!fAllOfSameCoulombType)
    {
      //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
      if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
      else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
      else fBohrRadius = 1000000000;
      fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
    }

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {
//cout << "iPartAn = " << iPartAn << endl;
      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();
      TH1* tCfToFit = tKStarCfLite->Cf();

      TAxis* tXaxis = tCfToFit->GetXaxis();
      int tNbinsX = tCfToFit->GetNbinsX();
      int tNbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit

      int tLambdaMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kLambda)->GetMinuitParamNumber();
      int tRadiusMinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRadius)->GetMinuitParamNumber();
      int tRef0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef0)->GetMinuitParamNumber();
      int tImf0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf0)->GetMinuitParamNumber();
      int td0MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd0)->GetMinuitParamNumber();

      int tRef02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kRef02)->GetMinuitParamNumber();
      int tImf02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kImf02)->GetMinuitParamNumber();
      int td02MinuitParamNumber = tFitPartialAnalysis->GetFitParameter(kd02)->GetMinuitParamNumber();


      int tNormMinuitParamNumber = tFitPartialAnalysis->GetFitNormParameter()->GetMinuitParamNumber();

      int tNFitParams = tFitPartialAnalysis->GetNFitParams() + 1;  //the +1 accounts for the normalization parameter
      assert(tNFitParams == 9);
      double *tPar = new double[tNFitParams];

      tPar[0] = par[tLambdaMinuitParamNumber];
      tPar[1] = par[tRadiusMinuitParamNumber];
      tPar[2] = par[tRef0MinuitParamNumber];
      tPar[3] = par[tImf0MinuitParamNumber];
      tPar[4] = par[td0MinuitParamNumber];
      tPar[5] = par[tRef02MinuitParamNumber];
      tPar[6] = par[tImf02MinuitParamNumber];
      tPar[7] = par[td02MinuitParamNumber];
      tPar[8] = par[tNormMinuitParamNumber];

      for(int i=0; i<tNFitParams; i++)  //assure all parameters exist
      {
        if(std::isnan(tPar[i])) {cout <<"CRASH:  In CalculateChi2, a tPar elemement " << i << " DNE!!!!!" << endl;}
        assert(!std::isnan(tPar[i]));
      }

      double tmp;

      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxis->GetBinLowEdge(ix);
        double tKStarMax = tXaxis->GetBinLowEdge(ix+1);

        double tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,iAnaly);
        double tCfContent = tPar[8]*tCfContentUnNorm;

        if(tCfToFit->GetBinContent(ix)!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          tmp = (tCfToFit->GetBinContent(ix) - tCfContent)/tCfToFit->GetBinError(ix);

          fChi2Vec[iAnaly] += tmp*tmp;
          fChi2 += tmp*tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }

      delete[] tPar;
    }

  }

  chi2 = fChi2;
  if(fChi2 < fChi2GlobalMin) fChi2GlobalMin = fChi2;
//gObjectTable->Print();
cout << "fChi2 = " << fChi2 << endl;
cout << "fChi2GlobalMin = " << fChi2GlobalMin << endl << endl;
}
*/


//________________________________________________________________________________________________________________
void CoulombFitter::CalculateFakeChi2(int &npar, double &chi2, double *par)
{
  fNCalls++;
  cout << "\tfNCalls = " << fNCalls << endl;

  cout << "\t\tParameter update: " << endl;
  cout << "\t\t\tpar[0] = Lambda = " << par[0] << endl;
  cout << "\t\t\tpar[1] = Radius = " << par[1] << endl;

  cout << "\t\t\tpar[2] = ReF0  = " << par[2] << endl;
  cout << "\t\t\tpar[3] = ImF0  = " << par[3] << endl;
  cout << "\t\t\tpar[4] = D0    = " << par[4] << endl;


  TAxis *tXaxis = fFakeCf->GetXaxis();
  double tChi2 = 0.;
  double tmp = 0.;
  int tNpfits = 0;

  double tKStarMin, tKStarMax;

  int tNbinsK = fFakeCf->GetNbinsX();

  int tAnalysisNumber = 0;  //TODO

  for(int ix=1; ix<=tNbinsK; ix++)
  {
    tKStarMin = tXaxis->GetBinLowEdge(ix);
    tKStarMax = tXaxis->GetBinLowEdge(ix+1);

    if(fUseStaticPairs && fIncludeSingletAndTriplet) tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else if(fIncludeSingletAndTriplet) tmp = (fFakeCf->GetBinContent(ix) - par[8]*GetFitCfContentComplete(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else if(fUseStaticPairs) tmp = (fFakeCf->GetBinContent(ix) - par[5]*GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    else tmp = (fFakeCf->GetBinContent(ix) - par[5]*GetFitCfContent(tKStarMin,tKStarMax,par,tAnalysisNumber))/fFakeCf->GetBinError(ix);
    tChi2 += tmp*tmp;
    tNpfits++;
  }

  chi2 = tChi2;
cout << "tChi2 = " << tChi2 << endl << endl;
}


//________________________________________________________________________________________________________________
double CoulombFitter::GetChi2(TH1* aFitHistogram)
{
  fChi2 = 0.;
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(iAnaly);
    AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

    int tNFitPartialAnalysis = tFitPairAnalysis->GetNFitPartialAnalysis();
    for(int iPartAn=0; iPartAn<tNFitPartialAnalysis; iPartAn++)
    {

      FitPartialAnalysis* tFitPartialAnalysis = tFitPairAnalysis->GetFitPartialAnalysis(iPartAn);
      CfLite* tKStarCfLite = tFitPartialAnalysis->GetKStarCfLite();
      TH1* tCfToFit = tKStarCfLite->Cf();

      TAxis* tXaxis = tCfToFit->GetXaxis();
      int tNbinsX = tCfToFit->GetNbinsX();
      int tNbinsXToFit = tCfToFit->FindBin(fMaxFitKStar);
      if(tCfToFit->GetBinLowEdge(tNbinsXToFit) == fMaxFitKStar) tNbinsXToFit--;
      int tNbinsXFitHist = aFitHistogram->FindBin(fMaxFitKStar);
      if(aFitHistogram->GetBinLowEdge(tNbinsXFitHist) == fMaxFitKStar) tNbinsXFitHist--;

      if(tNbinsXToFit > tNbinsX) {tNbinsXToFit = tNbinsX;}  //in case I accidentally include an overflow bin in nbinsXToFit
      assert(tNbinsXToFit == tNbinsXFitHist);

      double tmp;
      for(int ix=1; ix <= tNbinsXToFit; ix++)
      {
        double tKStarMin = tXaxis->GetBinLowEdge(ix);
        double tKStarMax = tXaxis->GetBinLowEdge(ix+1);

        double tCfContent = aFitHistogram->GetBinContent(ix);

        if(tCfToFit->GetBinContent(ix)!=0 && tCfContent!=0) //even if only in one single bin, t*Content=0 causes fChi2->nan
        {
          tmp = (tCfToFit->GetBinContent(ix) - tCfContent)/tCfToFit->GetBinError(ix);

          fChi2Vec[iAnaly] += tmp*tmp;
          fChi2 += tmp*tmp;

          fNpFitsVec[iAnaly]++;
          fNpFits++;
        }
      }
    }
  }
  return fChi2;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogram(TString aName, int aAnalysisNumber)
{
  FitPairAnalysis* tFitPairAnalysis = fFitSharedAnalyses->GetFitPairAnalysis(aAnalysisNumber);
  AnalysisType tAnalysisType = tFitPairAnalysis->GetAnalysisType();

  if(!fAllOfSameCoulombType)
  {
    //assert(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM || tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP);
    if(tAnalysisType == kXiKchP || tAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else if(tAnalysisType == kXiKchM || tAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
    else fBohrRadius = 1000000000;
    fWaveFunction->SetCurrentAnalysisType(tAnalysisType);
  }

  int tNFitParams = tFitPairAnalysis->GetNFitParams(); //should be equal to 8

  TH1* tKStarCf = tFitPairAnalysis->GetKStarCf();
//  int tNbins = tKStarCf->GetNbinsX();
  int tNbins = tKStarCf->FindBin(fMaxFitKStar);
  if(tKStarCf->GetBinLowEdge(tNbins) == fMaxFitKStar) tNbins--;
  double tBinKStarMin = tKStarCf->GetBinLowEdge(1);
  double tBinKStarMax = tKStarCf->GetBinLowEdge(tNbins+1);

  TH1D* tReturnHist = new TH1D(aName,aName,tNbins,tBinKStarMin,tBinKStarMax);

  double *tPar = new double[tNFitParams];
  double *tParError = new double[tNFitParams];

  for(int iPar=0; iPar<tNFitParams; iPar++)
  {
    ParameterType tParamType = static_cast<ParameterType>(iPar);
    tPar[iPar] = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValue();
    tParError[iPar] = tFitPairAnalysis->GetFitParameter(tParamType)->GetFitValueError();
  }


  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent;
  for(int ix=1; ix <= tNbins; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);

    if(fUseStaticPairs && fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fUseStaticPairs)
    {
      tCfContentUnNorm = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }
    else
    {
      tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,aAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }
    tReturnHist->SetBinContent(ix,tCfContent);
  }

  delete[] tPar;
  delete[] tParError;

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogramSample(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0, double aImF0, double aD0, double aNorm)
{
  cout << "Beginning CreateFitHistogramSample" << endl;
ChronoTimer tTimer;
tTimer.Start();

  if(!fAllOfSameCoulombType)
  {
    //assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
    else fBohrRadius = 1000000000;
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }

  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[6];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0;
  tPar[3] = aImF0;
  tPar[4] = aD0;
  tPar[5] = aNorm;

  int tAnalysisNumber = 0;  //TODO

  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent, tCfContentv2;
  for(int ix=1; ix <= aNbinsK; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);


//---------------------------------------------
ChronoTimer tLoopTimer;
tLoopTimer.Start();

        tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
        tCfContent = aNorm*tCfContentUnNorm;

tLoopTimer.Stop();
cout << "Finished GetFitCfContent ";
tLoopTimer.PrintInterval();
cout << endl;

/*
//---------------------------------------------
        std::clock_t start2 = std::clock();

        tCfContentv2 = GetFitCfContentSerialv2(tKStarMin,tKStarMax,tPar,tAnalysisNumber);

        double duration2 = (std::clock() - start2)/(double) CLOCKS_PER_SEC;
        cout << "Finished GetFitCfContentSerialv2 in " << duration2 << " seconds" << endl;


        cout << "tCfContent = " << tCfContent << endl;
        cout << "tCfContentv2 = " << tCfContentv2 << endl << endl;
//---------------------------------------------
*/

    tReturnHist->SetBinContent(ix,tCfContent);
    tReturnHist->SetBinError(ix,.1/ix);
  }

tTimer.Stop();
cout << "Finished CreateFitHistogramSample ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}


//________________________________________________________________________________________________________________
TH1* CoulombFitter::CreateFitHistogramSampleComplete(TString aName, AnalysisType aAnalysisType, int aNbinsK, double aKMin, double aKMax, double aLambda, double aR, double aReF0s, double aImF0s, double aD0s, double aReF0t, double aImF0t, double aD0t, double aNorm)
{
  cout << "Beginning CreateFitHistogramSample" << endl;
ChronoTimer tTimer;
tTimer.Start();

  if(!fAllOfSameCoulombType)
  {
    //assert(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM || aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP);
    if(aAnalysisType == kXiKchP || aAnalysisType == kAXiKchM) fBohrRadius = -gBohrRadiusXiK; //attractive
    else if(aAnalysisType == kXiKchM || aAnalysisType == kAXiKchP) fBohrRadius = gBohrRadiusXiK; //repulsive
    else fBohrRadius = 1000000000;
    fWaveFunction->SetCurrentAnalysisType(aAnalysisType);
  }

  TH1D* tReturnHist = new TH1D(aName,aName,aNbinsK,aKMin,aKMax);

  double *tPar = new double[9];
  tPar[0] = aLambda;
  tPar[1] = aR;
  tPar[2] = aReF0s;
  tPar[3] = aImF0s;
  tPar[4] = aD0s;
  tPar[5] = aReF0t;
  tPar[6] = aImF0t;
  tPar[7] = aD0t;
  tPar[8] = aNorm;

  int tAnalysisNumber = 0;  //TODO

  double tKStarMin, tKStarMax, tCfContentUnNorm, tCfContent;
  for(int ix=1; ix <= aNbinsK; ix++)
  {
    tKStarMin = tReturnHist->GetBinLowEdge(ix);
    tKStarMax = tReturnHist->GetBinLowEdge(ix+1);


//---------------------------------------------
//ChronoTimer tLoopTimer;
//tLoopTimer.Start();
    
    if(fUseStaticPairs && fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentCompletewStaticPairs(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fIncludeSingletAndTriplet)
    {
      tCfContentUnNorm = GetFitCfContentComplete(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[8]*tCfContentUnNorm;
    }
    else if(fUseStaticPairs)
    {
      tCfContentUnNorm = GetFitCfContentwStaticPairs(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }
    else
    {
      tCfContentUnNorm = GetFitCfContent(tKStarMin,tKStarMax,tPar,tAnalysisNumber);
      tCfContent = tPar[5]*tCfContentUnNorm;
    }

//tLoopTimer.Stop();
//cout << "Finished GetFitCfContentComplete ";
//tLoopTimer.PrintInterval();
//cout << endl;

    tReturnHist->SetBinContent(ix,tCfContent);
  }

tTimer.Stop();
cout << "Finished CreateFitHistogramSampleComplete ";
tTimer.PrintInterval();

  delete[] tPar;

  fFakeCf = tReturnHist;  //TODO delete this

  return tReturnHist;
}


//________________________________________________________________________________________________________________
void CoulombFitter::DoFit()
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

//  arglist[0] = 0.0000000001;
//  fMinuit->mnexcm("SET EPS", arglist, 1, fErrFlg);

  // Set strategy to be used
  // 0 = economize (many params and/or fcn takes a long time to calculate and/or not interested in precise param erros)
  // 1 = default
  // 2 = Minuit allowed to waste calls in order to ensure precision (fcn evaluated in short time and/or param erros must be reliable)
  arglist[0] = 1;
  fMinuit->mnexcm("SET STR", arglist ,1,fErrFlg);

  // Now ready for minimization step
  arglist[0] = 5000;
  arglist[1] = 0.1;
  fMinuit->mnexcm("MIGRAD", arglist ,2,fErrFlg);  //I do not think MIGRAD will work here because depends on derivates, etc
//  fMinuit->mnexcm("MINI", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
//  fMinuit->mnexcm("SIM", arglist ,2,fErrFlg);  //MINI also uses MIGRAD, so probably won't work
//  fMinuit->mnscan();

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


//TODO return the fit histogram to the analyses
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    TH1* tHistTest = CreateFitHistogram("FitHis",iAnaly);
  }
/*
  for(int iAnaly=0; iAnaly<fNAnalyses; iAnaly++)
  {
    fFitSharedAnalyses->GetFitPairAnalysis(iAnaly)->SetFit(CreateFitFunction("fit",iAnaly));
  }
*/
}


//________________________________________________________________________________________________________________


