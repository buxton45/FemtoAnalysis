///////////////////////////////////////////////////////////////////////////
// InterpolationHistograms:                                              //
///////////////////////////////////////////////////////////////////////////

#include "InterpolationHistograms.h"

#ifdef __ROOT__
ClassImp(InterpolationHistograms)
#endif



//________________________________________________________________________________________________________________
InterpolationHistograms::InterpolationHistograms(TString aSaveFileName, AnalysisType aAnalysisType):

  fSaveFileName(aSaveFileName),
  fAnalysisType(aAnalysisType),
  fSaveFile1(0),
  fSaveFile2(0),
  fSaveFile3(0),

  fSaveFileReal1(0), fSaveFileReal2(0), fSaveFileImag1(0), fSaveFileImag2(0),

  fNbinsK(10), fNbinsR(10), fNbinsTheta(10), fNbinsReF0(10), fNbinsImF0(10), fNbinsD0(10),
  fKStarMin(0.), fRStarMin(0.), fThetaMin(0.), fReF0Min(0.), fImF0Min(0.), fD0Min(0.),
  fKStarMax(1.), fRStarMax(1.), fThetaMax(1.), fReF0Max(1.), fImF0Max(1.), fD0Max(1.),

  fWaveFunction(0),

  fGamowFactor(0),

  fLednickyHFunction(0),

  fHyperGeo1F1Real(0),
  fHyperGeo1F1Imag(0),

  fGTildeReal(0),
  fGTildeImag(0),

  fExpTermReal(0),
  fExpTermImag(0),

  fCoulombScatteringLengthReal(0),
  fCoulombScatteringLengthImag(0)

{
  fWaveFunction = new WaveFunction();
  fSession = fWaveFunction->GetSession();

  fWaveFunction->SetCurrentAnalysisType(fAnalysisType);
  fSaveFileName += TString::Format("_%s", cAnalysisBaseTags[fAnalysisType]);
}



//________________________________________________________________________________________________________________
InterpolationHistograms::~InterpolationHistograms()
{
  cout << "InterpolationHistograms object is being deleted!!!" << endl;
}




//________________________________________________________________________________________________________________
void InterpolationHistograms::SetKStarBinning(int aNbins, double aMin, double aMax)
{
  fNbinsK = aNbins;
  fKStarMin = aMin;
  fKStarMax = aMax;
}
//________________________________________________________________________________________________________________
void InterpolationHistograms::SetRStarBinning(int aNbins, double aMin, double aMax)
{
  fNbinsR = aNbins;
  fRStarMin = aMin;
  fRStarMax = aMax;
}
//________________________________________________________________________________________________________________
void InterpolationHistograms::SetThetaBinning(int aNbins, double aMin, double aMax)
{
  fNbinsTheta = aNbins;
  fThetaMin = aMin;
  fThetaMax = aMax;
}
//________________________________________________________________________________________________________________
void InterpolationHistograms::SetReF0Binning(int aNbins, double aMin, double aMax)
{
  fNbinsReF0 = aNbins;
  fReF0Min = aMin;
  fReF0Max = aMax;
}
//________________________________________________________________________________________________________________
void InterpolationHistograms::SetImF0Binning(int aNbins, double aMin, double aMax)
{
  fNbinsImF0 = aNbins;
  fImF0Min = aMin;
  fImF0Max = aMax;
}
//________________________________________________________________________________________________________________
void InterpolationHistograms::SetD0Binning(int aNbins, double aMin, double aMax)
{
  fNbinsD0 = aNbins;
  fD0Min = aMin;
  fD0Max = aMax;
}



//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildGamowFactor(int aNbinsK, double aKLow, double aKHigh)
{
  TString tName = "GamowFactor";
  fGamowFactor = new TH1D(tName,tName,aNbinsK,aKLow,aKHigh);

  double tKStar, tValue;

cout << "In BuildGamowFactor" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "iK = " << iK << endl;
    tKStar = fGamowFactor->GetXaxis()->GetBinCenter(iK+1);  //histograms start at 1, not 0
    tValue = fWaveFunction->GetGamowFactor(tKStar);
    fGamowFactor->SetBinContent(iK+1,tValue);
  }

}

//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildLednickyHFunction(int aNbinsK, double aKLow, double aKHigh)
{
  TString tName = "LednickyHFunction";
  fLednickyHFunction = new TH1D(tName,tName,aNbinsK,aKLow,aKHigh);

  double tKStar, tValue;

cout << "In BuildLednickyHFunction" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "iK = " << iK << endl;
    tKStar = fLednickyHFunction->GetXaxis()->GetBinCenter(iK+1);  //histograms start at 1, not 0
    tValue = fWaveFunction->GetLednickyHFunction(tKStar);
    fLednickyHFunction->SetBinContent(iK+1,tValue);
  }

}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildHyperGeo1F1Histograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh, int aNbinsTheta, double aThetaLow, double aThetaHigh)
{
  TString tNameReal = "HyperGeo1F1Real";
  TString tNameImag = "HyperGeo1F1Imag";

  fHyperGeo1F1Real = new TH3D(tNameReal,tNameReal, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh, aNbinsTheta,aThetaLow,aThetaHigh);
  fHyperGeo1F1Imag = new TH3D(tNameImag,tNameImag, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh, aNbinsTheta,aThetaLow,aThetaHigh);

  TAxis* tKaxis = fHyperGeo1F1Real->GetXaxis();
  TAxis* tRaxis = fHyperGeo1F1Real->GetYaxis();
  TAxis* tThetaaxis = fHyperGeo1F1Real->GetZaxis();

  double tKStar, tRStar, tTheta;
  complex<double> tValue;

cout << "In BuildHyperGeo1F1Histograms" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "\tiK = " << iK << endl;
    tKStar = tKaxis->GetBinCenter(iK+1); //histograms start at 1, not 0
    for(int iR=0; iR<aNbinsR; iR++)
    {
//cout << "\t\tiR = " << iR << endl;
      tRStar = tRaxis->GetBinCenter(iR+1);
      for(int iTheta=0; iTheta<aNbinsTheta; iTheta++)
      {
//cout << "\t\t\tiTheta = " << iTheta << endl;
        tTheta = tThetaaxis->GetBinCenter(iTheta+1);

        double tEta = fWaveFunction->GetEta(tKStar);
        double tXi = fWaveFunction->GetLowerCaseXi(tKStar,tRStar,tTheta);

        complex<double> tA (0.,-tEta);
        complex<double> tB (1.,0.);
        complex<double> tZ (0.,tXi);

        tValue = fSession->GetHyperGeo1F1(tA,tB,tZ);

        fHyperGeo1F1Real->SetBinContent(iK+1,iR+1,iTheta+1,real(tValue));
        fHyperGeo1F1Imag->SetBinContent(iK+1,iR+1,iTheta+1,imag(tValue)); 
      }
    }
  }

}

//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildGTildeHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh)
{
  TString tNameReal = "GTildeReal";
  TString tNameImag = "GTildeImag";

  fGTildeReal = new TH2D(tNameReal,tNameReal, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh);
  fGTildeImag = new TH2D(tNameImag,tNameImag, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh);

  TAxis* tKaxis = fGTildeReal->GetXaxis();
  TAxis* tRaxis = fGTildeReal->GetYaxis();

  double tKStar, tRStar;
  complex<double> tValue;

cout << "In BuildGTildeHistograms" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "\tiK = " << iK << endl;
    tKStar = tKaxis->GetBinCenter(iK+1); //histograms start at 1, not 0
    for(int iR=0; iR<aNbinsR; iR++)
    {
//cout << "\t\tiR = " << iR << endl;
      tRStar = tRaxis->GetBinCenter(iR+1);

      tValue = fWaveFunction->GetGTilde(tKStar,tRStar);

      fGTildeReal->SetBinContent(iK+1,iR+1,real(tValue));
      fGTildeImag->SetBinContent(iK+1,iR+1,imag(tValue));
    }
  }

}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildExpTermHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsR, double aRLow, double aRHigh, int aNbinsTheta, double aThetaLow, double aThetaHigh)
{
  TString tNameReal = "ExpTermReal";
  TString tNameImag = "ExpTermImag";

  fExpTermReal = new TH3D(tNameReal,tNameReal, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh, aNbinsTheta,aThetaLow,aThetaHigh);
  fExpTermImag = new TH3D(tNameImag,tNameImag, aNbinsK,aKLow,aKHigh, aNbinsR,aRLow,aRHigh, aNbinsTheta,aThetaLow,aThetaHigh);

  TAxis* tKaxis = fExpTermReal->GetXaxis();
  TAxis* tRaxis = fExpTermReal->GetYaxis();
  TAxis* tThetaaxis = fExpTermReal->GetZaxis();

  double tKStar, tRStar, tTheta;
  complex<double> tValue;
  complex<double> tImI (0.,1.);

cout << "In BuildExpTermHistograms" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "\tiK = " << iK << endl;
    tKStar = tKaxis->GetBinCenter(iK+1); //histograms start at 1, not 0
    for(int iR=0; iR<aNbinsR; iR++)
    {
//cout << "\t\tiR = " << iR << endl;
      tRStar = tRaxis->GetBinCenter(iR+1);
      for(int iTheta=0; iTheta<aNbinsTheta; iTheta++)
      {
//cout << "\t\t\tiTheta = " << iTheta << endl;
        tTheta = tThetaaxis->GetBinCenter(iTheta+1);

        tValue = exp(-tImI*(tKStar/hbarc)*tRStar*cos(tTheta));

        fExpTermReal->SetBinContent(iK+1,iR+1,iTheta+1,real(tValue));
        fExpTermImag->SetBinContent(iK+1,iR+1,iTheta+1,imag(tValue)); 
      }
    }
  }

}



//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildScatteringLengthHistograms(int aNbinsK, double aKLow, double aKHigh, int aNbinsReF0, double aReF0Low, double aReF0High, int aNbinsImF0, double aImF0Low, double aImF0High, int aNbinsD0, double aD0Low, double aD0High)
{
  int tDim = 4;
  int tNbins[4] = {aNbinsK,aNbinsReF0,aNbinsImF0,aNbinsD0};
  double tXmin[4] = {aKLow,aReF0Low,aImF0Low,aD0Low};
  double tXmax[4] = {aKHigh,aReF0High,aImF0High,aD0High};

  TString tNameReal = "CoulombScatteringLengthReal";
  TString tNameImag = "CoulombScatteringLengthImag";


  fCoulombScatteringLengthReal = new THnD(tNameReal,tNameReal,tDim,tNbins,tXmin,tXmax);
  fCoulombScatteringLengthImag = new THnD(tNameImag,tNameImag,tDim,tNbins,tXmin,tXmax);
    TAxis* tKaxis = fCoulombScatteringLengthReal->GetAxis(0);
    TAxis* tReF0axis = fCoulombScatteringLengthReal->GetAxis(1);
    TAxis* tImF0axis = fCoulombScatteringLengthReal->GetAxis(2);
    TAxis* tD0axis = fCoulombScatteringLengthReal->GetAxis(3);

  double tKStar, tReF0, tImF0, tD0;
  complex<double> tValue;
  int tCurrentBin[4];

cout << "In BuildScatteringLengthHistograms" << endl;

  for(int iK=0; iK<aNbinsK; iK++)
  {
cout << "\tiK = " << iK << endl;
    tKStar = tKaxis->GetBinCenter(iK+1); //histograms start at 1, not 0
    for(int iReF0=0; iReF0<aNbinsReF0; iReF0++)
    {
cout << "\t\tiReF0 = " << iReF0 << endl;
      tReF0 = tReF0axis->GetBinCenter(iReF0+1);
      for(int iImF0=0; iImF0<aNbinsImF0; iImF0++)
      {
//cout << "\t\t\tiImF0 = " << iImF0 << endl;
        tImF0 = tImF0axis->GetBinCenter(iImF0+1);
        for(int iD0=0; iD0<aNbinsD0; iD0++)
        {
//cout << "\t\t\t\tiD0 = " << iD0 << endl;
          tD0 = tD0axis->GetBinCenter(iD0+1);

          tValue = fWaveFunction->GetScatteringLength(tKStar,tReF0,tImF0,tD0);

          //int tCurrentBin[4] = {iK+1,iReF0+1,iImF0+1,iD0+1};
          tCurrentBin[0] = iK+1;
          tCurrentBin[1] = iReF0+1;
          tCurrentBin[2] = iImF0+1;
          tCurrentBin[3] = iD0+1;

          fCoulombScatteringLengthReal->SetBinContent(tCurrentBin,real(tValue));
          fCoulombScatteringLengthImag->SetBinContent(tCurrentBin,imag(tValue));
        }
      }
    }
  }
}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildAndSaveSplitScatteringLengthHistograms()
{
  TString tSaveFileRealName1 = fSaveFileName+"ScatLenReal1.root";
  TString tSaveFileImagName1 = fSaveFileName+"ScatLenImag1.root";
  TString tSaveFileRealName2 = fSaveFileName+"ScatLenReal2.root";
  TString tSaveFileImagName2 = fSaveFileName+"ScatLenImag2.root";

  int tNbinsK1 = fNbinsK/2;
  double tKStarMin1 = fKStarMin;
  double tKStarMax1 = fKStarMax/2;

  int tNbinsK2 = fNbinsK/2;
  double tKStarMin2 = tKStarMax1;
  double tKStarMax2 = fKStarMax;

  //--------------------------------------

  BuildScatteringLengthHistograms(tNbinsK1,tKStarMin1,tKStarMax1, fNbinsReF0,fReF0Min,fReF0Max, fNbinsImF0,fImF0Min,fImF0Max, fNbinsD0,fD0Min,fD0Max);

  fSaveFileReal1 = new TFile(tSaveFileRealName1, "recreate");
    fCoulombScatteringLengthReal->Write();
    fCoulombScatteringLengthReal->Delete();
  fSaveFileReal1->Close();

  fSaveFileImag1 = new TFile(tSaveFileImagName1, "recreate");
    fCoulombScatteringLengthImag->Write();
    fCoulombScatteringLengthImag->Delete();
  fSaveFileImag1->Close();

//--------------------------------------

  BuildScatteringLengthHistograms(tNbinsK2,tKStarMin2,tKStarMax2, fNbinsReF0,fReF0Min,fReF0Max, fNbinsImF0,fImF0Min,fImF0Max, fNbinsD0,fD0Min,fD0Max);

  fSaveFileReal2 = new TFile(tSaveFileRealName2, "recreate");
    fCoulombScatteringLengthReal->Write();
    fCoulombScatteringLengthReal->Delete();
  fSaveFileReal2->Close();

  fSaveFileImag2 = new TFile(tSaveFileImagName2, "recreate");
    fCoulombScatteringLengthImag->Write();
    fCoulombScatteringLengthImag->Delete();
  fSaveFileImag2->Close();

}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildAndSaveAllOthers()
{
  TString tSaveFileName1 = fSaveFileName+".root";

  //-------------------------------

  fSaveFile1 = new TFile(tSaveFileName1, "recreate");
/*
  BuildGamowFactor(fNbinsK,fKStarMin,fKStarMax);
    fGamowFactor->Write();
    fGamowFactor->Delete();
*/
  BuildHyperGeo1F1Histograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax, fNbinsTheta,fThetaMin,fThetaMax);
    fHyperGeo1F1Real->Write();
    fHyperGeo1F1Imag->Write();
    fHyperGeo1F1Real->Delete();
    fHyperGeo1F1Imag->Delete();

  BuildGTildeHistograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax);
    fGTildeReal->Write();
    fGTildeImag->Write();
    fGTildeReal->Delete();
    fGTildeImag->Delete();
/*
  BuildExpTermHistograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax, fNbinsTheta,fThetaMin,fThetaMax);
    fExpTermReal->Write();
    fExpTermImag->Write();
    fExpTermReal->Delete();
    fExpTermImag->Delete();
*/
  fSaveFile1->Close();

}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildAndSaveAll()
{
  TString tSaveFileName1 = fSaveFileName+".root";
  TString tSaveFileName2 = fSaveFileName+"ScatLenReal.root";
  TString tSaveFileName3 = fSaveFileName+"ScatLenImag.root";

  //-------------------------------

  fSaveFile1 = new TFile(tSaveFileName1, "recreate");

  BuildGamowFactor(fNbinsK,fKStarMin,fKStarMax);
    fGamowFactor->Write();
    fGamowFactor->Delete();

  BuildHyperGeo1F1Histograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax, fNbinsTheta,fThetaMin,fThetaMax);
    fHyperGeo1F1Real->Write();
    fHyperGeo1F1Imag->Write();
    fHyperGeo1F1Real->Delete();
    fHyperGeo1F1Imag->Delete();

  BuildGTildeHistograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax);
    fGTildeReal->Write();
    fGTildeImag->Write();
    fGTildeReal->Delete();
    fGTildeImag->Delete();

  BuildExpTermHistograms(fNbinsK,fKStarMin,fKStarMax, fNbinsR,fRStarMin,fRStarMax, fNbinsTheta,fThetaMin,fThetaMax);
    fExpTermReal->Write();
    fExpTermImag->Write();
    fExpTermReal->Delete();
    fExpTermImag->Delete();

  fSaveFile1->Close();

  //-------------------------------

  BuildScatteringLengthHistograms(fNbinsK,fKStarMin,fKStarMax, fNbinsReF0,fReF0Min,fReF0Max, fNbinsImF0,fImF0Min,fImF0Max, fNbinsD0,fD0Min,fD0Max);

  fSaveFile2 = new TFile(tSaveFileName2, "recreate");
    fCoulombScatteringLengthReal->Write();
    fCoulombScatteringLengthReal->Delete();
  fSaveFile2->Close();

  fSaveFile3 = new TFile(tSaveFileName3, "recreate");
    fCoulombScatteringLengthImag->Write();
    fCoulombScatteringLengthImag->Delete();
  fSaveFile3->Close();

}


//________________________________________________________________________________________________________________
void InterpolationHistograms::BuildAndSaveLednickyHFunction()
{
  TString tSaveFileName1 = fSaveFileName+".root";

  //-------------------------------

  fSaveFile1 = new TFile(tSaveFileName1, "recreate");

  BuildLednickyHFunction(fNbinsK,fKStarMin,fKStarMax);
    fLednickyHFunction->Write();
    fLednickyHFunction->Delete();

  fSaveFile1->Close();

  //-------------------------------

}





