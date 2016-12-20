///////////////////////////////////////////////////////////////////////////
// FitChi2Histograms:                                                    //
///////////////////////////////////////////////////////////////////////////


#include "FitChi2Histograms.h"

#ifdef __ROOT__
ClassImp(FitChi2Histograms)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitChi2Histograms::FitChi2Histograms() :
  fMinuitFitParameters(0),
  fChi2HistCollection(0),
  fInvChi2HistCollection(0),
  fChi2CountsHistCollection(0)
{
  fChi2HistCollection = new TObjArray();
  fInvChi2HistCollection = new TObjArray();
  fChi2CountsHistCollection = new TObjArray();

  //TODO if desired, setup a way to initiate these to different values
  SetupChi2BinInfo(500,0.,5000);

  SetupBinInfo(kLambda,100,0.,1.);
  SetupBinInfo(kRadius,200,0.,20.);
  
  SetupBinInfo(kRef0,100,-5.0,5.);
  SetupBinInfo(kImf0,100,-5.0,5.);
  SetupBinInfo(kd0,100,-5.0,5.);

  SetupBinInfo(kRef02,100,-5.0,5.);
  SetupBinInfo(kImf02,100,-5.0,5.);
  SetupBinInfo(kd02,100,-5.0,5.);
}

//________________________________________________________________________________________________________________
FitChi2Histograms::~FitChi2Histograms()
{
  cout << "FitChi2Histograms object is being deleted!!!" << endl;
}

//________________________________________________________________________________________________________________
int FitChi2Histograms::Factorial(int aInput)
{
  if(aInput == 0) return 1;

  return aInput*Factorial(aInput-1);
}

//________________________________________________________________________________________________________________
int FitChi2Histograms::nChoosek(int aN, int aK)
{
  return Factorial(aN)/(Factorial(aK)*Factorial(aN-aK));
}

//________________________________________________________________________________________________________________
BinInfo FitChi2Histograms::GetBinInfo(ParameterType aParamType)
{
  switch(aParamType)
  {
    case kLambda:
      return fLambdaBinInfo;
    break;

    case kRadius:
      return fRadiusBinInfo;
    break;

    case kRef0:
      return fReF0sBinInfo;
    break;

    case kImf0:
      return fImF0sBinInfo;
    break;

    case kd0:
      return fD0sBinInfo;
    break;

    case kRef02:
      return fReF0tBinInfo;
    break;

    case kImf02:
      return fImF0tBinInfo;
    break;

    case kd02:
      return fD0tBinInfo;
    break;

  }

}

//________________________________________________________________________________________________________________
void FitChi2Histograms::SetupBinInfo(ParameterType aParamType, int aNBins, double aMin, double aMax)
{
  BinInfo tBinInfo;

  tBinInfo.nBins = aNBins;
  tBinInfo.minVal = aMin;
  tBinInfo.maxVal = aMax;
  tBinInfo.binWidth = (aMax-aMin)/aNBins;

  switch(aParamType)
  {
    case kLambda:
      fLambdaBinInfo = tBinInfo;
    break;

    case kRadius:
      fRadiusBinInfo = tBinInfo;
    break;

    case kRef0:
      fReF0sBinInfo = tBinInfo;
    break;

    case kImf0:
      fImF0sBinInfo = tBinInfo;
    break;

    case kd0:
      fD0sBinInfo = tBinInfo;
    break;

    case kRef02:
      fReF0tBinInfo = tBinInfo;
    break;

    case kImf02:
      fImF0tBinInfo = tBinInfo;
    break;

    case kd02:
      fD0tBinInfo = tBinInfo;
    break;

  }

}

/*
//________________________________________________________________________________________________________________
void FitChi2Histograms::SetupBinInfo(ParameterType aParamType, int aNBins, double aMin, double aMax)
{
  BinInfo tBinInfo;

  tBinInfo.nBins = aNBins;
  tBinInfo.minVal = aMin;
  tBinInfo.maxVal = aMax;
  tBinInfo.binWidth = (aMax-aMin)/aNBins;

  SetBinInfo(aParamType,tBinInfo);
}
*/
//________________________________________________________________________________________________________________
void FitChi2Histograms::SetupChi2BinInfo(int aNBins, double aMin, double aMax)
{
  fChi2BinInfo.nBins = aNBins;
  fChi2BinInfo.minVal = aMin;
  fChi2BinInfo.maxVal = aMax;
  fChi2BinInfo.binWidth = (aMax-aMin)/aNBins;
}


//________________________________________________________________________________________________________________
void FitChi2Histograms::AddParameter(int aMinuitParamNumber, FitParameter* aParam)
{
  assert((int)fMinuitFitParameters.size() == aMinuitParamNumber);  //ensures parameters are loaded into vector
                                                              // (sequentially) in correct spot
  fMinuitFitParameters.push_back(aParam);
}

/*
//________________________________________________________________________________________________________________
void FitChi2Histograms::InitiateHistograms()
{
  int tNParameters = fMinuitFitParameters.size();
  int tNHistograms = nChoosek(tNParameters,2);
  assert(tNHistograms == 28);  //True in typical case of 8 parameters

  TString tHistName;
  FitParameter *tFitParam1, *tFitParam2;
  BinInfo tBinInfo1, tBinInfo2;

  for(int iHist1=0; iHist1<tNParameters; iHist1++)
  {
    for(int iHist2=iHist1+1; iHist2<tNParameters; iHist2++)
    {
      tFitParam1 = fMinuitFitParameters[iHist1];
      tFitParam2 = fMinuitFitParameters[iHist2];

      tHistName = tFitParam1->GetName() + TString("vs") + tFitParam2->GetName();

      tBinInfo1 = GetBinInfo(tFitParam1->GetType());
      tBinInfo2 = GetBinInfo(tFitParam2->GetType());

      TH3D* tHisto = new TH3D(tHistName,tHistName, fChi2BinInfo.nBins,fChi2BinInfo.minVal,fChi2BinInfo.maxVal, tBinInfo1.nBins,tBinInfo1.minVal,tBinInfo1.maxVal, tBinInfo2.nBins,tBinInfo2.minVal,tBinInfo2.maxVal);

      fChi2HistCollection->Add(tHisto);
    }
  }

  assert(fChi2HistCollection->GetEntries() == tNHistograms);
}


//________________________________________________________________________________________________________________
void FitChi2Histograms::FillHistograms(double aChi2, double *aParams)
{
  int tNParameters = fMinuitFitParameters.size();
  int tIndex = 0;
  for(int iHist1=0; iHist1<tNParameters; iHist1++)
  {
    for(int iHist2=iHist1+1; iHist2<tNParameters; iHist2++)
    {
      ((TH3D*)fChi2HistCollection->At(tIndex))->Fill(aChi2,aParams[iHist1],aParams[iHist2]);
      tIndex++;
    }
  }

}
*/


//________________________________________________________________________________________________________________
void FitChi2Histograms::InitiateHistograms()
{
  int tNParameters = fMinuitFitParameters.size();
  int tNHistograms = nChoosek(tNParameters,2);
  assert(tNHistograms == 28 || tNHistograms == 10);  //True in typical case of 8 parameters

  TString tHistName, tInvHistName, tCountsHistName;
  FitParameter *tFitParam1, *tFitParam2;
  BinInfo tBinInfo1, tBinInfo2;

  for(int iHist1=0; iHist1<tNParameters; iHist1++)
  {
    for(int iHist2=iHist1+1; iHist2<tNParameters; iHist2++)
    {
      tFitParam1 = fMinuitFitParameters[iHist1];
      tFitParam2 = fMinuitFitParameters[iHist2];

      tHistName = tFitParam1->GetName() + TString("vs") + tFitParam2->GetName();
      tInvHistName = tHistName + TString("InvChi2");
      tCountsHistName = tHistName + TString("Counts");

      tBinInfo1 = GetBinInfo(tFitParam1->GetType());
      tBinInfo2 = GetBinInfo(tFitParam2->GetType());

      TH2D* tHist = new TH2D(tHistName,tHistName, tBinInfo1.nBins,tBinInfo1.minVal,tBinInfo1.maxVal, tBinInfo2.nBins,tBinInfo2.minVal,tBinInfo2.maxVal);
      fChi2HistCollection->Add(tHist);

      TH2D* tInvHist = new TH2D(tInvHistName,tInvHistName, tBinInfo1.nBins,tBinInfo1.minVal,tBinInfo1.maxVal, tBinInfo2.nBins,tBinInfo2.minVal,tBinInfo2.maxVal);
      fInvChi2HistCollection->Add(tInvHist);

      TH2D* tCountsHist = new TH2D(tCountsHistName,tCountsHistName, tBinInfo1.nBins,tBinInfo1.minVal,tBinInfo1.maxVal, tBinInfo2.nBins,tBinInfo2.minVal,tBinInfo2.maxVal);
      fChi2CountsHistCollection->Add(tCountsHist);
    }
  }

  assert(fChi2HistCollection->GetEntries() == tNHistograms);
}


//________________________________________________________________________________________________________________
void FitChi2Histograms::FillHistograms(double aChi2, double *aParams)
{
  int tNParameters = fMinuitFitParameters.size();
  int tIndex = 0;
  for(int iHist1=0; iHist1<tNParameters; iHist1++)
  {
    for(int iHist2=iHist1+1; iHist2<tNParameters; iHist2++)
    {
      ((TH2D*)fChi2HistCollection->At(tIndex))->Fill(aParams[iHist1],aParams[iHist2],aChi2);
      ((TH2D*)fInvChi2HistCollection->At(tIndex))->Fill(aParams[iHist1],aParams[iHist2],pow(aChi2,-1));
      ((TH2D*)fChi2CountsHistCollection->At(tIndex))->Fill(aParams[iHist1],aParams[iHist2]);
      tIndex++;
    }
  }

}


//________________________________________________________________________________________________________________
void FitChi2Histograms::SaveHistograms(TString aFileName)
{
  //-----First, normalize
  int tNParameters = fMinuitFitParameters.size();
  int tIndex = 0;
  for(int iHist1=0; iHist1<tNParameters; iHist1++)
  {
    for(int iHist2=iHist1+1; iHist2<tNParameters; iHist2++)
    {
      ((TH2D*)fChi2HistCollection->At(tIndex))->Divide(((TH2D*)fChi2CountsHistCollection->At(tIndex)));
      ((TH2D*)fInvChi2HistCollection->At(tIndex))->Divide(((TH2D*)fChi2CountsHistCollection->At(tIndex)));
      tIndex++;
    }
  }




  TFile* tFile = new TFile(aFileName, "recreate");
//    fChi2HistCollection->Write("fChi2HistCollection",TObject::kSingleKey);
    fChi2HistCollection->Write("fChi2HistCollection");
    fInvChi2HistCollection->Write("fInvChi2HistCollection");
  tFile->Close();
}


