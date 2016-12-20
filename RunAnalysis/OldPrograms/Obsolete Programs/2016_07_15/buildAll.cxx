///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAll                                                              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAll.h"

#ifdef __ROOT__
ClassImp(buildAll)
#endif

//________________________________________________________________________________________________________________
buildAll::buildAll(vector<TString> &aVectorOfFileNames):
  fDebug(kFALSE),
  fOutputPurityFitInfo(kFALSE),

  //KStar CF------------------
  fMinNormCF(0.32),
  fMaxNormCF(0.4),
  fMinNormBinCF(60),
  fMaxNormBinCF(75),

  //Average Separation CF------------------
  fMinNormAvgSepCF(14.99),
  fMaxNormAvgSepCF(19.99),
  fMinNormBinAvgSepCF(150),
  fMaxNormBinAvgSepCF(200),

  //Separation CFs------------------
  fMinNormBinSepCF(150),
  fMaxNormBinSepCF(200)


{
  SetVectorOfFileNames(aVectorOfFileNames);

  //Purity calculations------------------
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;
  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = 1.15068;
  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;

  fK0Short1BgFitLow[0] = 0.423677;
  fK0Short1BgFitLow[1] = 0.452;
  fK0Short1BgFitHigh[0] = 0.536;
  fK0Short1BgFitHigh[1] = 0.563677;
  fK0Short1ROI[0] = KaonMass-0.013677;
  fK0Short1ROI[1] = KaonMass+0.020323;


}


//________________________________________________________________________________________________________________
buildAll::~buildAll()
{
  cout << "Object is being deleted" << endl;
}

//________________________________________________________________________________________________________________
TH1F* buildAll::buildCF(TString name, TString title, TH1* Num, TH1* Denom, int aMinNormBin, int aMaxNormBin)
{
//  cout << "For hist "<< Num->GetName()<<", num pair = " << Num->Integral(1,Num->FindBin(.05)) << endl;
  double NumScale = Num->Integral(aMinNormBin,aMaxNormBin);
  double DenScale = Denom->Integral(aMinNormBin,aMaxNormBin);

  TH1F* CF = (TH1F*)Num->Clone(name);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!CF->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << name << ", so calling it now" << endl;
    CF->Sumw2();
  }

  CF->Divide(Denom);
  CF->Scale(DenScale/NumScale);
  CF->SetTitle(title);

  return CF;
}

//________________________________________________________________________________________________________________
TH1F* buildAll::CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin)
{
  double scale = 0.;
  int counter = 0;
  double temp = 0.;

  int SizeOfCfCollection = aCfCollection->GetEntries();
  int SizeOfNumCollection = aNumCollection->GetEntries();

  if(SizeOfCfCollection != SizeOfNumCollection) {cout << "ERROR: In CombineCFs, the CfCollection and NumCollection ARE NOT EQUAL IN SIZE!!!!" << endl;}

  TH1F* ReturnCf = (TH1F*)(aCfCollection->At(0))->Clone(aReturnName);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnCf->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aReturnName << ", so calling it now" << endl;
    ReturnCf->Sumw2();
  }

  ReturnCf->SetTitle(aReturnTitle);

  TH1F* Num1 = (TH1F*)(aNumCollection->At(0));
  temp = Num1->Integral(aMinNormBin,aMaxNormBin);

  scale+=temp;
  counter++;

  ReturnCf->Scale(temp);

  if(fDebug)
  {
    cout << "Return Name: " << ReturnCf->GetName() << endl;
    cout << "  Including: " << ((TH1F*)aCfCollection->At(0))->GetName() << endl;
    cout << "\t" << Num1->GetName() << "  NumScale: " << Num1->Integral(aMinNormBin,aMaxNormBin) << endl;
  }

  for(int i=1; i<SizeOfCfCollection; i++)
  {
    temp = ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin);

    ReturnCf->Add((TH1F*)aCfCollection->At(i),temp);
    scale += temp;
    counter ++;

    if(fDebug)
    {
      cout << "  Including: " << ((TH1F*)aCfCollection->At(i))->GetName() << endl;
      cout << "\t" << ((TH1F*)aNumCollection->At(i))->GetName() << " NumScale: " << ((TH1F*)aNumCollection->At(i))->Integral(aMinNormBin,aMaxNormBin) << endl;
    }
  }
  ReturnCf->Scale(1./scale);

  if(fDebug)
  {
    cout << "  Final overall scale = " << scale << endl;
    cout << "  Number of histograms combined = " << counter << endl << endl;
  }

  return ReturnCf;

}



//________________________________________________________________________________________________________________
void buildAll::SetVectorOfFileNames(vector<TString> &aVectorOfNames)
{
  int InputVectorSize = aVectorOfNames.size();
  if(InputVectorSize != 5) {cout << "MISSING FILE(S)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl << endl <<  "INPUT VECTOR HAS SIZE: " << InputVectorSize << endl;}

  for(int i=0; i<InputVectorSize; i++)
  {
    fVectorOfFileNames.push_back(aVectorOfNames[i]);
  }

  //Double check that all names were loaded correctly
  int OutputVectorSize = fVectorOfFileNames.size();
  cout << "fVectorOfFileNames now has size " << OutputVectorSize << " and contains..." << endl;
  for(int i=0; i<OutputVectorSize; i++)
  {
    cout << "\t" << fVectorOfFileNames[i] << endl;
  }
  cout << endl;
}


//________________________________________________________________________________________________________________
TObjArray* buildAll::ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName)
{
  TFile f1(aFileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)femtolist->FindObject(aDirectoryName);

  return ReturnArray;
}



//________________________________________________________________________________________________________________
TH1F* buildAll::GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH1F *ReturnHisto = (TH1F*)aAnalysisDirectory->FindObject(aHistoName);
  TH1F *ReturnHistoClone = (TH1F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHistoClone->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aCloneHistoName << ", so calling it now" << endl;
    ReturnHistoClone->Sumw2();
  }

  return (TH1F*)ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TObjArray* buildAll::BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin)
{
  TObjArray* ReturnCollection = new TObjArray();

  int SizeOfNumCollection = aNumCollection->GetEntries();
  int SizeOfDenCollection = aDenCollection->GetEntries();

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildCollectionOfCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildCollectionOfCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}

  for(int i=0; i<SizeOfNumCollection; i++)
  {
    //------Add a file tag--------
    TString aNumName = ((TH1F*)aNumCollection->At(i))->GetName();
    TString aHistoName = aContainedHistosBaseName;

    if (aNumName.Contains("Bp1")) {aHistoName+="_Bp1";}
    else if (aNumName.Contains("Bp2")) {aHistoName+="_Bp2";}
    else if (aNumName.Contains("Bm1")) {aHistoName+="_Bm1";}
    else if (aNumName.Contains("Bm2")) {aHistoName+="_Bm2";}
    else if (aNumName.Contains("Bm3")) {aHistoName+="_Bm3";}

    ReturnCollection->Add(buildCF(aHistoName,aHistoName,((TH1F*)aNumCollection->At(i)),((TH1F*)aDenCollection->At(i)),aMinNormBin,aMaxNormBin));
  }
  return ReturnCollection;
}


//________________________________________________________________________________________________________________
void buildAll::SetPurityRegimes(TH1F* aLambdaPurity)
{
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;

  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = aLambdaPurity->GetBinLowEdge(aLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;

}


//________________________________________________________________________________________________________________
void buildAll::SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity)
{
  fLamBgFitLow[0] = 1.09;
  fLamBgFitLow[1] = 1.102;

  fLamBgFitHigh[0] = 1.130;
  fLamBgFitHigh[1] = aLambdaPurity->GetBinLowEdge(aLambdaPurity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fLamROI[0] = LambdaMass-0.0038;
  fLamROI[1] = LambdaMass+0.0038;


  fK0Short1BgFitLow[0] = aK0Short1Purity->GetBinLowEdge(1);  //default:  Purity->GetBinLowEdge(1)
  fK0Short1BgFitLow[1] = 0.452;

  fK0Short1BgFitHigh[0] = 0.536;
  fK0Short1BgFitHigh[1] = aK0Short1Purity->GetBinLowEdge(aK0Short1Purity->GetNbinsX()+1);  //default:  Purity->GetBinLowEdge(Purity->GetNbinsX()+1)

  fK0Short1ROI[0] = KaonMass-0.013677;
  fK0Short1ROI[1] = KaonMass+0.020323;

}




//________________________________________________________________________________________________________________
TObjArray* buildAll::CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
{
  char buffer[50];
  sprintf(buffer, "fitBgd_%s",aReturnFitName);

  reject = kTRUE;
  ffBgFitLow[0] = aBgFitLow[0];
  ffBgFitLow[1] = aBgFitLow[1];
  ffBgFitHigh[0] = aBgFitHigh[0];
  ffBgFitHigh[1] = aBgFitHigh[1];
  TF1 *fitBgd = new TF1("fitBgd",PurityBgFitFunction,aPurityHisto->GetBinLowEdge(1),aPurityHisto->GetBinLowEdge(aPurityHisto->GetNbinsX()+1),5);
  if(fOutputPurityFitInfo){aPurityHisto->Fit("fitBgd","0");}  //option 0 = Do not plot the result of the fit
  else{aPurityHisto->Fit("fitBgd","0q");}  //option q = quiet mode = minimum printing

  reject = kFALSE;
  TF1 *fitBgd2 = new TF1(buffer,PurityBgFitFunction,aPurityHisto->GetBinLowEdge(1),aPurityHisto->GetBinLowEdge(aPurityHisto->GetNbinsX()+1),5);
  fitBgd2->SetParameters(fitBgd->GetParameters());

  //--------------------------------------------------------------------------------------------
  double bgd = fitBgd2->Integral(aROI[0],aROI[1]);
  bgd /= aPurityHisto->GetBinWidth(0);  //divide by bin size
  cout << aReturnFitName << ": " << "bgd = " << bgd << endl;
  //-----
  double sigpbgd = aPurityHisto->Integral(aPurityHisto->FindBin(aROI[0]),aPurityHisto->FindBin(aROI[1]));
  cout << aReturnFitName << ": " << "sig+bgd = " << sigpbgd << endl;
  //-----
  double sig = sigpbgd-bgd;
  cout << aReturnFitName << ": " << "sig = " << sig << endl;
  //-----
  double pur = sig/sigpbgd;
  cout << aReturnFitName << ": " << "Pur = " << pur << endl << endl;

  TVectorD *vInfo = new TVectorD(4);
    (*vInfo)(0) = bgd;
    (*vInfo)(1) = sigpbgd;
    (*vInfo)(2) = sig;
    (*vInfo)(3) = pur;

  TVectorD *vROI = new TVectorD(2);
    (*vROI)(0) = aROI[0];
    (*vROI)(1) = aROI[1];

  TVectorD *vBgFitLow = new TVectorD(2);
    (*vBgFitLow)(0) = aBgFitLow[0];
    (*vBgFitLow)(1) = aBgFitLow[1];

  TVectorD *vBgFitHigh = new TVectorD(2);
    (*vBgFitHigh)(0) = aBgFitHigh[0];
    (*vBgFitHigh)(1) = aBgFitHigh[1];
  //--------------------------------------------------------------------------------------------
  TObjArray* temp = new TObjArray();
  temp->Add(fitBgd2);
  temp->Add(vInfo);
  temp->Add(vROI);
  temp->Add(vBgFitLow);
  temp->Add(vBgFitHigh);
  return temp;

}

//________________________________________________________________________________________________________________
TH1F* buildAll::CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms)
{
  TH1F* ReturnHisto = (TH1F*)(aCollectionOfHistograms->At(0))->Clone(aReturnHistoName);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aReturnHistoName << ", so calling it now" << endl;
    ReturnHisto->Sumw2();
  }

  for(int i=1; i<aCollectionOfHistograms->GetEntries(); i++)
  {
    ReturnHisto->Add(((TH1F*)aCollectionOfHistograms->At(i)));
  }

  return ReturnHisto;

}


//________________________________________________________________________________________________________________
void buildAll::DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg)
{
  TF1* fitBgd = (TF1*)aFitList->At(0);
    fitBgd->SetLineColor(4);
  //-----
  TVectorD* vInfo = (TVectorD*)aFitList->At(1);
  //-----
  TVectorD* vROI = (TVectorD*)aFitList->At(2);
  //-----
  TVectorD* vBgFitLow = (TVectorD*)aFitList->At(3);
  //-----
  TVectorD* vBgFitHigh = (TVectorD*)aFitList->At(4);
  //--------------------------------------------------------------------------------------------

  TLine *lROImin, *lROImax, *lBgFitLowMin, *lBgFitLowMax, *lBgFitHighMin, *lBgFitHighMax;

  if(!ZoomBg)
  {
    double HistoMaxValue = aPurityHisto->GetMaximum();
    lROImin = new TLine((*vROI)(0),0,(*vROI)(0),HistoMaxValue);
    lROImax = new TLine((*vROI)(1),0,(*vROI)(1),HistoMaxValue);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),0,(*vBgFitLow)(0),HistoMaxValue);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),0,(*vBgFitLow)(1),HistoMaxValue);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),0,(*vBgFitHigh)(0),HistoMaxValue);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),0,(*vBgFitHigh)(1),HistoMaxValue);
  }

  if(ZoomBg)
  {
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin((*vBgFitLow)(0)),aPurityHisto->FindBin((*vBgFitLow)(1)));
      double maxLow = aPurityHisto->GetMaximum();
      double minLow = aPurityHisto->GetMinimum();
    aPurityHisto->GetXaxis()->SetRange(aPurityHisto->FindBin((*vBgFitHigh)(0)),aPurityHisto->FindBin((*vBgFitHigh)(1))-1);
      double maxHigh = aPurityHisto->GetMaximum();
      double minHigh = aPurityHisto->GetMinimum();
    double maxBg;
      if(maxLow>maxHigh) maxBg = maxLow;
      else maxBg = maxHigh;
      //cout << "Background max = " << maxBg << endl;
    double minBg;
      if(minLow<minHigh) minBg = minLow;
      else minBg = minHigh;
      //cout << "Background min = " << minBg << endl;
    //--Extend the y-range that I plot

    double rangeBg = maxBg-minBg;
    maxBg+=rangeBg/10.;
    minBg-=rangeBg/10.;

    aPurityHisto->GetXaxis()->SetRange(1,aPurityHisto->GetNbinsX());
    aPurityHisto->GetYaxis()->SetRangeUser(minBg,maxBg);
    //--------------------------------------------------------------------------------------------
    lROImin = new TLine((*vROI)(0),minBg,(*vROI)(0),maxBg);
    lROImax = new TLine((*vROI)(1),minBg,(*vROI)(1),maxBg);
    //-----
    lBgFitLowMin = new TLine((*vBgFitLow)(0),minBg,(*vBgFitLow)(0),maxBg);
    lBgFitLowMax = new TLine((*vBgFitLow)(1),minBg,(*vBgFitLow)(1),maxBg);
    //-----
    lBgFitHighMin = new TLine((*vBgFitHigh)(0),minBg,(*vBgFitHigh)(0),maxBg);
    lBgFitHighMax = new TLine((*vBgFitHigh)(1),minBg,(*vBgFitHigh)(1),maxBg);
  }

  //--------------------------------------------------------------------------------------------
  aPurityHisto->SetLineColor(1);
  aPurityHisto->SetLineWidth(3);

  lROImin->SetLineColor(3);
  lROImax->SetLineColor(3);

  lBgFitLowMin->SetLineColor(2);
  lBgFitLowMax->SetLineColor(2);

  lBgFitHighMin->SetLineColor(2);
  lBgFitHighMax->SetLineColor(2);


  //--------------------------------------------------------------------------------------------
  aPurityHisto->DrawCopy("Ehist");
  fitBgd->Draw("same");
  lROImin->Draw();
  lROImax->Draw();
  lBgFitLowMin->Draw();
  lBgFitLowMax->Draw();
  lBgFitHighMin->Draw();
  lBgFitHighMax->Draw();

  if(!ZoomBg)
  {
    TPaveText *myText = new TPaveText(0.12,0.65,0.42,0.85,"NDC");
    char buffer[50];
    double purity = (*vInfo)(3);
    const char* title = aPurityHisto->GetName();
    //char title[20] = aPurityHisto->GetName();
    sprintf(buffer, "%s = %.2f%%",title, 100.*purity);
    myText->AddText(buffer);
    myText->Draw();
  }
}


//-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAll)

//________________________________________________________________________________________________________________
TH2F* buildAll::Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
{
  TH2F *ReturnHisto = (TH2F*)aAnalysisDirectory->FindObject(aHistoName);
  TH2F *ReturnHistoClone = (TH2F*)ReturnHisto->Clone(aCloneHistoName);
  ReturnHistoClone->SetDirectory(0);
  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHistoClone->GetSumw2N())
  {
    cout << "Sumw2 NOT already called on " << aCloneHistoName << ", so calling it now" << endl;
    ReturnHistoClone->Sumw2();
  }

  return (TH2F*)ReturnHistoClone;
}


//________________________________________________________________________________________________________________
TObjArray* buildAll::BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumberOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin)
{
  cout << "***** Working on collection of SepCfs containing: " << aContainedHistosBaseName << " *****" << endl;

  TObjArray* ReturnCollection = new TObjArray();

  int SizeOfNumCollection = a2DNumCollection->GetEntries();
  int SizeOfDenCollection = a2DDenCollection->GetEntries();

  //-----Rescaling normalization bins
  aMinNormBin /= aRebinFactor;
  aMaxNormBin /= aRebinFactor;

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildSepCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildSepCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}

  for(int aBin=1; aBin<=aNumberOfXbins; aBin++)
  {
    TObjArray* CfCollection = new TObjArray();
    TObjArray* NumCollection = new TObjArray();

    for(int i=0; i<SizeOfNumCollection; i++)
    {

      //------Add a file tag--------
      TString aNumName = ((TH2F*)a2DNumCollection->At(i))->GetName();
      TString aHistoName = aContainedHistosBaseName;
        aHistoName += "_bin";
        aHistoName += aBin;

      if (aNumName.Contains("Bp1")) {aHistoName+="_Bp1";}
      else if (aNumName.Contains("Bp2")) {aHistoName+="_Bp2";}
      else if (aNumName.Contains("Bm1")) {aHistoName+="_Bm1";}
      else if (aNumName.Contains("Bm2")) {aHistoName+="_Bm2";}
      else if (aNumName.Contains("Bm3")) {aHistoName+="_Bm3";}

      TH1D* Num = ((TH2F*)a2DNumCollection->At(i))->ProjectionY("",aBin,aBin);
        TH1D* NumClone = (TH1D*)Num->Clone();  //Projections are always TH1D for some reason.  My functions work with TH1F, not sure how they will react to TH1D, but worth checking
				        //At some point I should go back and do dynamic_cast where necessary
      TH1D* Den = ((TH2F*)a2DDenCollection->At(i))->ProjectionY("",aBin,aBin);
        TH1D* DenClone = (TH1D*)Den->Clone();

      //------Rebinning
      NumClone->Rebin(aRebinFactor);
      DenClone->Rebin(aRebinFactor);
      //-----

      CfCollection->Add(buildCF(aHistoName,aHistoName,NumClone,DenClone,aMinNormBin,aMaxNormBin));
      NumCollection->Add(NumClone);

    }
    TString CfName = aContainedHistosBaseName+"_bin";
    CfName+=aBin;
    TH1F* CfForSpecificBin = CombineCFs(CfName,CfName,CfCollection,NumCollection,aMinNormBin, aMaxNormBin);

    ReturnCollection->Add(CfForSpecificBin);

  }

  cout << "Final size of SepCfCollection: " << ReturnCollection->GetEntries() << endl;
  return ReturnCollection;

}


//________________________________________________________________________________________________________________
TObjArray* buildAll::BuildCowCfs(int aRebinFactor, TString aContainedHistosBaseName, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin)
{
  cout << "***** Working on collection of CowCfs containing: " << aContainedHistosBaseName << " *****" << endl;

  TObjArray* ReturnCollection = new TObjArray();

  int SizeOfNumCollection = a2DNumCollection->GetEntries();
  int SizeOfDenCollection = a2DDenCollection->GetEntries();

  //-----Rescaling normalization bins
  aMinNormBin /= aRebinFactor;
  aMaxNormBin /= aRebinFactor;

  if( (SizeOfNumCollection != 5) && (SizeOfDenCollection != 5) ) {cout << "In BuildCowCfs, the input collection sizes DO NOT EQUAL 5!!!!!" << endl;}
  if( SizeOfNumCollection != SizeOfDenCollection ) {cout << "In BuildCowCfs, the NumCollection and DenCollection DO NOT HAVE EQUAL SIZES!!!!!!!" << endl;}


  TObjArray* CfCollection1 = new TObjArray();
  TObjArray* NumCollection1 = new TObjArray();

  TObjArray* CfCollection2 = new TObjArray();
  TObjArray* NumCollection2 = new TObjArray();

  for(int i=0; i<SizeOfNumCollection; i++)
  {

    //------Add a file tag--------
    TString aNumName = ((TH2F*)a2DNumCollection->At(i))->GetName();
    TString aHistoName1 = aContainedHistosBaseName;
      aHistoName1+="1";
    TString aHistoName2 = aContainedHistosBaseName;
      aHistoName2+="2";

    if (aNumName.Contains("Bp1")) {aHistoName1+="_Bp1";}
    else if (aNumName.Contains("Bp2")) {aHistoName1+="_Bp2";}
    else if (aNumName.Contains("Bm1")) {aHistoName1+="_Bm1";}
    else if (aNumName.Contains("Bm2")) {aHistoName1+="_Bm2";}
    else if (aNumName.Contains("Bm3")) {aHistoName1+="_Bm3";}

    TH1D* Num1 = ((TH2F*)a2DNumCollection->At(i))->ProjectionY(aHistoName1,1,20);
      TH1D* Num1Clone = (TH1D*)Num1->Clone();  //Projections are always TH1D for some reason.  My functions work with TH1F, not sure how they will react to TH1D, but worth checking
				        //At some point I should go back and do dynamic_cast where necessary
    TH1D* Den1 = ((TH2F*)a2DDenCollection->At(i))->ProjectionY(aHistoName1,1,20);
      TH1D* Den1Clone = (TH1D*)Den1->Clone();

    TH1D* Num2 = ((TH2F*)a2DNumCollection->At(i))->ProjectionY(aHistoName2,21,40);
      TH1D* Num2Clone = (TH1D*)Num2->Clone();  //Projections are always TH1D for some reason.  My functions work with TH1F, not sure how they will react to TH1D, but worth checking
				        //At some point I should go back and do dynamic_cast where necessary
    TH1D* Den2 = ((TH2F*)a2DDenCollection->At(i))->ProjectionY(aHistoName2,21,40);
      TH1D* Den2Clone = (TH1D*)Den2->Clone();

    //------Rebinning
    Num1Clone->Rebin(aRebinFactor);
    Den1Clone->Rebin(aRebinFactor);

    Num2Clone->Rebin(aRebinFactor);
    Den2Clone->Rebin(aRebinFactor);
    //-----

    CfCollection1->Add(buildCF(aHistoName1,aHistoName1,Num1Clone,Den1Clone,aMinNormBin,aMaxNormBin));
    NumCollection1->Add(Num1Clone);

    CfCollection2->Add(buildCF(aHistoName2,aHistoName2,Num2Clone,Den2Clone,aMinNormBin,aMaxNormBin));
    NumCollection2->Add(Num2Clone);

  }
  TString CfName1 = aContainedHistosBaseName+"1";
  TH1F* CfForSpecificBin1 = CombineCFs(CfName1,CfName1,CfCollection1,NumCollection1,aMinNormBin, aMaxNormBin);

  TString CfName2 = aContainedHistosBaseName+"2";
  TH1F* CfForSpecificBin2 = CombineCFs(CfName2,CfName2,CfCollection2,NumCollection2,aMinNormBin, aMaxNormBin);

  ReturnCollection->Add(CfForSpecificBin1);
  ReturnCollection->Add(CfForSpecificBin2);

 
  cout << "Final size of CowCfCollection: " << ReturnCollection->GetEntries() << endl;
  return ReturnCollection;

}

//________________________________________________________________________________________________________________
