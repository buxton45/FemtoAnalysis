///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamK03                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamK03.h"

#ifdef __ROOT__
ClassImp(buildAllcLamK03)
#endif

//________________________________________________________________________________________________________________
buildAllcLamK03::buildAllcLamK03(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0):
  fDebug(kFALSE),
  fOutputPurityFitInfo(kFALSE),

  //General stuff----------------
  fDirNameLamK0(aDirNameLamK0),
  fDirNameALamK0(aDirNameALamK0),

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
  SetAnalysisDirectories();


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
buildAllcLamK03::~buildAllcLamK03()
{
  cout << "Object is being deleted" << endl;
}

//________________________________________________________________________________________________________________
TH1F* buildAllcLamK03::buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin)
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
TH1F* buildAllcLamK03::CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin)
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
void buildAllcLamK03::SetVectorOfFileNames(vector<TString> &aVectorOfNames)
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
TObjArray* buildAllcLamK03::ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName)
{
  TFile f1(aFileName);
  TList *femtolist = (TList*)f1.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)femtolist->FindObject(aDirectoryName);

  return ReturnArray;
}

//________________________________________________________________________________________________________________
void buildAllcLamK03::SetAnalysisDirectories()
{
  TString FileNameBp1, FileNameBp2, FileNameBm1, FileNameBm2, FileNameBm3;
  //The files should be ordered Bp1, Bp2, Bm1, Bm2, Bm3...but if they are not, the following for loop
  // will pick out the correct file names to associate with the directories

  for(unsigned int i=0; i<fVectorOfFileNames.size(); i++)
  {
    if(fVectorOfFileNames[i].Contains("Bp1")){FileNameBp1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bp2")){FileNameBp2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm1")){FileNameBm1 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm2")){FileNameBm2 = fVectorOfFileNames[i];}
    else if(fVectorOfFileNames[i].Contains("Bm3")){FileNameBm3 = fVectorOfFileNames[i];}
    else{cout << "ERROR: In SetAnalysisDirectories....a file name was found to NOT contain Bp1,Bp2,Bm1,Bm2,or Bm3!!!!!" << endl;}
  }

  fDirLamK0Bp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameLamK0);
  fDirALamK0Bp1 = ConnectAnalysisDirectory(FileNameBp1,fDirNameALamK0);

  fDirLamK0Bp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameLamK0);
  fDirALamK0Bp2 = ConnectAnalysisDirectory(FileNameBp2,fDirNameALamK0);

  fDirLamK0Bm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameLamK0);
  fDirALamK0Bm1 = ConnectAnalysisDirectory(FileNameBm1,fDirNameALamK0);

  fDirLamK0Bm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameLamK0);
  fDirALamK0Bm2 = ConnectAnalysisDirectory(FileNameBm2,fDirNameALamK0);

  fDirLamK0Bm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameLamK0);
  fDirALamK0Bm3 = ConnectAnalysisDirectory(FileNameBm3,fDirNameALamK0);
}

//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK03::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
{
  if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bp1;}
  else if(aFile.Contains("Bp1") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bp1;}

  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bp2;}
  else if(aFile.Contains("Bp2") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bp2;}

  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm1;}
  else if(aFile.Contains("Bm1") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm1;}

  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm2;}
  else if(aFile.Contains("Bm2") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm2;}

  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("LamK0")){return fDirLamK0Bm3;}
  else if(aFile.Contains("Bm3") && aDirectoryName.EqualTo("ALamK0")){return fDirALamK0Bm3;}

  else
  {
    cout << "ERROR in GetAnalysisDirectory:  No directory to return!!!!!" << endl;
    return 0;
  }

}


//________________________________________________________________________________________________________________
TH1F* buildAllcLamK03::GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
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
TObjArray* buildAllcLamK03::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
{
  TObjArray* ReturnCollection = new TObjArray();

  if(aDirectoryName.EqualTo(fDirNameLamK0))
  {
    ReturnCollection->Add(GetHistoClone(fDirLamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirLamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirLamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirLamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirLamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamK0))
  {
    ReturnCollection->Add(GetHistoClone(fDirALamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(GetHistoClone(fDirALamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(GetHistoClone(fDirALamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(GetHistoClone(fDirALamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(GetHistoClone(fDirALamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOfHistograms!!!!!!!!!!!!!!!!" << endl;}

  return ReturnCollection;

}

//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK03::BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin)
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
void buildAllcLamK03::BuildCFCollections()
{
  cout << "_________________________Beginning BuildCFCollections()_________________________" << endl;

  TString NumName_LamK0 = "NumKStarCF_LamK0";
  TString DenName_LamK0 = "DenKStarCF_LamK0";

  TString NumName_ALamK0 = "NumKStarCF_ALamK0";
  TString DenName_ALamK0 = "DenKStarCF_ALamK0";

  cout << "fMinNormBinCF (default constructor) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (default constructor) = " << fMaxNormBinCF << endl;

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamK0.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamK0.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamK0.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_LamK0+=CentralityTag;
  DenName_LamK0+=CentralityTag;
  NumName_ALamK0+=CentralityTag;
  DenName_ALamK0+=CentralityTag;
  //----------------------------------------------------------------------

  fNumCollection_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_LamK0);
  fDenCollection_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_LamK0);

  fNumCollection_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_ALamK0);
  fDenCollection_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_ALamK0);

  //-----Set the normalization bins from the normalization values (ex set fMinNormBinCF from fMinNormCF)
  fMinNormBinCF = ((TH1F*)fNumCollection_LamK0->At(0))->FindBin(fMinNormCF);
  fMaxNormBinCF = ((TH1F*)fNumCollection_LamK0->At(0))->FindBin(fMaxNormCF);
  cout << "fMinNormBinCF (set) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (set) = " << fMaxNormBinCF << endl << endl;

  fCfCollection_LamK0 = BuildCollectionOfCfs("fCf_LamK0",fNumCollection_LamK0,fDenCollection_LamK0,fMinNormBinCF,fMaxNormBinCF);
  fCf_LamK0_Tot = CombineCFs("fCf_LamK0_Tot","Lam-K0 (Tot)",fCfCollection_LamK0,fNumCollection_LamK0,fMinNormBinCF,fMaxNormBinCF);

  fCfCollection_ALamK0 = BuildCollectionOfCfs("fCf_ALamK0",fNumCollection_ALamK0,fDenCollection_ALamK0,fMinNormBinCF,fMaxNormBinCF);
  fCf_ALamK0_Tot = CombineCFs("fCf_ALamK0_Tot","ALam-K0 (Tot)",fCfCollection_ALamK0,fNumCollection_ALamK0,fMinNormBinCF,fMaxNormBinCF);
/*
  //--------------BpTot-----------------------------------------------------
  TObjArray* fNumCollection_LamK0_BpTot = new TObjArray();
    fNumCollection_LamK0_BpTot->Add(fNumCollection_LamK0->At(0));
    fNumCollection_LamK0_BpTot->Add(fNumCollection_LamK0->At(1));
  TObjArray* fCfCollection_LamK0_BpTot = new TObjArray();
    fCfCollection_LamK0_BpTot->Add(fCfCollection_LamK0->At(0));
    fCfCollection_LamK0_BpTot->Add(fCfCollection_LamK0->At(1));
  fCf_LamK0_BpTot = CombineCFs("fCf_LamK0_BpTot","Lam-K0 (B+ Tot)",fCfCollection_LamK0_BpTot,fNumCollection_LamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamK0_BpTot = new TObjArray();
    fNumCollection_ALamK0_BpTot->Add(fNumCollection_ALamK0->At(0));
    fNumCollection_ALamK0_BpTot->Add(fNumCollection_ALamK0->At(1));
  TObjArray* fCfCollection_ALamK0_BpTot = new TObjArray();
    fCfCollection_ALamK0_BpTot->Add(fCfCollection_ALamK0->At(0));
    fCfCollection_ALamK0_BpTot->Add(fCfCollection_ALamK0->At(1));
  fCf_ALamK0_BpTot = CombineCFs("fCf_ALamK0_BpTot","ALam-K0 (B+ Tot)",fCfCollection_ALamK0_BpTot,fNumCollection_ALamK0_BpTot,fMinNormBinCF,fMaxNormBinCF);


  //--------------BmTot-----------------------------------------------------
  TObjArray *fNumCollection_LamK0_BmTot = new TObjArray();
    fNumCollection_LamK0_BmTot->Add(fNumCollection_LamK0->At(2));
    fNumCollection_LamK0_BmTot->Add(fNumCollection_LamK0->At(3));
    fNumCollection_LamK0_BmTot->Add(fNumCollection_LamK0->At(4));
  TObjArray* fCfCollection_LamK0_BmTot = new TObjArray();
    fCfCollection_LamK0_BmTot->Add(fCfCollection_LamK0->At(2));
    fCfCollection_LamK0_BmTot->Add(fCfCollection_LamK0->At(3));
    fCfCollection_LamK0_BmTot->Add(fCfCollection_LamK0->At(4));
  fCf_LamK0_BmTot = CombineCFs("fCf_LamK0_BmTot","Lam-K0 (B- Tot)",fCfCollection_LamK0_BmTot,fNumCollection_LamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);

  TObjArray* fNumCollection_ALamK0_BmTot = new TObjArray();
    fNumCollection_ALamK0_BmTot->Add(fNumCollection_ALamK0->At(2));
    fNumCollection_ALamK0_BmTot->Add(fNumCollection_ALamK0->At(3));
    fNumCollection_ALamK0_BmTot->Add(fNumCollection_ALamK0->At(4));
  TObjArray* fCfCollection_ALamK0_BmTot = new TObjArray();
    fCfCollection_ALamK0_BmTot->Add(fCfCollection_ALamK0->At(2));
    fCfCollection_ALamK0_BmTot->Add(fCfCollection_ALamK0->At(3));
    fCfCollection_ALamK0_BmTot->Add(fCfCollection_ALamK0->At(4));
  fCf_ALamK0_BmTot = CombineCFs("fCf_ALamK0_BmTot","ALam-K0 (B- Tot)",fCfCollection_ALamK0_BmTot,fNumCollection_ALamK0_BmTot,fMinNormBinCF,fMaxNormBinCF);
*/

  cout << "_________________________Done BuildCFCollections()_________________________" << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawFinalCFs(TCanvas *aCanvas)
{
  aCanvas->cd();
  gStyle->SetOptStat(0);

  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aCfLamK0 = (TH1F*)fCf_LamK0_Tot->Clone();
  TH1F *aCfALamK0 = (TH1F*)fCf_ALamK0_Tot->Clone();

  TAxis *xax1 = aCfLamK0->GetXaxis();
    xax1->SetTitle("k* (GeV/c)");
    xax1->SetTitleSize(0.05);
    xax1->SetTitleOffset(1.0);
    //xax1->CenterTitle();
  TAxis *yax1 = aCfLamK0->GetYaxis();
    yax1->SetRangeUser(0.8,1.1);
    yax1->SetTitle("C(k*)");
    yax1->SetTitleSize(0.05);
    yax1->SetTitleOffset(1.0);
    yax1->CenterTitle();

  aCfLamK0->SetMarkerStyle(20);
  aCfLamK0->SetMarkerSize(1);
  aCfLamK0->SetMarkerColor(1);
  aCfLamK0->SetLineColor(1);
  aCfLamK0->SetTitle("#Lambda(#bar{#Lambda})-K^{0}");

  aCfALamK0->SetMarkerStyle(20);
  aCfALamK0->SetMarkerSize(1);
  aCfALamK0->SetMarkerColor(2);
  aCfALamK0->SetLineColor(2);

  aCfLamK0->Draw();
  aCfALamK0->Draw("same");

  TLine *line = new TLine(0,1,0.4,1);
  line->SetLineColor(14);
  line->Draw();

  TLegend *leg1 = new TLegend(0.65,0.15,0.85,0.35);
    leg1->SetFillColor(0);
    leg1->AddEntry(aCfLamK0, "#Lambda-K^{0}", "p");
    leg1->AddEntry(aCfALamK0, "#bar{#Lambda}-K^{0}", "p");
    leg1->Draw();

}

//________________________________________________________________________________________________________________
void buildAllcLamK03::BuildAvgSepCollections()
{
  cout << "_________________________Beginning BuildAvgSepCollections()_________________________" << endl;

  TString NumName_AvgSepPosPos_LamK0 = "NumPosPosAvgSepCF_LamK0";
  TString DenName_AvgSepPosPos_LamK0 = "DenPosPosAvgSepCF_LamK0";
  TString NumName_AvgSepPosNeg_LamK0 = "NumPosNegAvgSepCF_LamK0";
  TString DenName_AvgSepPosNeg_LamK0 = "DenPosNegAvgSepCF_LamK0";
  TString NumName_AvgSepNegPos_LamK0 = "NumNegPosAvgSepCF_LamK0";
  TString DenName_AvgSepNegPos_LamK0 = "DenNegPosAvgSepCF_LamK0";
  TString NumName_AvgSepNegNeg_LamK0 = "NumNegNegAvgSepCF_LamK0";
  TString DenName_AvgSepNegNeg_LamK0 = "DenNegNegAvgSepCF_LamK0";

  TString NumName_AvgSepPosPos_ALamK0 = "NumPosPosAvgSepCF_ALamK0";
  TString DenName_AvgSepPosPos_ALamK0 = "DenPosPosAvgSepCF_ALamK0";
  TString NumName_AvgSepPosNeg_ALamK0 = "NumPosNegAvgSepCF_ALamK0";
  TString DenName_AvgSepPosNeg_ALamK0 = "DenPosNegAvgSepCF_ALamK0";
  TString NumName_AvgSepNegPos_ALamK0 = "NumNegPosAvgSepCF_ALamK0";
  TString DenName_AvgSepNegPos_ALamK0 = "DenNegPosAvgSepCF_ALamK0";
  TString NumName_AvgSepNegNeg_ALamK0 = "NumNegNegAvgSepCF_ALamK0";
  TString DenName_AvgSepNegNeg_ALamK0 = "DenNegNegAvgSepCF_ALamK0";

  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamK0.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamK0.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamK0.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_AvgSepPosPos_LamK0 += CentralityTag;
  DenName_AvgSepPosPos_LamK0 += CentralityTag;
  NumName_AvgSepPosNeg_LamK0 += CentralityTag;
  DenName_AvgSepPosNeg_LamK0 += CentralityTag;
  NumName_AvgSepNegPos_LamK0 += CentralityTag;
  DenName_AvgSepNegPos_LamK0 += CentralityTag;
  NumName_AvgSepNegNeg_LamK0 += CentralityTag;
  DenName_AvgSepNegNeg_LamK0 += CentralityTag;

  NumName_AvgSepPosPos_ALamK0 += CentralityTag;
  DenName_AvgSepPosPos_ALamK0 += CentralityTag;
  NumName_AvgSepPosNeg_ALamK0 += CentralityTag;
  DenName_AvgSepPosNeg_ALamK0 += CentralityTag;
  NumName_AvgSepNegPos_ALamK0 += CentralityTag;
  DenName_AvgSepNegPos_ALamK0 += CentralityTag;
  NumName_AvgSepNegNeg_ALamK0 += CentralityTag;
  DenName_AvgSepNegNeg_ALamK0 += CentralityTag;
  //----------------------------------------------------------------------
  cout << "fMinNormBinAvgSepCF (default constructor) = " << fMinNormBinAvgSepCF << endl;
  cout << "fMaxNormBinAvgSepCF (default constructor) = " << fMaxNormBinAvgSepCF << endl;

  //--LamK0
  fAvgSepNumCollection_PosPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepPosPos_LamK0);
  fAvgSepDenCollection_PosPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepPosPos_LamK0);

  //-----Set the normalization bins from the normalization values (ex set fMinNormBinAvgSepCF from fMinNormAvgSepCF)
  fMinNormBinAvgSepCF = ((TH1F*)fAvgSepNumCollection_PosPos_LamK0->At(0))->FindBin(fMinNormAvgSepCF);
  fMaxNormBinAvgSepCF = ((TH1F*)fAvgSepNumCollection_PosPos_LamK0->At(0))->FindBin(fMaxNormAvgSepCF);
  cout << "fMinNormBinAvgSepCF (set) = " << fMinNormBinAvgSepCF << endl;
  cout << "fMaxNormBinAvgSepCF (set) = " << fMaxNormBinAvgSepCF << endl << endl;

  fAvgSepCfCollection_PosPos_LamK0 = BuildCollectionOfCfs("fAvgSepCf_PosPos_LamK0",fAvgSepNumCollection_PosPos_LamK0,fAvgSepDenCollection_PosPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosPos_LamK0_Tot = CombineCFs("fAvgSepCf_PosPos_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_PosPos_LamK0,fAvgSepNumCollection_PosPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_PosNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepPosNeg_LamK0);
  fAvgSepDenCollection_PosNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepPosNeg_LamK0);
  fAvgSepCfCollection_PosNeg_LamK0 = BuildCollectionOfCfs("fAvgSepCf_PosNeg_LamK0",fAvgSepNumCollection_PosNeg_LamK0,fAvgSepDenCollection_PosNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosNeg_LamK0_Tot = CombineCFs("fAvgSepCf_PosNeg_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_PosNeg_LamK0,fAvgSepNumCollection_PosNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepNegPos_LamK0);
  fAvgSepDenCollection_NegPos_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepNegPos_LamK0);
  fAvgSepCfCollection_NegPos_LamK0 = BuildCollectionOfCfs("fAvgSepCf_NegPos_LamK0",fAvgSepNumCollection_NegPos_LamK0,fAvgSepDenCollection_NegPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegPos_LamK0_Tot = CombineCFs("fAvgSepCf_NegPos_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_NegPos_LamK0,fAvgSepNumCollection_NegPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,NumName_AvgSepNegNeg_LamK0);
  fAvgSepDenCollection_NegNeg_LamK0 = LoadCollectionOfHistograms(fDirNameLamK0,DenName_AvgSepNegNeg_LamK0);
  fAvgSepCfCollection_NegNeg_LamK0 = BuildCollectionOfCfs("fAvgSepCf_NegNeg_LamK0",fAvgSepNumCollection_NegNeg_LamK0,fAvgSepDenCollection_NegNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegNeg_LamK0_Tot = CombineCFs("fAvgSepCf_NegNeg_LamK0_Tot","Lam-K0 (Tot)",fAvgSepCfCollection_NegNeg_LamK0,fAvgSepNumCollection_NegNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  //--ALam-K0
  fAvgSepNumCollection_PosPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepPosPos_ALamK0);
  fAvgSepDenCollection_PosPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepPosPos_ALamK0);
  fAvgSepCfCollection_PosPos_ALamK0 = BuildCollectionOfCfs("fAvgSepCf_PosPos_ALamK0",fAvgSepNumCollection_PosPos_ALamK0,fAvgSepDenCollection_PosPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosPos_ALamK0_Tot = CombineCFs("fAvgSepCf_PosPos_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_PosPos_ALamK0,fAvgSepNumCollection_PosPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_PosNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepPosNeg_ALamK0);
  fAvgSepDenCollection_PosNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepPosNeg_ALamK0);
  fAvgSepCfCollection_PosNeg_ALamK0 = BuildCollectionOfCfs("fAvgSepCf_PosNeg_ALamK0",fAvgSepNumCollection_PosNeg_ALamK0,fAvgSepDenCollection_PosNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_PosNeg_ALamK0_Tot = CombineCFs("fAvgSepCf_PosNeg_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_PosNeg_ALamK0,fAvgSepNumCollection_PosNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepNegPos_ALamK0);
  fAvgSepDenCollection_NegPos_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepNegPos_ALamK0);
  fAvgSepCfCollection_NegPos_ALamK0 = BuildCollectionOfCfs("fAvgSepCf_NegPos_ALamK0",fAvgSepNumCollection_NegPos_ALamK0,fAvgSepDenCollection_NegPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegPos_ALamK0_Tot = CombineCFs("fAvgSepCf_NegPos_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_NegPos_ALamK0,fAvgSepNumCollection_NegPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  fAvgSepNumCollection_NegNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,NumName_AvgSepNegNeg_ALamK0);
  fAvgSepDenCollection_NegNeg_ALamK0 = LoadCollectionOfHistograms(fDirNameALamK0,DenName_AvgSepNegNeg_ALamK0);
  fAvgSepCfCollection_NegNeg_ALamK0 = BuildCollectionOfCfs("fAvgSepCf_NegNeg_ALamK0",fAvgSepNumCollection_NegNeg_ALamK0,fAvgSepDenCollection_NegNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);
  fAvgSepCf_NegNeg_ALamK0_Tot = CombineCFs("fAvgSepCf_NegNeg_ALamK0_Tot","ALam-K0 (Tot)",fAvgSepCfCollection_NegNeg_ALamK0,fAvgSepNumCollection_NegNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  cout << "_________________________Done BuildAvgSepCollections()_________________________" << endl << endl << endl;
}

//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aAvgSepCf_PosPos_LamK0 = (TH1F*)fAvgSepCf_PosPos_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_PosNeg_LamK0 = (TH1F*)fAvgSepCf_PosNeg_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegPos_LamK0 = (TH1F*)fAvgSepCf_NegPos_LamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegNeg_LamK0 = (TH1F*)fAvgSepCf_NegNeg_LamK0_Tot->Clone();
  //
  TH1F *aAvgSepCf_PosPos_ALamK0 = (TH1F*)fAvgSepCf_PosPos_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_PosNeg_ALamK0 = (TH1F*)fAvgSepCf_PosNeg_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegPos_ALamK0 = (TH1F*)fAvgSepCf_NegPos_ALamK0_Tot->Clone();
  TH1F *aAvgSepCf_NegNeg_ALamK0 = (TH1F*)fAvgSepCf_NegNeg_ALamK0_Tot->Clone();

  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);


  aCanvasLamK0->cd();
  aCanvasLamK0->Divide(2,2);

  aCanvasLamK0->cd(1);
  aAvgSepCf_PosPos_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosPos_LamK0->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(2);
  aAvgSepCf_PosNeg_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosNeg_LamK0->SetTitle("p(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(3);
  aAvgSepCf_NegPos_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegPos_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_LamK0->Draw();
  line->Draw();

  aCanvasLamK0->cd(4);
  aAvgSepCf_NegNeg_LamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegNeg_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_LamK0->Draw();
  line->Draw();




  aCanvasALamK0->cd();
  aCanvasALamK0->Divide(2,2);

  aCanvasALamK0->cd(1);
  aAvgSepCf_PosPos_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosPos_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(2);
  aAvgSepCf_PosNeg_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_PosNeg_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(3);
  aAvgSepCf_NegPos_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegPos_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0->cd(4);
  aAvgSepCf_NegNeg_ALamK0->GetYaxis()->SetRangeUser(-0.5,5.);
  aAvgSepCf_NegNeg_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_ALamK0->Draw();
  line->Draw();

}













//________________________________________________________________________________________________________________
void buildAllcLamK03::SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity)
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
TObjArray* buildAllcLamK03::CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2])
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
TH1F* buildAllcLamK03::CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms)
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
void buildAllcLamK03::BuildPurityCollections()
{
  cout << "_________________________Beginning BuildPurityCollections()_________________________" << endl;

  TString Name_LambdaPurity = "LambdaPurity";
  TString Name_K0Short1Purity = "K0ShortPurity1";
  TString Name_AntiLambdaPurity = "AntiLambdaPurity";
  TString Name_K0Short2Purity = "K0ShortPurity1";
  //----------------------------------------------------------
  fLambdaPurityHistogramCollection = LoadCollectionOfHistograms(fDirNameLamK0,Name_LambdaPurity);
  fK0Short1PurityHistogramCollection = LoadCollectionOfHistograms(fDirNameLamK0,Name_K0Short1Purity);

  fAntiLambdaPurityHistogramCollection = LoadCollectionOfHistograms(fDirNameALamK0,Name_AntiLambdaPurity);
  fK0Short2PurityHistogramCollection = LoadCollectionOfHistograms(fDirNameALamK0,Name_K0Short2Purity);
  //----------------------------------------------------------
  SetPurityRegimes((TH1F*)fLambdaPurityHistogramCollection->At(0),(TH1F*)fK0Short1PurityHistogramCollection->At(0));
  //----------------------------------------------------------
  cout << "********** Lambda Purity for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fLambdaPurityHistogramCollection->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("LambdaPurity",(TH1F*)fLambdaPurityHistogramCollection->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fLambdaPurityListCollection.push_back(TempList);
  }

  cout << "********** K0Short1 Purity for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fK0Short1PurityHistogramCollection->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("K0Short1Purity",(TH1F*)fK0Short1PurityHistogramCollection->At(i),fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
    fK0Short1PurityListCollection.push_back(TempList);
  }

  cout << "********** AntiLambda Purity for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fAntiLambdaPurityHistogramCollection->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("AntiLambdaPurity",(TH1F*)fAntiLambdaPurityHistogramCollection->At(i),fLamBgFitLow,fLamBgFitHigh,fLamROI);
    fAntiLambdaPurityListCollection.push_back(TempList);
  }

  cout << "********** K0Short2 Purity for each individual file (Bp1, Bp2, ...) **********" << endl;
  for(int i=0; i<fK0Short2PurityHistogramCollection->GetEntries(); i++)
  {
    TObjArray *TempList = new TObjArray();
    TempList = CalculatePurity("K0Short2Purity",(TH1F*)fK0Short2PurityHistogramCollection->At(i),fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
    fK0Short2PurityListCollection.push_back(TempList);
  }
  //----------------------------------------------------------
  fLambdaPurityTot = CombineCollectionOfHistograms("fLambdaPurityTot",fLambdaPurityHistogramCollection);
  fK0Short1PurityTot = CombineCollectionOfHistograms("fK0Short1PurityTot",fK0Short1PurityHistogramCollection);
  fAntiLambdaPurityTot = CombineCollectionOfHistograms("fAntiLambdaPurityTot",fAntiLambdaPurityHistogramCollection);
  fK0Short2PurityTot = CombineCollectionOfHistograms("fK0Short2PurityTot",fK0Short2PurityHistogramCollection);
  //----------------------------------------------------------
  cout << "****************************** TOTAL Lambda Purity ******************************" << endl;
  fLambdaPurityListTot = CalculatePurity("LambdaPurity",fLambdaPurityTot,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  cout << "****************************** TOTAL K0Short1 Purity ******************************" << endl;
  fK0Short1PurityListTot = CalculatePurity("K0Short1Purity",fK0Short1PurityTot,fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);
  cout << "****************************** TOTAL AntiLambda Purity ******************************" << endl;
  fAntiLambdaPurityListTot = CalculatePurity("AntiLambdaPurity",fAntiLambdaPurityTot,fLamBgFitLow,fLamBgFitHigh,fLamROI);
  cout << "****************************** TOTAL K0Short2 Purity ******************************" << endl;
  fK0Short2PurityListTot = CalculatePurity("K0Short2Purity",fK0Short2PurityTot,fK0Short1BgFitLow,fK0Short1BgFitHigh,fK0Short1ROI);


  cout << "_________________________Done BuildPurityCollections()_________________________" << endl << endl << endl;
}



//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg)
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


//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawFinalPurity(TCanvas *aCanvas)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TH1F *aLambdaPurity = (TH1F*)fLambdaPurityTot->Clone();
  TH1F *aK0Short1Purity = (TH1F*)fK0Short1PurityTot->Clone();
  TH1F *aAntiLambdaPurity = (TH1F*)fAntiLambdaPurityTot->Clone();
  TH1F *aK0Short2Purity = (TH1F*)fK0Short2PurityTot->Clone();

  TObjArray *aLambdaPurityList = (TObjArray*)fLambdaPurityListTot->Clone();
  TObjArray *aK0Short1PurityList = (TObjArray*)fK0Short1PurityListTot->Clone();
  TObjArray *aAntiLambdaPurityList = (TObjArray*)fAntiLambdaPurityListTot->Clone();
  TObjArray *aK0Short2PurityList = (TObjArray*)fK0Short2PurityListTot->Clone();


  aCanvas->cd();
  aCanvas->Divide(2,4);
  //-----
  aCanvas->cd(1);
  DrawPurity(aLambdaPurity,aLambdaPurityList,false);
  aCanvas->cd(2);
  DrawPurity(aLambdaPurity,aLambdaPurityList,true);
  //-----
  aCanvas->cd(3);
  DrawPurity(aK0Short1Purity,aK0Short1PurityList,false);
  aCanvas->cd(4);
  DrawPurity(aK0Short1Purity,aK0Short1PurityList,true);
  //-----
  aCanvas->cd(5);
  DrawPurity(aAntiLambdaPurity,aAntiLambdaPurityList,false);
  aCanvas->cd(6);
  DrawPurity(aAntiLambdaPurity,aAntiLambdaPurityList,true);
  //-----
  aCanvas->cd(7);
  DrawPurity(aK0Short2Purity,aK0Short2PurityList,false);
  aCanvas->cd(8);
  DrawPurity(aK0Short2Purity,aK0Short2PurityList,true);

}



//------27 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK03::GetCfCollection(TString aType, TString aDirectoryName)
{
  if(aType.EqualTo("Num") && aDirectoryName.EqualTo("LamK0")){return fNumCollection_LamK0;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("LamK0")){return fDenCollection_LamK0;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("LamK0")){return fCfCollection_LamK0;}

  else if(aType.EqualTo("Num") && aDirectoryName.EqualTo("ALamK0")){return fNumCollection_ALamK0;}
  else if(aType.EqualTo("Den") && aDirectoryName.EqualTo("ALamK0")){return fDenCollection_ALamK0;}
  else if(aType.EqualTo("Cf") && aDirectoryName.EqualTo("ALamK0")){return fCfCollection_ALamK0;}

  else
  {
    cout << "ERROR in GetCfCollection:  No collection to return!!!!!" << endl;
    return 0;
  }

}


//-----29 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
//________________________________________________________________________________________________________________
void buildAllcLamK03::SaveAll(TFile* aFile)
{
  //TFile *aReturnFile = new TFile(aName, aOption);

  //-----KStar Cfs
  fNumCollection_LamK0->Write("fNumCollection_LamK0",TObject::kSingleKey);
  fNumCollection_ALamK0->Write("fNumCollection_ALamK0",TObject::kSingleKey);

  fDenCollection_LamK0->Write("fDenCollection_LamK0",TObject::kSingleKey);
  fDenCollection_ALamK0->Write("fDenCollection_ALamK0",TObject::kSingleKey);

  fCfCollection_LamK0->Write("fCfCollection_LamK0",TObject::kSingleKey);
  fCfCollection_ALamK0->Write("fCfCollection_ALamK0",TObject::kSingleKey);

  fCf_LamK0_Tot->Write("fCf_LamK0_Tot");
  fCf_ALamK0_Tot->Write("fCf_ALamK0_Tot");

  //-----Average Separation Cfs
    //-----PosPos
  fAvgSepNumCollection_PosPos_LamK0->Write("fAvgSepNumCollection_PosPos_LamK0",TObject::kSingleKey);
  fAvgSepNumCollection_PosPos_ALamK0->Write("fAvgSepNumCollection_PosPos_ALamK0",TObject::kSingleKey);

  fAvgSepDenCollection_PosPos_LamK0->Write("fAvgSepDenCollection_PosPos_LamK0",TObject::kSingleKey);
  fAvgSepDenCollection_PosPos_ALamK0->Write("fAvgSepDenCollection_PosPos_ALamK0",TObject::kSingleKey);

  fAvgSepCfCollection_PosPos_LamK0->Write("fAvgSepCfCollection_PosPos_LamK0",TObject::kSingleKey);
  fAvgSepCfCollection_PosPos_ALamK0->Write("fAvgSepCfCollection_PosPos_ALamK0",TObject::kSingleKey);

  fAvgSepCf_PosPos_LamK0_Tot->Write("fAvgSepCf_PosPos_LamK0_Tot");
  fAvgSepCf_PosPos_ALamK0_Tot->Write("fAvgSepCf_PosPos_ALamK0_Tot");

    //-----PosNeg
  fAvgSepNumCollection_PosNeg_LamK0->Write("fAvgSepNumCollection_PosNeg_LamK0",TObject::kSingleKey);
  fAvgSepNumCollection_PosNeg_ALamK0->Write("fAvgSepNumCollection_PosNeg_ALamK0",TObject::kSingleKey);

  fAvgSepDenCollection_PosNeg_LamK0->Write("fAvgSepDenCollection_PosNeg_LamK0",TObject::kSingleKey);
  fAvgSepDenCollection_PosNeg_ALamK0->Write("fAvgSepDenCollection_PosNeg_ALamK0",TObject::kSingleKey);

  fAvgSepCfCollection_PosNeg_LamK0->Write("fAvgSepCfCollection_PosNeg_LamK0",TObject::kSingleKey);
  fAvgSepCfCollection_PosNeg_ALamK0->Write("fAvgSepCfCollection_PosNeg_ALamK0",TObject::kSingleKey);

  fAvgSepCf_PosNeg_LamK0_Tot->Write("fAvgSepCf_PosNeg_LamK0_Tot");
  fAvgSepCf_PosNeg_ALamK0_Tot->Write("fAvgSepCf_PosNeg_ALamK0_Tot");

    //-----NegPos
  fAvgSepNumCollection_NegPos_LamK0->Write("fAvgSepNumCollection_NegPos_LamK0",TObject::kSingleKey);
  fAvgSepNumCollection_NegPos_ALamK0->Write("fAvgSepNumCollection_NegPos_ALamK0",TObject::kSingleKey);

  fAvgSepDenCollection_NegPos_LamK0->Write("fAvgSepDenCollection_NegPos_LamK0",TObject::kSingleKey);
  fAvgSepDenCollection_NegPos_ALamK0->Write("fAvgSepDenCollection_NegPos_ALamK0",TObject::kSingleKey);

  fAvgSepCfCollection_NegPos_LamK0->Write("fAvgSepCfCollection_NegPos_LamK0",TObject::kSingleKey);
  fAvgSepCfCollection_NegPos_ALamK0->Write("fAvgSepCfCollection_NegPos_ALamK0",TObject::kSingleKey);

  fAvgSepCf_NegPos_LamK0_Tot->Write("fAvgSepCf_NegPos_LamK0_Tot");
  fAvgSepCf_NegPos_ALamK0_Tot->Write("fAvgSepCf_NegPos_ALamK0_Tot");

    //-----NegNeg
  fAvgSepNumCollection_NegNeg_LamK0->Write("fAvgSepNumCollection_NegNeg_LamK0",TObject::kSingleKey);
  fAvgSepNumCollection_NegNeg_ALamK0->Write("fAvgSepNumCollection_NegNeg_ALamK0",TObject::kSingleKey);

  fAvgSepDenCollection_NegNeg_LamK0->Write("fAvgSepDenCollection_NegNeg_LamK0",TObject::kSingleKey);
  fAvgSepDenCollection_NegNeg_ALamK0->Write("fAvgSepDenCollection_NegNeg_ALamK0",TObject::kSingleKey);

  fAvgSepCfCollection_NegNeg_LamK0->Write("fAvgSepCfCollection_NegNeg_LamK0",TObject::kSingleKey);
  fAvgSepCfCollection_NegNeg_ALamK0->Write("fAvgSepCfCollection_NegNeg_ALamK0",TObject::kSingleKey);

  fAvgSepCf_NegNeg_LamK0_Tot->Write("fAvgSepCf_NegNeg_LamK0_Tot");
  fAvgSepCf_NegNeg_ALamK0_Tot->Write("fAvgSepCf_NegNeg_ALamK0_Tot");
}



//-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)

//________________________________________________________________________________________________________________
TH2F* buildAllcLamK03::Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName)
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
TObjArray* buildAllcLamK03::LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName)
{
  TObjArray* ReturnCollection = new TObjArray();

  if(aDirectoryName.EqualTo(fDirNameLamK0))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirLamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirLamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else if(aDirectoryName.EqualTo(fDirNameALamK0))
  {
    ReturnCollection->Add(Get2DHistoClone(fDirALamK0Bp1,aHistoName,aHistoName+"_Bp1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamK0Bp2,aHistoName,aHistoName+"_Bp2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamK0Bm1,aHistoName,aHistoName+"_Bm1"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamK0Bm2,aHistoName,aHistoName+"_Bm2"));
    ReturnCollection->Add(Get2DHistoClone(fDirALamK0Bm3,aHistoName,aHistoName+"_Bm3"));
  }

  else{cout << "ERROR IN LoadCollectionOf2DHistograms!!!!!!!!!!!!!!!!" << endl;}

  return ReturnCollection;

}


//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK03::BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumberOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin)
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
        TH1F* NumClone = Num->Clone();  //Projections are always TH1D for some reason.  My functions work with TH1F, not sure how they will react to TH1D, but worth checking
				        //At some point I should go back and do dynamic_cast where necessary
      TH1D* Den = ((TH2F*)a2DDenCollection->At(i))->ProjectionY("",aBin,aBin);
        TH1F* DenClone = Den->Clone();

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
void buildAllcLamK03::BuildSepCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildSepCollections()_________________________" << endl;

  //-----LamK0
  TString NumName_SepPosPos_LamK0 = "NumPosPosSepCFs_LamK0";
  TString DenName_SepPosPos_LamK0 = "DenPosPosSepCFs_LamK0";

  TString NumName_SepPosNeg_LamK0 = "NumPosNegSepCFs_LamK0";
  TString DenName_SepPosNeg_LamK0 = "DenPosNegSepCFs_LamK0";

  TString NumName_SepNegPos_LamK0 = "NumNegPosSepCFs_LamK0";
  TString DenName_SepNegPos_LamK0 = "DenNegPosSepCFs_LamK0";

  TString NumName_SepNegNeg_LamK0 = "NumNegNegSepCFs_LamK0";
  TString DenName_SepNegNeg_LamK0 = "DenNegNegSepCFs_LamK0";


  //-----ALamK0
  TString NumName_SepPosPos_ALamK0 = "NumPosPosSepCFs_ALamK0";
  TString DenName_SepPosPos_ALamK0 = "DenPosPosSepCFs_ALamK0";

  TString NumName_SepPosNeg_ALamK0 = "NumPosNegSepCFs_ALamK0";
  TString DenName_SepPosNeg_ALamK0 = "DenPosNegSepCFs_ALamK0";

  TString NumName_SepNegPos_ALamK0 = "NumNegPosSepCFs_ALamK0";
  TString DenName_SepNegPos_ALamK0 = "DenNegPosSepCFs_ALamK0";

  TString NumName_SepNegNeg_ALamK0 = "NumNegNegSepCFs_ALamK0";
  TString DenName_SepNegNeg_ALamK0 = "DenNegNegSepCFs_ALamK0";


  //---------Adding Centrality tag if necessary---------------------------
  TString CentralityTag;
  if(fDirNameLamK0.Contains("_0010")) {CentralityTag = "_0010";}
  else if(fDirNameLamK0.Contains("_1030")) {CentralityTag = "_1030";}
  else if(fDirNameLamK0.Contains("_3050")) {CentralityTag = "_3050";}
  else{CentralityTag = "";}

  NumName_SepPosPos_LamK0 += CentralityTag;
  DenName_SepPosPos_LamK0 += CentralityTag;
  NumName_SepPosNeg_LamK0 += CentralityTag;
  DenName_SepPosNeg_LamK0 += CentralityTag;
  NumName_SepNegPos_LamK0 += CentralityTag;
  DenName_SepNegPos_LamK0 += CentralityTag;
  NumName_SepNegNeg_LamK0 += CentralityTag;
  DenName_SepNegNeg_LamK0 += CentralityTag;

  NumName_SepPosPos_ALamK0 += CentralityTag;
  DenName_SepPosPos_ALamK0 += CentralityTag;
  NumName_SepPosNeg_ALamK0 += CentralityTag;
  DenName_SepPosNeg_ALamK0 += CentralityTag;
  NumName_SepNegPos_ALamK0 += CentralityTag;
  DenName_SepNegPos_ALamK0 += CentralityTag;
  NumName_SepNegNeg_ALamK0 += CentralityTag;
  DenName_SepNegNeg_ALamK0 += CentralityTag;

  int NumXbins = 8;

  //----------------------------------------------------------------------

  //-----For now, simply set fMinNormBinSepCF = fMinNormBinAvgSepCF && fMaxNormBinSepCF = fMaxNormBinAvgSepCF
  cout << "fMinNormBinSepCF (default constructor) = " << fMinNormBinSepCF << endl;
  cout << "fMaxNormBinSepCF (default constructor) = " << fMaxNormBinSepCF << endl;
  fMinNormBinSepCF = fMinNormBinAvgSepCF;
  fMaxNormBinSepCF = fMaxNormBinAvgSepCF;
  cout << "fMinNormBinSepCF (set) = " << fMinNormBinSepCF << endl;
  cout << "fMaxNormBinSepCF (set) = " << fMaxNormBinSepCF << endl << endl;

  //--LamK0
  f2DSepNumCollection_PosPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_SepPosPos_LamK0);
  f2DSepDenCollection_PosPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_SepPosPos_LamK0);
  f1DSepCfCollection_PosPos_LamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_PosPos_LamK0",NumXbins,f2DSepNumCollection_PosPos_LamK0,f2DSepDenCollection_PosPos_LamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_PosNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_SepPosNeg_LamK0);
  f2DSepDenCollection_PosNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_SepPosNeg_LamK0);
  f1DSepCfCollection_PosNeg_LamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_PosNeg_LamK0",NumXbins,f2DSepNumCollection_PosNeg_LamK0,f2DSepDenCollection_PosNeg_LamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_NegPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_SepNegPos_LamK0);
  f2DSepDenCollection_NegPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_SepNegPos_LamK0);
  f1DSepCfCollection_NegPos_LamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_NegPos_LamK0",NumXbins,f2DSepNumCollection_NegPos_LamK0,f2DSepDenCollection_NegPos_LamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_NegNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_SepNegNeg_LamK0);
  f2DSepDenCollection_NegNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_SepNegNeg_LamK0);
  f1DSepCfCollection_NegNeg_LamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_NegNeg_LamK0",NumXbins,f2DSepNumCollection_NegNeg_LamK0,f2DSepDenCollection_NegNeg_LamK0,fMinNormBinSepCF,fMaxNormBinSepCF);


  //--ALamK0
  f2DSepNumCollection_PosPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_SepPosPos_ALamK0);
  f2DSepDenCollection_PosPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_SepPosPos_ALamK0);
  f1DSepCfCollection_PosPos_ALamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_PosPos_ALamK0",NumXbins,f2DSepNumCollection_PosPos_ALamK0,f2DSepDenCollection_PosPos_ALamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_PosNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_SepPosNeg_ALamK0);
  f2DSepDenCollection_PosNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_SepPosNeg_ALamK0);
  f1DSepCfCollection_PosNeg_ALamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_PosNeg_ALamK0",NumXbins,f2DSepNumCollection_PosNeg_ALamK0,f2DSepDenCollection_PosNeg_ALamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_NegPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_SepNegPos_ALamK0);
  f2DSepDenCollection_NegPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_SepNegPos_ALamK0);
  f1DSepCfCollection_NegPos_ALamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_NegPos_ALamK0",NumXbins,f2DSepNumCollection_NegPos_ALamK0,f2DSepDenCollection_NegPos_ALamK0,fMinNormBinSepCF,fMaxNormBinSepCF);

  f2DSepNumCollection_NegNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_SepNegNeg_ALamK0);
  f2DSepDenCollection_NegNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_SepNegNeg_ALamK0);
  f1DSepCfCollection_NegNeg_ALamK0 = BuildSepCfs(aRebinFactor,"f1DSepCf_NegNeg_ALamK0",NumXbins,f2DSepNumCollection_NegNeg_ALamK0,f2DSepDenCollection_NegNeg_ALamK0,fMinNormBinSepCF,fMaxNormBinSepCF);


  cout << "_________________________Done BuildSepCollections()_________________________" << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DSepCfCollection_PosPos_LamK0 = (TObjArray*)f1DSepCfCollection_PosPos_LamK0->Clone();
  TObjArray *a1DSepCfCollection_PosNeg_LamK0 = (TObjArray*)f1DSepCfCollection_PosNeg_LamK0->Clone();
  TObjArray *a1DSepCfCollection_NegPos_LamK0 = (TObjArray*)f1DSepCfCollection_NegPos_LamK0->Clone();
  TObjArray *a1DSepCfCollection_NegNeg_LamK0 = (TObjArray*)f1DSepCfCollection_NegNeg_LamK0->Clone();

  TObjArray *a1DSepCfCollection_PosPos_ALamK0 = (TObjArray*)f1DSepCfCollection_PosPos_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_PosNeg_ALamK0 = (TObjArray*)f1DSepCfCollection_PosNeg_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_NegPos_ALamK0 = (TObjArray*)f1DSepCfCollection_NegPos_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_NegNeg_ALamK0 = (TObjArray*)f1DSepCfCollection_NegNeg_ALamK0->Clone();


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = 0.8;
  double YRangeMax = 2.0;

  int NumXbins = 8;

  //------------------------------------------------------------------
  //-------LamK0LikeSigns------------
  aCanvasLamK0LikeSigns->cd();
  aCanvasLamK0LikeSigns->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasLamK0LikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_PosPos_LamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
/*
    TString bin1Content = "";
      bin1Content += aHistoToDraw->GetBinContent(1);
    cout << "BIN1 CONTENT = " << bin1Content << endl;
*/
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasLamK0LikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_NegNeg_LamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------LamK0UnlikeSigns------------
  aCanvasLamK0UnlikeSigns->cd();
  aCanvasLamK0UnlikeSigns->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasLamK0UnlikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_PosNeg_LamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("p(#Lambda) - #pi^{-}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasLamK0UnlikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_NegPos_LamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{-}(#Lambda) - #pi^{+}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------ALamK0LikeSigns------------
  aCanvasALamK0LikeSigns->cd();
  aCanvasALamK0LikeSigns->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasALamK0LikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_PosPos_ALamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasALamK0LikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_NegNeg_ALamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  //-------ALamK0UnlikeSigns------------
  aCanvasALamK0UnlikeSigns->cd();
  aCanvasALamK0UnlikeSigns->Divide(8,2,0,0);

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1;
    aCanvasALamK0UnlikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_PosNeg_ALamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{-}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

  for(int i=0; i<NumXbins; i++)
  {
    int aCanvasInt = i+1+8;
    aCanvasALamK0UnlikeSigns->cd(aCanvasInt);

    TH1F* aHistoToDraw = (TH1F*)a1DSepCfCollection_NegPos_ALamK0->At(i);
      aHistoToDraw->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
      aHistoToDraw->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{+}(K^{0})");
      aHistoToDraw->SetMarkerStyle(20);
      aHistoToDraw->SetMarkerSize(0.5);
    aHistoToDraw->Draw();
    line->Draw();
  }

}


//________________________________________________________________________________________________________________
void buildAllcLamK03::DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DSepCfCollection_PosPos_LamK0 = (TObjArray*)f1DSepCfCollection_PosPos_LamK0->Clone();
  TObjArray *a1DSepCfCollection_PosNeg_LamK0 = (TObjArray*)f1DSepCfCollection_PosNeg_LamK0->Clone();
  TObjArray *a1DSepCfCollection_NegPos_LamK0 = (TObjArray*)f1DSepCfCollection_NegPos_LamK0->Clone();
  TObjArray *a1DSepCfCollection_NegNeg_LamK0 = (TObjArray*)f1DSepCfCollection_NegNeg_LamK0->Clone();

  TObjArray *a1DSepCfCollection_PosPos_ALamK0 = (TObjArray*)f1DSepCfCollection_PosPos_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_PosNeg_ALamK0 = (TObjArray*)f1DSepCfCollection_PosNeg_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_NegPos_ALamK0 = (TObjArray*)f1DSepCfCollection_NegPos_ALamK0->Clone();
  TObjArray *a1DSepCfCollection_NegNeg_ALamK0 = (TObjArray*)f1DSepCfCollection_NegNeg_ALamK0->Clone();

  //---------------------------------------------------------------------------------------------------------
  TH1F* aAvgSepCf_PosPos_LamK0 = (TH1F*)a1DSepCfCollection_PosPos_LamK0->At(0);
    aAvgSepCf_PosPos_LamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_PosNeg_LamK0 = (TH1F*)a1DSepCfCollection_PosNeg_LamK0->At(0);
    aAvgSepCf_PosNeg_LamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_NegPos_LamK0 = (TH1F*)a1DSepCfCollection_NegPos_LamK0->At(0);
    aAvgSepCf_NegPos_LamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_NegNeg_LamK0 = (TH1F*)a1DSepCfCollection_NegNeg_LamK0->At(0);
    aAvgSepCf_NegNeg_LamK0->SetBit(TH1::kIsAverage);

  TH1F* aAvgSepCf_PosPos_ALamK0 = (TH1F*)a1DSepCfCollection_PosPos_ALamK0->At(0);
    aAvgSepCf_PosPos_ALamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_PosNeg_ALamK0 = (TH1F*)a1DSepCfCollection_PosNeg_ALamK0->At(0);
    aAvgSepCf_PosNeg_ALamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_NegPos_ALamK0 = (TH1F*)a1DSepCfCollection_NegPos_ALamK0->At(0);
    aAvgSepCf_NegPos_ALamK0->SetBit(TH1::kIsAverage);
  TH1F* aAvgSepCf_NegNeg_ALamK0 = (TH1F*)a1DSepCfCollection_NegNeg_ALamK0->At(0);
    aAvgSepCf_NegNeg_ALamK0->SetBit(TH1::kIsAverage);


  //---------------------------------------------------------------------------------------------------------
  int NumXbins = 8;
  for(int i=1; i<NumXbins; i++)
  {
    TH1F* tempSepCf_PosPos_LamK0 = ((TH1F*)a1DSepCfCollection_PosPos_LamK0->At(i))->Clone();
      tempSepCf_PosPos_LamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_PosPos_LamK0->Add(tempSepCf_PosPos_LamK0);
    TH1F* tempSepCf_PosNeg_LamK0 = ((TH1F*)a1DSepCfCollection_PosNeg_LamK0->At(i))->Clone();
      tempSepCf_PosNeg_LamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_PosNeg_LamK0->Add(tempSepCf_PosNeg_LamK0);
    TH1F* tempSepCf_NegPos_LamK0 = ((TH1F*)a1DSepCfCollection_NegPos_LamK0->At(i))->Clone();
      tempSepCf_NegPos_LamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_NegPos_LamK0->Add(tempSepCf_NegPos_LamK0);
    TH1F* tempSepCf_NegNeg_LamK0 = ((TH1F*)a1DSepCfCollection_NegNeg_LamK0->At(i))->Clone();
      tempSepCf_NegNeg_LamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_NegNeg_LamK0->Add(tempSepCf_NegNeg_LamK0);

    TH1F* tempSepCf_PosPos_ALamK0 = ((TH1F*)a1DSepCfCollection_PosPos_ALamK0->At(i))->Clone();
      tempSepCf_PosPos_ALamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_PosPos_ALamK0->Add(tempSepCf_PosPos_ALamK0);
    TH1F* tempSepCf_PosNeg_ALamK0 = ((TH1F*)a1DSepCfCollection_PosNeg_ALamK0->At(i))->Clone();
      tempSepCf_PosNeg_ALamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_PosNeg_ALamK0->Add(tempSepCf_PosNeg_ALamK0);
    TH1F* tempSepCf_NegPos_ALamK0 = ((TH1F*)a1DSepCfCollection_NegPos_ALamK0->At(i))->Clone();
      tempSepCf_NegPos_ALamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_NegPos_ALamK0->Add(tempSepCf_NegPos_ALamK0);
    TH1F* tempSepCf_NegNeg_ALamK0 = ((TH1F*)a1DSepCfCollection_NegNeg_ALamK0->At(i))->Clone();
      tempSepCf_NegNeg_ALamK0->SetBit(TH1::kIsAverage);
      aAvgSepCf_NegNeg_ALamK0->Add(tempSepCf_NegNeg_ALamK0);

  }


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = 0.8;
  double YRangeMax = 1.2;

  //------------------------------------------------------------------
  //-------LamK0LikeSigns------------
  aCanvasLamK0LikeSigns->cd();
  aCanvasLamK0LikeSigns->Divide(1,2);

  aCanvasLamK0LikeSigns->cd(1);
  aAvgSepCf_PosPos_LamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosPos_LamK0->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_LamK0->Draw();
  line->Draw();

  aCanvasLamK0LikeSigns->cd(2);
  aAvgSepCf_NegNeg_LamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegNeg_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_LamK0->Draw();
  line->Draw();

  //-------LamK0UnlikeSigns------------
  aCanvasLamK0UnlikeSigns->cd();
  aCanvasLamK0UnlikeSigns->Divide(1,2);

  aCanvasLamK0UnlikeSigns->cd(1);
  aAvgSepCf_PosNeg_LamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosNeg_LamK0->SetTitle("p(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_LamK0->Draw();
  line->Draw();

  aCanvasLamK0UnlikeSigns->cd(2);
  aAvgSepCf_NegPos_LamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegPos_LamK0->SetTitle("#pi^{-}(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_LamK0->Draw();
  line->Draw();

  //-------ALamK0LikeSigns------------
  aCanvasALamK0LikeSigns->cd();
  aCanvasALamK0LikeSigns->Divide(1,2);

  aCanvasALamK0LikeSigns->cd(1);
  aAvgSepCf_PosPos_ALamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosPos_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0LikeSigns->cd(2);
  aAvgSepCf_NegNeg_ALamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegNeg_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_ALamK0->Draw();
  line->Draw();

  //-------ALamK0UnlikeSigns------------
  aCanvasALamK0UnlikeSigns->cd();
  aCanvasALamK0UnlikeSigns->Divide(1,2);

  aCanvasALamK0UnlikeSigns->cd(1);
  aAvgSepCf_PosNeg_ALamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosNeg_ALamK0->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_ALamK0->Draw();
  line->Draw();

  aCanvasALamK0UnlikeSigns->cd(2);
  aAvgSepCf_NegPos_ALamK0->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegPos_ALamK0->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_ALamK0->Draw();
  line->Draw();


}

//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
