///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamK0                                                        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "buildAllcLamK0.h"

#ifdef __ROOT__
ClassImp(buildAllcLamK0)
#endif

//________________________________________________________________________________________________________________
buildAllcLamK0::buildAllcLamK0(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0):
  buildAll(aVectorOfFileNames),

  //General stuff----------------
  fDirNameLamK0(aDirNameLamK0),
  fDirNameALamK0(aDirNameALamK0)

{
  SetAnalysisDirectories();
}


//________________________________________________________________________________________________________________
buildAllcLamK0::~buildAllcLamK0()
{
  cout << "Object is being deleted" << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamK0::SetAnalysisDirectories()
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
TObjArray* buildAllcLamK0::GetAnalysisDirectory(TString aFile, TString aDirectoryName)
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
TObjArray* buildAllcLamK0::LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName)
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
void buildAllcLamK0::BuildCFCollections()
{
  cout << "_________________________Beginning BuildCFCollections()_________________________" << endl;

  TString NumName_LamK0 = "NumKStarCf_LamK0";
  TString DenName_LamK0 = "DenKStarCf_LamK0";

  TString NumName_ALamK0 = "NumKStarCf_ALamK0";
  TString DenName_ALamK0 = "DenKStarCf_ALamK0";

  cout << "fMinNormBinCF (default constructor) = " << fMinNormBinCF << endl;
  cout << "fMaxNormBinCF (default constructor) = " << fMaxNormBinCF << endl;

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
void buildAllcLamK0::DrawFinalCFs(TCanvas *aCanvas)
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
void buildAllcLamK0::BuildAvgSepCollections()
{
  cout << "_________________________Beginning BuildAvgSepCollections()_________________________" << endl;

  TString NumName_AvgSepPosPos_LamK0 = "NumPosPosAvgSepCf_LamK0";
  TString DenName_AvgSepPosPos_LamK0 = "DenPosPosAvgSepCf_LamK0";
  TString NumName_AvgSepPosNeg_LamK0 = "NumPosNegAvgSepCf_LamK0";
  TString DenName_AvgSepPosNeg_LamK0 = "DenPosNegAvgSepCf_LamK0";
  TString NumName_AvgSepNegPos_LamK0 = "NumNegPosAvgSepCf_LamK0";
  TString DenName_AvgSepNegPos_LamK0 = "DenNegPosAvgSepCf_LamK0";
  TString NumName_AvgSepNegNeg_LamK0 = "NumNegNegAvgSepCf_LamK0";
  TString DenName_AvgSepNegNeg_LamK0 = "DenNegNegAvgSepCf_LamK0";

  TString NumName_AvgSepPosPos_ALamK0 = "NumPosPosAvgSepCf_ALamK0";
  TString DenName_AvgSepPosPos_ALamK0 = "DenPosPosAvgSepCf_ALamK0";
  TString NumName_AvgSepPosNeg_ALamK0 = "NumPosNegAvgSepCf_ALamK0";
  TString DenName_AvgSepPosNeg_ALamK0 = "DenPosNegAvgSepCf_ALamK0";
  TString NumName_AvgSepNegPos_ALamK0 = "NumNegPosAvgSepCf_ALamK0";
  TString DenName_AvgSepNegPos_ALamK0 = "DenNegPosAvgSepCf_ALamK0";
  TString NumName_AvgSepNegNeg_ALamK0 = "NumNegNegAvgSepCf_ALamK0";
  TString DenName_AvgSepNegNeg_ALamK0 = "DenNegNegAvgSepCf_ALamK0";

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
void buildAllcLamK0::DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0)
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
void buildAllcLamK0::BuildPurityCollections()
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
void buildAllcLamK0::DrawFinalPurity(TCanvas *aCanvas)
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



//------27 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK0::GetCfCollection(TString aType, TString aDirectoryName)
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


//-----29 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
//________________________________________________________________________________________________________________
void buildAllcLamK0::SaveAll(TFile* aFile)
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



//-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
//________________________________________________________________________________________________________________
TObjArray* buildAllcLamK0::LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName)
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
void buildAllcLamK0::BuildSepCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildSepCollections()_________________________" << endl;

  //-----LamK0
  TString NumName_SepPosPos_LamK0 = "NumPosPosSepCfs_LamK0";
  TString DenName_SepPosPos_LamK0 = "DenPosPosSepCfs_LamK0";

  TString NumName_SepPosNeg_LamK0 = "NumPosNegSepCfs_LamK0";
  TString DenName_SepPosNeg_LamK0 = "DenPosNegSepCfs_LamK0";

  TString NumName_SepNegPos_LamK0 = "NumNegPosSepCfs_LamK0";
  TString DenName_SepNegPos_LamK0 = "DenNegPosSepCfs_LamK0";

  TString NumName_SepNegNeg_LamK0 = "NumNegNegSepCfs_LamK0";
  TString DenName_SepNegNeg_LamK0 = "DenNegNegSepCfs_LamK0";


  //-----ALamK0
  TString NumName_SepPosPos_ALamK0 = "NumPosPosSepCfs_ALamK0";
  TString DenName_SepPosPos_ALamK0 = "DenPosPosSepCfs_ALamK0";

  TString NumName_SepPosNeg_ALamK0 = "NumPosNegSepCfs_ALamK0";
  TString DenName_SepPosNeg_ALamK0 = "DenPosNegSepCfs_ALamK0";

  TString NumName_SepNegPos_ALamK0 = "NumNegPosSepCfs_ALamK0";
  TString DenName_SepNegPos_ALamK0 = "DenNegPosSepCfs_ALamK0";

  TString NumName_SepNegNeg_ALamK0 = "NumNegNegSepCfs_ALamK0";
  TString DenName_SepNegNeg_ALamK0 = "DenNegNegSepCfs_ALamK0";


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


  cout << "_________________________Done BuildSepCollections()_________________________" << endl << endl << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamK0::DrawFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns)
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
void buildAllcLamK0::DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns)
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
void buildAllcLamK0::BuildCowCollections(int aRebinFactor)
{
  cout << "_________________________Beginning BuildCowCollections()_________________________" << endl;

  //-----LamK0
  TString NumName_CowPosPos_LamK0 = "NumPosPosAvgSepCfCowboysAndSailors_LamK0";
  TString DenName_CowPosPos_LamK0 = "DenPosPosAvgSepCfCowboysAndSailors_LamK0";

  TString NumName_CowPosNeg_LamK0 = "NumPosNegAvgSepCfCowboysAndSailors_LamK0";
  TString DenName_CowPosNeg_LamK0 = "DenPosNegAvgSepCfCowboysAndSailors_LamK0";

  TString NumName_CowNegPos_LamK0 = "NumNegPosAvgSepCfCowboysAndSailors_LamK0";
  TString DenName_CowNegPos_LamK0 = "DenNegPosAvgSepCfCowboysAndSailors_LamK0";

  TString NumName_CowNegNeg_LamK0 = "NumNegNegAvgSepCfCowboysAndSailors_LamK0";
  TString DenName_CowNegNeg_LamK0 = "DenNegNegAvgSepCfCowboysAndSailors_LamK0";


  //-----ALamK0
  TString NumName_CowPosPos_ALamK0 = "NumPosPosAvgSepCfCowboysAndSailors_ALamK0";
  TString DenName_CowPosPos_ALamK0 = "DenPosPosAvgSepCfCowboysAndSailors_ALamK0";

  TString NumName_CowPosNeg_ALamK0 = "NumPosNegAvgSepCfCowboysAndSailors_ALamK0";
  TString DenName_CowPosNeg_ALamK0 = "DenPosNegAvgSepCfCowboysAndSailors_ALamK0";

  TString NumName_CowNegPos_ALamK0 = "NumNegPosAvgSepCfCowboysAndSailors_ALamK0";
  TString DenName_CowNegPos_ALamK0 = "DenNegPosAvgSepCfCowboysAndSailors_ALamK0";

  TString NumName_CowNegNeg_ALamK0 = "NumNegNegAvgSepCfCowboysAndSailors_ALamK0";
  TString DenName_CowNegNeg_ALamK0 = "DenNegNegAvgSepCfCowboysAndSailors_ALamK0";


  //----------------------------------------------------------------------

  //--LamK0
  f2DCowNumCollection_PosPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_CowPosPos_LamK0);
  f2DCowDenCollection_PosPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_CowPosPos_LamK0);
  f1DCowCfCollection_PosPos_LamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_PosPos_LamK0",f2DCowNumCollection_PosPos_LamK0,f2DCowDenCollection_PosPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_PosNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_CowPosNeg_LamK0);
  f2DCowDenCollection_PosNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_CowPosNeg_LamK0);
  f1DCowCfCollection_PosNeg_LamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_PosNeg_LamK0",f2DCowNumCollection_PosNeg_LamK0,f2DCowDenCollection_PosNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_NegPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_CowNegPos_LamK0);
  f2DCowDenCollection_NegPos_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_CowNegPos_LamK0);
  f1DCowCfCollection_NegPos_LamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_NegPos_LamK0",f2DCowNumCollection_NegPos_LamK0,f2DCowDenCollection_NegPos_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_NegNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,NumName_CowNegNeg_LamK0);
  f2DCowDenCollection_NegNeg_LamK0 = LoadCollectionOf2DHistograms(fDirNameLamK0,DenName_CowNegNeg_LamK0);
  f1DCowCfCollection_NegNeg_LamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_NegNeg_LamK0",f2DCowNumCollection_NegNeg_LamK0,f2DCowDenCollection_NegNeg_LamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  //--ALamK0
  f2DCowNumCollection_PosPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_CowPosPos_ALamK0);
  f2DCowDenCollection_PosPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_CowPosPos_ALamK0);
  f1DCowCfCollection_PosPos_ALamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_PosPos_ALamK0",f2DCowNumCollection_PosPos_ALamK0,f2DCowDenCollection_PosPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_PosNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_CowPosNeg_ALamK0);
  f2DCowDenCollection_PosNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_CowPosNeg_ALamK0);
  f1DCowCfCollection_PosNeg_ALamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_PosNeg_ALamK0",f2DCowNumCollection_PosNeg_ALamK0,f2DCowDenCollection_PosNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_NegPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_CowNegPos_ALamK0);
  f2DCowDenCollection_NegPos_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_CowNegPos_ALamK0);
  f1DCowCfCollection_NegPos_ALamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_NegPos_ALamK0",f2DCowNumCollection_NegPos_ALamK0,f2DCowDenCollection_NegPos_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);

  f2DCowNumCollection_NegNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,NumName_CowNegNeg_ALamK0);
  f2DCowDenCollection_NegNeg_ALamK0 = LoadCollectionOf2DHistograms(fDirNameALamK0,DenName_CowNegNeg_ALamK0);
  f1DCowCfCollection_NegNeg_ALamK0 = BuildCowCfs(aRebinFactor,"f1DCowCf_NegNeg_ALamK0",f2DCowNumCollection_NegNeg_ALamK0,f2DCowDenCollection_NegNeg_ALamK0,fMinNormBinAvgSepCF,fMaxNormBinAvgSepCF);


  cout << "_________________________Done BuildCowCollections()_________________________" << endl << endl << endl;
}


//________________________________________________________________________________________________________________
void buildAllcLamK0::DrawFinalCowCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0)
{
  //I make Clones of the origins and give them generic names
  //This will simplify things in the future if I wish to instead pass histograms to this method
  TObjArray *a1DCowCfCollection_PosPos_LamK0 = (TObjArray*)f1DCowCfCollection_PosPos_LamK0->Clone();
  TObjArray *a1DCowCfCollection_PosNeg_LamK0 = (TObjArray*)f1DCowCfCollection_PosNeg_LamK0->Clone();
  TObjArray *a1DCowCfCollection_NegPos_LamK0 = (TObjArray*)f1DCowCfCollection_NegPos_LamK0->Clone();
  TObjArray *a1DCowCfCollection_NegNeg_LamK0 = (TObjArray*)f1DCowCfCollection_NegNeg_LamK0->Clone();

  TObjArray *a1DCowCfCollection_PosPos_ALamK0 = (TObjArray*)f1DCowCfCollection_PosPos_ALamK0->Clone();
  TObjArray *a1DCowCfCollection_PosNeg_ALamK0 = (TObjArray*)f1DCowCfCollection_PosNeg_ALamK0->Clone();
  TObjArray *a1DCowCfCollection_NegPos_ALamK0 = (TObjArray*)f1DCowCfCollection_NegPos_ALamK0->Clone();
  TObjArray *a1DCowCfCollection_NegNeg_ALamK0 = (TObjArray*)f1DCowCfCollection_NegNeg_ALamK0->Clone();


  //--A horizontal line at y=1 to help guide the eye
  TLine *line = new TLine(0,1,20,1);
  line->SetLineColor(14);

  double YRangeMin = -0.5;
  double YRangeMax = 5.0;

  double aMarkerSize = 0.5;
  double aMarkerStyle = 20;

  //----------------------------------------------------------------

  aCanvasLamK0->cd();
  aCanvasLamK0->Divide(2,2);

  aCanvasLamK0->cd(1);
  TH1D* aAvgSepCf_PosPos_LamK01 = (TH1D*)a1DCowCfCollection_PosPos_LamK0->At(0);
    aAvgSepCf_PosPos_LamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosPos_LamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosPos_LamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_PosPos_LamK02 = (TH1D*)a1DCowCfCollection_PosPos_LamK0->At(1);
    aAvgSepCf_PosPos_LamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosPos_LamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosPos_LamK02->SetMarkerColor(2);
  aAvgSepCf_PosPos_LamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosPos_LamK01->SetTitle("p(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_LamK01->Draw();
  aAvgSepCf_PosPos_LamK02->Draw("same");
  line->Draw();
  TLegend *legLamK01 = new TLegend(0.7,0.7,0.9,0.9);
    legLamK01->AddEntry(aAvgSepCf_PosPos_LamK01, "P(#pi^{+}(K^{0})) > P(p(#Lambda))", "p");
    legLamK01->AddEntry(aAvgSepCf_PosPos_LamK02, "P(#pi^{+}(K^{0})) < P(p(#Lambda))", "p");
    legLamK01->Draw();

  aCanvasLamK0->cd(2);
  TH1D* aAvgSepCf_PosNeg_LamK01 = (TH1D*)a1DCowCfCollection_PosNeg_LamK0->At(0);
    aAvgSepCf_PosNeg_LamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosNeg_LamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosNeg_LamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_PosNeg_LamK02 = (TH1D*)a1DCowCfCollection_PosNeg_LamK0->At(1);
    aAvgSepCf_PosNeg_LamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosNeg_LamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosNeg_LamK02->SetMarkerColor(2);
  aAvgSepCf_PosNeg_LamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosNeg_LamK01->SetTitle("p(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_LamK01->Draw();
  aAvgSepCf_PosNeg_LamK02->Draw("same");
  line->Draw();
  TLegend *legLamK02 = new TLegend(0.7,0.7,0.9,0.9);
    legLamK02->AddEntry(aAvgSepCf_PosNeg_LamK01, "P(#pi^{-}(K^{0})) > P(p(#Lambda))", "p");
    legLamK02->AddEntry(aAvgSepCf_PosNeg_LamK02, "P(#pi^{-}(K^{0})) < P(p(#Lambda))", "p");
    legLamK02->Draw();

  aCanvasLamK0->cd(3);
  TH1D* aAvgSepCf_NegPos_LamK01 = (TH1D*)a1DCowCfCollection_NegPos_LamK0->At(0);
    aAvgSepCf_NegPos_LamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegPos_LamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegPos_LamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_NegPos_LamK02 = (TH1D*)a1DCowCfCollection_NegPos_LamK0->At(1);
    aAvgSepCf_NegPos_LamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegPos_LamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegPos_LamK02->SetMarkerColor(2);
  aAvgSepCf_NegPos_LamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegPos_LamK01->SetTitle("#pi^{-}(#Lambda) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_LamK01->Draw();
  aAvgSepCf_NegPos_LamK02->Draw("same");
  line->Draw();
  TLegend *legLamK03 = new TLegend(0.7,0.7,0.9,0.9);
    legLamK03->AddEntry(aAvgSepCf_NegPos_LamK01, "P(#pi^{+}(K^{0})) > P(#pi^{-}(#Lambda))", "p");
    legLamK03->AddEntry(aAvgSepCf_NegPos_LamK02, "P(#pi^{+}(K^{0})) < P(#pi^{-}(#Lambda))", "p");
    legLamK03->Draw();

  aCanvasLamK0->cd(4);
  TH1D* aAvgSepCf_NegNeg_LamK01 = (TH1D*)a1DCowCfCollection_NegNeg_LamK0->At(0);
    aAvgSepCf_NegNeg_LamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegNeg_LamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegNeg_LamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_NegNeg_LamK02 = (TH1D*)a1DCowCfCollection_NegNeg_LamK0->At(1);
    aAvgSepCf_NegNeg_LamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegNeg_LamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegNeg_LamK02->SetMarkerColor(2);
  aAvgSepCf_NegNeg_LamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegNeg_LamK01->SetTitle("#pi^{-}(#Lambda) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_LamK01->Draw();
  aAvgSepCf_NegNeg_LamK02->Draw("same");
  line->Draw();
  TLegend *legLamK04 = new TLegend(0.7,0.7,0.9,0.9);
    legLamK04->AddEntry(aAvgSepCf_NegNeg_LamK01, "P(#pi^{-}(K^{0})) > P(#pi^{-}(#Lambda))", "p");
    legLamK04->AddEntry(aAvgSepCf_NegNeg_LamK02, "P(#pi^{-}(K^{0})) < P(#pi^{-}(#Lambda))", "p");
    legLamK04->Draw();




  aCanvasALamK0->cd();
  aCanvasALamK0->Divide(2,2);

  aCanvasALamK0->cd(1);
  TH1D* aAvgSepCf_PosPos_ALamK01 = (TH1D*)a1DCowCfCollection_PosPos_ALamK0->At(0);
    aAvgSepCf_PosPos_ALamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosPos_ALamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosPos_ALamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_PosPos_ALamK02 = (TH1D*)a1DCowCfCollection_PosPos_ALamK0->At(1);
    aAvgSepCf_PosPos_ALamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosPos_ALamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosPos_ALamK02->SetMarkerColor(2);
  aAvgSepCf_PosPos_ALamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosPos_ALamK01->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_PosPos_ALamK01->Draw();
  aAvgSepCf_PosPos_ALamK02->Draw("same");
  line->Draw();
  TLegend *legALamK01 = new TLegend(0.7,0.7,0.9,0.9);
    legALamK01->AddEntry(aAvgSepCf_PosPos_ALamK01, "P(#pi^{+}(K^{0})) > P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamK01->AddEntry(aAvgSepCf_PosPos_ALamK02, "P(#pi^{+}(K^{0})) < P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamK01->Draw();

  aCanvasALamK0->cd(2);
  TH1D* aAvgSepCf_PosNeg_ALamK01 = (TH1D*)a1DCowCfCollection_PosNeg_ALamK0->At(0);
    aAvgSepCf_PosNeg_ALamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosNeg_ALamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosNeg_ALamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_PosNeg_ALamK02 = (TH1D*)a1DCowCfCollection_PosNeg_ALamK0->At(1);
    aAvgSepCf_PosNeg_ALamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_PosNeg_ALamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_PosNeg_ALamK02->SetMarkerColor(2);
  aAvgSepCf_PosNeg_ALamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_PosNeg_ALamK01->SetTitle("#pi^{+}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_PosNeg_ALamK01->Draw();
  aAvgSepCf_PosNeg_ALamK02->Draw("same");
  line->Draw();
  TLegend *legALamK02 = new TLegend(0.7,0.7,0.9,0.9);
    legALamK02->AddEntry(aAvgSepCf_PosNeg_ALamK01, "P(#pi^{-}(K^{0})) > P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamK02->AddEntry(aAvgSepCf_PosNeg_ALamK02, "P(#pi^{-}(K^{0})) < P(#pi^{+}(#bar{#Lambda}))", "p");
    legALamK02->Draw();

  aCanvasALamK0->cd(3);
  TH1D* aAvgSepCf_NegPos_ALamK01 = (TH1D*)a1DCowCfCollection_NegPos_ALamK0->At(0);
    aAvgSepCf_NegPos_ALamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegPos_ALamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegPos_ALamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_NegPos_ALamK02 = (TH1D*)a1DCowCfCollection_NegPos_ALamK0->At(1);
    aAvgSepCf_NegPos_ALamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegPos_ALamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegPos_ALamK02->SetMarkerColor(2);
  aAvgSepCf_NegPos_ALamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegPos_ALamK01->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{+}(K^{0})");
  aAvgSepCf_NegPos_ALamK01->Draw();
  aAvgSepCf_NegPos_ALamK02->Draw("same");
  line->Draw();
  TLegend *legALamK03 = new TLegend(0.7,0.7,0.9,0.9);
    legALamK03->AddEntry(aAvgSepCf_NegPos_ALamK01, "P(#pi^{+}(K^{0})) > P(#bar{p}(#bar{#Lambda}))", "p");
    legALamK03->AddEntry(aAvgSepCf_NegPos_ALamK02, "P(#pi^{+}(K^{0})) < P(#bar{p}(#bar{#Lambda}))", "p");
    legALamK03->Draw();

  aCanvasALamK0->cd(4);
  TH1D* aAvgSepCf_NegNeg_ALamK01 = (TH1D*)a1DCowCfCollection_NegNeg_ALamK0->At(0);
    aAvgSepCf_NegNeg_ALamK01->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegNeg_ALamK01->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegNeg_ALamK01->SetMarkerColor(1);
  TH1D* aAvgSepCf_NegNeg_ALamK02 = (TH1D*)a1DCowCfCollection_NegNeg_ALamK0->At(1);
    aAvgSepCf_NegNeg_ALamK02->SetMarkerStyle(aMarkerStyle);
    aAvgSepCf_NegNeg_ALamK02->SetMarkerSize(aMarkerSize);
    aAvgSepCf_NegNeg_ALamK02->SetMarkerColor(2);
  aAvgSepCf_NegNeg_ALamK01->GetYaxis()->SetRangeUser(YRangeMin,YRangeMax);
  aAvgSepCf_NegNeg_ALamK01->SetTitle("#bar{p}(#bar{#Lambda}) - #pi^{-}(K^{0})");
  aAvgSepCf_NegNeg_ALamK01->Draw();
  aAvgSepCf_NegNeg_ALamK02->Draw("same");
  line->Draw();
  TLegend *legALamK04 = new TLegend(0.7,0.7,0.9,0.9);
    legALamK04->AddEntry(aAvgSepCf_NegNeg_ALamK01, "P(#pi^{-}(K^{0})) > P(#bar{p}(#bar{#Lambda}))", "p");
    legALamK04->AddEntry(aAvgSepCf_NegNeg_ALamK02, "P(#pi^{-}(K^{0})) < P(#bar{p}(#bar{#Lambda}))", "p");
    legALamK04->Draw();

}




//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
//________________________________________________________________________________________________________________
