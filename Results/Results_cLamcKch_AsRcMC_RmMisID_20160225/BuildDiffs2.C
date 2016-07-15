#include "/home/jesse/Analysis/K0Lam/Analyze/Types.cxx"


//________________________________________________________________________________________________________________
TObjArray* ConnectAnalysisDirectory(TString aFileLocation, TString aDirectoryName)
{
  TFile tFile(aFileLocation);
  TList *tFemtolist = (TList*)tFile.Get("femtolist");
  TObjArray *ReturnArray = (TObjArray*)tFemtolist->FindObject(aDirectoryName);

  return ReturnArray;
}

//________________________________________________________________________________________________________________
TH2* Get2dHisto(TString aFileLocation, TString aDirName, TString aHistoName, TString aNewName)
{
  TObjArray *tDir = ConnectAnalysisDirectory(aFileLocation,aDirName);

  TH2 *tHisto = (TH2*)tDir->FindObject(aHistoName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << aHistoName << endl;}
  assert(tHisto);
  //----------------------------------

  TH2 *ReturnHisto = (TH2*)tHisto->Clone(aNewName);
  ReturnHisto->SetTitle(aNewName);
  ReturnHisto->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHisto->GetSumw2N()) {ReturnHisto->Sumw2();}

  ReturnHisto->GetXaxis()->SetTitle("k*_{true}");

  ReturnHisto->GetYaxis()->SetTitle("k*_{rec}");

  return (TH2*)ReturnHisto;
}

//________________________________________________________________________________________________________________
void NormalizeByTotalEntries(TH2* aHisto)
{
  double tNorm = aHisto->GetEntries();
  aHisto->Scale(1./tNorm);
}

//________________________________________________________________________________________________________________
void MakePaveTextPretty(TPaveText* aText)
{
  aText->SetFillColor(0);
  aText->SetTextSize(0.03);
  aText->SetBorderSize(1);
}

//________________________________________________________________________________________________________________
//--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**--&&**
//________________________________________________________________________________________________________________
void BuildDiffs2()
{
  enum NormalizationType {kUnNorm=0, kNormByTotEnt=1};

//------------------TPaveTexts: Begin----------------------
  TPaveText* tTextNoRm = new TPaveText(0.15,0.75,0.25,0.85,"NDC");
    tTextNoRm->AddText("Old");
    MakePaveTextPretty(tTextNoRm);

  TPaveText* tTextRmAll = new TPaveText(0.15,0.75,0.30,0.85,"NDC");
    tTextRmAll->AddText("RmAllMisID");
    MakePaveTextPretty(tTextRmAll);

  TPaveText* tTextKchAndLamFix2 = new TPaveText(0.15,0.75,0.25,0.85,"NDC");
    tTextKchAndLamFix2->AddText("MisIDFix");
    MakePaveTextPretty(tTextKchAndLamFix2);
  //-------------------

  TPaveText* tTextNoRmVsRmAll = new TPaveText(0.15,0.75,0.45,0.85,"NDC");
    tTextNoRmVsRmAll->AddText("Old - RmAllMisID");
//    tTextNoRmVsRmAll->AddText("= All MisID");
    MakePaveTextPretty(tTextNoRmVsRmAll);

  TPaveText* tTextNoRmVsKchAndLamFix2 = new TPaveText(0.15,0.75,0.45,0.85,"NDC");
    tTextNoRmVsKchAndLamFix2->AddText("Old - MisIDFix");
//    tTextNoRmVsKchAndLamFix2->AddText("");
    MakePaveTextPretty(tTextNoRmVsKchAndLamFix2);

  TPaveText* tTextRmAllVsKchAndLamFix2 = new TPaveText(0.15,0.75,0.45,0.85,"NDC");
    tTextRmAllVsKchAndLamFix2->AddText("RmAllMisID - MisIDFix");
//    tTextRmAllVsKchAndLamFix2->AddText("");
    MakePaveTextPretty(tTextRmAllVsKchAndLamFix2);
//------------------TPaveTexts: End------------------------


  TString FileDirectory = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_RmMisID_20160225/";

  TString FileNoRm = FileDirectory + "Results_cLamcKch_AsRcMC_20160224_Bp1.root";
  TString FileRmAll = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisID_20160225_Bp1.root";
  TString FileKchAndLamFix2 = FileDirectory + "Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229_Bp1.root";

  //--------------------------------------

  AnalysisType tAnalysisType;
    tAnalysisType = kLamKchP;
    //tAnalysisType = kALamKchP;
    //tAnalysisType = kLamKchM;
    //tAnalysisType = kALamKchM;

  KStarTrueVsRecType tKStarTrueVsRecType;
    tKStarTrueVsRecType = kSame;
    //tKStarTrueVsRecType = kMixed;

  NormalizationType tNormalizationType;
    tNormalizationType = kUnNorm;
    //tNormalizationType = kNormByTotEnt;

  //--------------------------------------
  TString tAnalysisBaseTag = TString(cAnalysisBaseTags[tAnalysisType]);
  TString tCentralityTag = "_0010";

  TString tDirectoryName = tAnalysisBaseTag + tCentralityTag;

  TString tHistoName;
    if(tKStarTrueVsRecType == kSame) tHistoName = cModelKStarTrueVsRecSameBaseTag;
    else if(tKStarTrueVsRecType == kMixed) tHistoName = cModelKStarTrueVsRecMixedBaseTag;
    else cout << "\t\t\t\t Invalid tKStarTrueVsRecType selection!" << endl;
  tHistoName += tAnalysisBaseTag;
    TString tNewNameNoRm = tHistoName + "_Old";
    TString tNewNameRmAll = tHistoName + "_RmAllMisID";
    TString tNewNameKchAndLamFix2 = tHistoName + "_MisIDFix";


  //--------------------------------------
  bool bSaveFigures = true;
  TString tSaveLocation = "~/Analysis/Presentations/AliFemto/20160330/Diffs/";
    tSaveLocation += TString(cAnalysisBaseTags[tAnalysisType]);
    tSaveLocation += "/";

  if(tKStarTrueVsRecType == kSame) tSaveLocation += "Same/";
  else if(tKStarTrueVsRecType == kMixed) tSaveLocation += "Mixed/";
  else cout << "NOPE, INVALID CHOICE!!!!!!!!" << endl;

  if(tNormalizationType == kUnNorm) tSaveLocation += "UnNormalized/";
  else if(tNormalizationType == kNormByTotEnt) tSaveLocation += "NormalizedByTotalEntries/";

  //----------------------------------------

  TH2* tNoRm = Get2dHisto(FileNoRm,tDirectoryName,tHistoName,tNewNameNoRm);
  TH2* tRmAll = Get2dHisto(FileRmAll,tDirectoryName,tHistoName,tNewNameRmAll);
  TH2* tKchAndLamFix2 = Get2dHisto(FileKchAndLamFix2,tDirectoryName,tHistoName,tNewNameKchAndLamFix2);

  tNoRm->SetLabelSize(0.025,"z");
  tRmAll->SetLabelSize(0.025,"z");
  tKchAndLamFix2->SetLabelSize(0.025,"z");

  //----------------------------------------
  if(tNormalizationType == kNormByTotEnt)
  {
    NormalizeByTotalEntries(tNoRm);
    NormalizeByTotalEntries(tRmAll);
    NormalizeByTotalEntries(tKchAndLamFix2);

    double tZlow = 0.00000001;
    double tZhigh = 0.01;

    tNoRm->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tRmAll->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tKchAndLamFix2->GetZaxis()->SetRangeUser(tZlow,tZhigh);
  }
  //-----------------------------------------

  TH2* tDiffNoRmVsRmAll = (TH2*)tNoRm->Clone("tDiffNoRmVsRmAll");
    tDiffNoRmVsRmAll->Add(tRmAll,-1);

  TH2* tDiffNoRmVsKchAndLamFix2 = (TH2*)tNoRm->Clone("tDiffNoRmVsKchAndLamFix2");
    tDiffNoRmVsKchAndLamFix2->Add(tKchAndLamFix2,-1);

  TH2* tDiffRmAllVsKchAndLamFix2 = (TH2*)tRmAll->Clone("tDiffRmAllVsKchAndLamFix2");
    tDiffRmAllVsKchAndLamFix2->Add(tKchAndLamFix2,-1);

  //-----------------------------------------
  TCanvas *tCanNoRmVsRmAll = new TCanvas("tCanNoRmVsRmAll","tCanNoRmVsRmAll");
    tCanNoRmVsRmAll->Divide(2,1);

 TCanvas *tCanNoRmVsKchAndLamFix2 = new TCanvas("tCanNoRmVsKchAndLamFix2","tCanNoRmVsKchAndLamFix2");
    tCanNoRmVsKchAndLamFix2->Divide(2,1);

 TCanvas *tCanRmAllVsKchAndLamFix2 = new TCanvas("tCanRmAllVsKchAndLamFix2","tCanRmAllVsKchAndLamFix2");
    tCanRmAllVsKchAndLamFix2->Divide(2,1);

  //-------------------------------------------
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  //-------------------------------------------
  TPad* myPad1 = tCanNoRmVsRmAll->cd(1);
    myPad1->Divide(1,2);

  myPad1->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad1->cd(2);
    gPad->SetLogz();
    tRmAll->DrawCopy("colz");
    tTextRmAll->Draw();

  tCanNoRmVsRmAll->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsRmAll->DrawCopy("colz");
    tTextNoRmVsRmAll->Draw();



  //-----------------------------
  TPad* myPad9 = tCanNoRmVsKchAndLamFix2->cd(1);
    myPad9->Divide(1,2);

  myPad9->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad9->cd(2);
    gPad->SetLogz();
    tKchAndLamFix2->DrawCopy("colz");
    tTextKchAndLamFix2->Draw();

  tCanNoRmVsKchAndLamFix2->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsKchAndLamFix2->DrawCopy("colz");
    tTextNoRmVsKchAndLamFix2->Draw();

  //-----------------------------
  TPad* myPad10 = tCanRmAllVsKchAndLamFix2->cd(1);
    myPad10->Divide(1,2);

  myPad10->cd(1);
    gPad->SetLogz();
    tRmAll->DrawCopy("colz");
    tTextRmAll->Draw();

  myPad10->cd(2);
    gPad->SetLogz();
    tKchAndLamFix2->DrawCopy("colz");
    tTextKchAndLamFix2->Draw();

  tCanRmAllVsKchAndLamFix2->cd(2);
    gPad->SetLogz();
    tDiffRmAllVsKchAndLamFix2->DrawCopy("colz");
    tTextRmAllVsKchAndLamFix2->Draw();


  //-------------------------------------------
  if(bSaveFigures)
  {
    tCanNoRmVsRmAll->SaveAs(tSaveLocation+"NoRmVsRmAll.eps");
    tCanNoRmVsKchAndLamFix2->SaveAs(tSaveLocation+"NoRmVsKchAndLamFix2.eps");
    tCanRmAllVsKchAndLamFix2->SaveAs(tSaveLocation+"RmAllVsKchAndLamFix2.eps");
  }




}
