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

  return (TH2*)ReturnHisto;
}

//________________________________________________________________________________________________________________
void NormalizeByTotalEntries(TH2* aHisto)
{
  double tNorm = aHisto->GetEntries();
  aHisto->Scale(1./tNorm);
}

//________________________________________________________________________________________________________________
void NormalizeEachColumn(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  for(int i=1; i<=tNbinsX; i++)
  {
    double tScale = aHisto->Integral(i,i,1,tNbinsY);
    if(tScale > 0.)
    {
      for(int j=1; j<=tNbinsY; j++)
      {
        double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
        aHisto->SetBinContent(i,j,tNewContent);
      }
    }
  }
}

//________________________________________________________________________________________________________________
void NormalizeEachRow(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  for(int j=1; j<=tNbinsY; j++)
  {
    double tScale = aHisto->Integral(1,tNbinsX,j,j);
    if(tScale > 0.)
    {
      for(int i=1; i<=tNbinsX; i++)
      {
        double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
        aHisto->SetBinContent(i,j,tNewContent);
      }
    }
  }
}
/*
//________________________________________________________________________________________________________________
void NormalizeCentralPeak(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  int tMinBinY = 99;
  int tMaxBinY = 102;

  for(int j=tMinBinY; j<=tMaxBinY; j++)
  {
    double tScale = aHisto->Integral(1,tNbinsX,j,j);
    for(int i=1; i<=tNbinsX; i++)
    {
      double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
      aHisto->SetBinContent(i,j,tNewContent);
    }
  }
}
*/
//________________________________________________________________________________________________________________
void NormalizeCentralPeak(TH2* aHisto)
{
  int tNbinsX = aHisto->GetNbinsX();
  int tNbinsY = aHisto->GetNbinsY();

  int tMinBinY = 99;
  int tMaxBinY = 102;

  double tScale = aHisto->Integral(1,tNbinsX,tMinBinY,tMaxBinY);

  aHisto->Scale(1.0/tScale);
}



void BuildDiffs()
{
  enum NormalizationType {kUnNorm=0, kNormByTotEnt=1, kNormColumns=2, kNormRows=3, kNormCentralPeak=4};

  //------------------TPaveTexts: Begin----------------------
  TPaveText* tTextNoRm = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRm->SetFillColor(0);
    tTextNoRm->SetTextSize(0.03);
    tTextNoRm->AddText("NoRm");
    tTextNoRm->SetBorderSize(0);

  TPaveText* tTextRmAll = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmAll->SetFillColor(0);
    tTextRmAll->SetTextSize(0.03);
    tTextRmAll->AddText("RmAll");
    tTextRmAll->SetBorderSize(0);

  TPaveText* tTextRmKch = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmKch->SetFillColor(0);
    tTextRmKch->SetTextSize(0.03);
    tTextRmKch->AddText("RmKch");
    tTextRmKch->SetBorderSize(0);

  TPaveText* tTextRmLam = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmLam->SetFillColor(0);
    tTextRmLam->SetTextSize(0.03);
    tTextRmLam->AddText("RmLam");
    tTextRmLam->SetBorderSize(0);

  TPaveText* tTextKchFix = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextKchFix->SetFillColor(0);
    tTextKchFix->SetTextSize(0.03);
    tTextKchFix->AddText("KchFix");
    tTextKchFix->SetBorderSize(0);

  TPaveText* tTextKchAndLamFix2 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextKchAndLamFix2->SetFillColor(0);
    tTextKchAndLamFix2->SetTextSize(0.03);
    tTextKchAndLamFix2->AddText("KchAndLamFix2");
    tTextKchAndLamFix2->SetBorderSize(0);

  TPaveText* tTextKchAndLamFix1 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextKchAndLamFix1->SetFillColor(0);
    tTextKchAndLamFix1->SetTextSize(0.03);
    tTextKchAndLamFix1->AddText("KchAndLamFix1");
    tTextKchAndLamFix1->SetBorderSize(0);

  //-------------------

  TPaveText* tTextNoRmVsRmAll = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsRmAll->SetFillColor(0);
    tTextNoRmVsRmAll->SetTextSize(0.03);
    tTextNoRmVsRmAll->AddText("NoRm - RmAll");
    tTextNoRmVsRmAll->AddText("= All MisID");
    tTextNoRmVsRmAll->SetBorderSize(0);

  TPaveText* tTextNoRmVsRmKch = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsRmKch->SetFillColor(0);
    tTextNoRmVsRmKch->SetTextSize(0.03);
    tTextNoRmVsRmKch->AddText("NoRm - RmKch");
    tTextNoRmVsRmKch->AddText("= MisID Kch");
    tTextNoRmVsRmKch->SetBorderSize(0);

  TPaveText* tTextNoRmVsRmLam = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsRmLam->SetFillColor(0);
    tTextNoRmVsRmLam->SetTextSize(0.03);
    tTextNoRmVsRmLam->AddText("NoRm - RmLam");
    tTextNoRmVsRmLam->AddText("= MisID Lambda");
    tTextNoRmVsRmLam->SetBorderSize(0);

  TPaveText* tTextRmKchVsRmAll = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmKchVsRmAll->SetFillColor(0);
    tTextRmKchVsRmAll->SetTextSize(0.03);
    tTextRmKchVsRmAll->AddText("RmKch - RmAll");
    tTextRmKchVsRmAll->AddText("= MisID Lambda");
    tTextRmKchVsRmAll->SetBorderSize(0);

  TPaveText* tTextRmLamVsRmAll = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmLamVsRmAll->SetFillColor(0);
    tTextRmLamVsRmAll->SetTextSize(0.03);
    tTextRmLamVsRmAll->AddText("RmLam - RmAll");
    tTextRmLamVsRmAll->AddText("= MisID Kch");
    tTextRmLamVsRmAll->SetBorderSize(0);

  TPaveText* tTextNoRmVsKchFix = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsKchFix->SetFillColor(0);
    tTextNoRmVsKchFix->SetTextSize(0.03);
    tTextNoRmVsKchFix->AddText("NoRm - KchFix");
    tTextNoRmVsKchFix->AddText("= Removed Kch");
    tTextNoRmVsKchFix->SetBorderSize(0);

  TPaveText* tTextKchFixVsRmKch = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextKchFixVsRmKch->SetFillColor(0);
    tTextKchFixVsRmKch->SetTextSize(0.03);
    tTextKchFixVsRmKch->AddText("KchFix - RmKch");
    tTextKchFixVsRmKch->AddText("= MisID Kch left");
    tTextKchFixVsRmKch->SetBorderSize(0);

  TPaveText* tTextRmKchVsKchFix = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmKchVsKchFix->SetFillColor(0);
    tTextRmKchVsKchFix->SetTextSize(0.03);
    tTextRmKchVsKchFix->AddText("RmKch - KchFix");
    tTextRmKchVsKchFix->AddText("= True Kch cut");
    tTextRmKchVsKchFix->SetBorderSize(0);

  TPaveText* tTextNoRmVsKchAndLamFix2 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsKchAndLamFix2->SetFillColor(0);
    tTextNoRmVsKchAndLamFix2->SetTextSize(0.03);
    tTextNoRmVsKchAndLamFix2->AddText("NoRm - KchAndLamFix2");
    tTextNoRmVsKchAndLamFix2->AddText("");
    tTextNoRmVsKchAndLamFix2->SetBorderSize(0);

  TPaveText* tTextRmAllVsKchAndLamFix2 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmAllVsKchAndLamFix2->SetFillColor(0);
    tTextRmAllVsKchAndLamFix2->SetTextSize(0.03);
    tTextRmAllVsKchAndLamFix2->AddText("RmAll - KchAndLamFix2");
    tTextRmAllVsKchAndLamFix2->AddText("");
    tTextRmAllVsKchAndLamFix2->SetBorderSize(0);

  TPaveText* tTextNoRmVsKchAndLamFix1 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextNoRmVsKchAndLamFix1->SetFillColor(0);
    tTextNoRmVsKchAndLamFix1->SetTextSize(0.03);
    tTextNoRmVsKchAndLamFix1->AddText("NoRm - KchAndLamFix1");
    tTextNoRmVsKchAndLamFix1->AddText("");
    tTextNoRmVsKchAndLamFix1->SetBorderSize(0);

  TPaveText* tTextRmAllVsKchAndLamFix1 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextRmAllVsKchAndLamFix1->SetFillColor(0);
    tTextRmAllVsKchAndLamFix1->SetTextSize(0.03);
    tTextRmAllVsKchAndLamFix1->AddText("RmAll - KchAndLamFix1");
    tTextRmAllVsKchAndLamFix1->AddText("");
    tTextRmAllVsKchAndLamFix1->SetBorderSize(0);

  TPaveText* tTextKchAndLamFix1VsKchAndLamFix2 = new TPaveText(0.15,0.80,0.35,0.90,"NDC");
    tTextKchAndLamFix1VsKchAndLamFix2->SetFillColor(0);
    tTextKchAndLamFix1VsKchAndLamFix2->SetTextSize(0.03);
    tTextKchAndLamFix1VsKchAndLamFix2->AddText("KchAndLamFix1 - KchAndLamFix2");
    tTextKchAndLamFix1VsKchAndLamFix2->AddText("");
    tTextKchAndLamFix1VsKchAndLamFix2->SetBorderSize(0);

  //------------------TPaveTexts: End------------------------


  TString FileDirectory = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_RmMisID_20160225/";

  TString FileNoRm = FileDirectory + "Results_cLamcKch_AsRcMC_20160224_Bp1.root";
  TString FileRmAll = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisID_20160225_Bp1.root";
  TString FileRmKch = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDKch_20160225_Bp1.root";
  TString FileRmLam = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDLam_20160225_Bp1.root";
  TString FileKchFix = FileDirectory + "Results_cLamcKch_AsRcMC_KchFix_20160226_Bp1.root";
  TString FileKchAndLamFix2 = FileDirectory + "Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229_Bp1.root";
  TString FileKchAndLamFix1 = FileDirectory + "Results_cLamcKch_AsRcMC_KchAndLamFix1_20160229_Bp1.root";

  //--------------------------------------

  AnalysisType tAnalysisType;
    tAnalysisType = kLamKchP;
    //tAnalysisType = kALamKchP;
    //tAnalysisType = kLamKchM;
    //tAnalysisType = kALamKchM;

  KStarTrueVsRecType tKStarTrueVsRecType;
    tKStarTrueVsRecType = kSame;
    //tKStarTrueVsRecType = kRotSame;
    //tKStarTrueVsRecType = kMixed;
    //tKStarTrueVsRecType = kRotMixed;

  NormalizationType tNormalizationType;
    tNormalizationType = kUnNorm;
    //tNormalizationType = kNormByTotEnt;
    //tNormalizationType = kNormColumns;
    //tNormalizationType = kNormRows;
    //tNormalizationType = kNormCentralPeak;

  //--------------------------------------
  TString tAnalysisBaseTag = TString(cAnalysisBaseTags[tAnalysisType]);
  TString tCentralityTag = "_0010";

  TString tDirectoryName = tAnalysisBaseTag + tCentralityTag;

  TString tHistoName;
    if(tKStarTrueVsRecType == kSame) tHistoName = cModelKStarTrueVsRecSameBaseTag;
    else if(tKStarTrueVsRecType == kRotSame) tHistoName = cModelKStarTrueVsRecRotSameBaseTag;
    else if(tKStarTrueVsRecType == kMixed) tHistoName = cModelKStarTrueVsRecMixedBaseTag;
    else if(tKStarTrueVsRecType == kRotMixed) tHistoName = cModelKStarTrueVsRecRotMixedBaseTag;
    else cout << "\t\t\t\t Invalid tKStarTrueVsRecType selection!" << endl;
  tHistoName += tAnalysisBaseTag;
    TString tNewNameNoRm = tHistoName + "_NoRm";
    TString tNewNameRmAll = tHistoName + "_RmAll";
    TString tNewNameRmKch = tHistoName + "_RmKch";
    TString tNewNameRmLam = tHistoName + "_RmLam";
    TString tNewNameKchFix = tHistoName + "_KchFix";
    TString tNewNameKchAndLamFix2 = tHistoName + "_KchAndLamFix2";
    TString tNewNameKchAndLamFix1 = tHistoName + "_KchAndLamFix1";


  //--------------------------------------
  bool bSaveFigures = true;
  TString tSaveLocation = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_RmMisID_20160225/Diffs/";
    tSaveLocation += TString(cAnalysisBaseTags[tAnalysisType]);
    tSaveLocation += "/";

  if(tKStarTrueVsRecType == kSame || tKStarTrueVsRecType == kRotSame)
  {
    tSaveLocation += "Same/";
    if(tKStarTrueVsRecType == kSame) tSaveLocation += "Normal/";
    else tSaveLocation += "Rot/";
  }

  if(tKStarTrueVsRecType == kMixed || tKStarTrueVsRecType == kRotMixed)
  {
    tSaveLocation += "Mixed/";
    if(tKStarTrueVsRecType == kMixed) tSaveLocation += "Normal/";
    else tSaveLocation += "Rot/";
  }

  if(tNormalizationType == kUnNorm) tSaveLocation += "UnNormalized/";
  else if(tNormalizationType == kNormByTotEnt) tSaveLocation += "NormalizedByTotalEntries/";

  //----------------------------------------

  TH2* tNoRm = Get2dHisto(FileNoRm,tDirectoryName,tHistoName,tNewNameNoRm);
  TH2* tRmAll = Get2dHisto(FileRmAll,tDirectoryName,tHistoName,tNewNameRmAll);
  TH2* tRmKch = Get2dHisto(FileRmKch,tDirectoryName,tHistoName,tNewNameRmKch);
  TH2* tRmLam = Get2dHisto(FileRmLam,tDirectoryName,tHistoName,tNewNameRmLam);
  TH2* tKchFix = Get2dHisto(FileKchFix,tDirectoryName,tHistoName,tNewNameKchFix);
  TH2* tKchAndLamFix2 = Get2dHisto(FileKchAndLamFix2,tDirectoryName,tHistoName,tNewNameKchAndLamFix2);
  TH2* tKchAndLamFix1 = Get2dHisto(FileKchAndLamFix1,tDirectoryName,tHistoName,tNewNameKchAndLamFix1);

  tNoRm->SetLabelSize(0.025,"z");
  tRmAll->SetLabelSize(0.025,"z");
  tRmKch->SetLabelSize(0.025,"z");
  tRmLam->SetLabelSize(0.025,"z");
  tKchFix->SetLabelSize(0.025,"z");
  tKchAndLamFix2->SetLabelSize(0.025,"z");
  tKchAndLamFix1->SetLabelSize(0.025,"z");

  //----------------------------------------

  if(tNormalizationType == kNormByTotEnt)
  {
    NormalizeByTotalEntries(tNoRm);
    NormalizeByTotalEntries(tRmAll);
    NormalizeByTotalEntries(tRmKch);
    NormalizeByTotalEntries(tRmLam);
    NormalizeByTotalEntries(tKchFix);
    NormalizeByTotalEntries(tKchAndLamFix2);
    NormalizeByTotalEntries(tKchAndLamFix1);

    double tZlow = 0.00000001;
    double tZhigh = 0.01;

    tNoRm->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tRmAll->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tRmKch->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tRmLam->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tKchFix->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tKchAndLamFix2->GetZaxis()->SetRangeUser(tZlow,tZhigh);
    tKchAndLamFix1->GetZaxis()->SetRangeUser(tZlow,tZhigh);
  }

  if(tNormalizationType == kNormColumns)
  {
    NormalizeEachColumn(tNoRm);
    NormalizeEachColumn(tRmAll);
    NormalizeEachColumn(tRmKch);
    NormalizeEachColumn(tRmLam);
    NormalizeEachColumn(tKchFix);
    NormalizeEachColumn(tKchAndLamFix2);
    NormalizeEachColumn(tKchAndLamFix1);
  }

  if(tNormalizationType == kNormRows)
  {
    NormalizeEachRow(tNoRm);
    NormalizeEachRow(tRmAll);
    NormalizeEachRow(tRmKch);
    NormalizeEachRow(tRmLam);
    NormalizeEachRow(tKchFix);
    NormalizeEachRow(tKchAndLamFix2);
    NormalizeEachRow(tKchAndLamFix1);
  }

  if(tNormalizationType == kNormCentralPeak)
  {
    NormalizeCentralPeak(tNoRm);
    NormalizeCentralPeak(tRmAll);
    NormalizeCentralPeak(tRmKch);
    NormalizeCentralPeak(tRmLam);
    NormalizeCentralPeak(tKchFix);
    NormalizeCentralPeak(tKchAndLamFix2);
    NormalizeCentralPeak(tKchAndLamFix1);
  }
  //-----------------------------------------

  TH2 *tDiffNoRmVsRmAll = (TH2*)tNoRm->Clone("tDiffNoRmVsRmAll");
    tDiffNoRmVsRmAll->Add(tRmAll,-1);
  TH2 *tDiffNoRmVsRmKch = (TH2*)tNoRm->Clone("tDiffNoRmVsRmKch");
    tDiffNoRmVsRmKch->Add(tRmKch,-1);
  TH2 *tDiffNoRmVsRmLam = (TH2*)tNoRm->Clone("tDiffNoRmVsRmLam");
    tDiffNoRmVsRmLam->Add(tRmLam,-1);
  TH2 *tDiffRmKchVsRmAll = (TH2*)tRmKch->Clone("tDiffRmKchVsRmAll");
    tDiffRmKchVsRmAll->Add(tRmAll,-1);
  TH2 *tDiffRmLamVsRmAll = (TH2*)tRmLam->Clone("tDiffRmLamVsRmAll");
    tDiffRmLamVsRmAll->Add(tRmAll,-1);

  TH2* tDiffNoRmVsKchFix = (TH2*)tNoRm->Clone("tDiffNoRmVsKchFix");
    tDiffNoRmVsKchFix->Add(tKchFix,-1);
  TH2* tDiffKchFixVsRmKch = (TH2*)tKchFix->Clone("tDiffKchFixVsRmKch");
    tDiffKchFixVsRmKch->Add(tRmKch,-1);
  TH2* tDiffRmKchVsKchFix = (TH2*)tRmKch->Clone("tDiffRmKchVsKchFix");
    tDiffRmKchVsKchFix->Add(tKchFix,-1);

  TH2* tDiffNoRmVsKchAndLamFix2 = (TH2*)tNoRm->Clone("tDiffNoRmVsKchAndLamFix2");
    tDiffNoRmVsKchAndLamFix2->Add(tKchAndLamFix2,-1);
  TH2* tDiffRmAllVsKchAndLamFix2 = (TH2*)tRmAll->Clone("tDiffRmAllVsKchAndLamFix2");
    tDiffRmAllVsKchAndLamFix2->Add(tKchAndLamFix2,-1);

  TH2* tDiffNoRmVsKchAndLamFix1 = (TH2*)tNoRm->Clone("tDiffNoRmVsKchAndLamFix1");
    tDiffNoRmVsKchAndLamFix1->Add(tKchAndLamFix1,-1);
  TH2* tDiffRmAllVsKchAndLamFix1 = (TH2*)tRmAll->Clone("tDiffRmAllVsKchAndLamFix1");
    tDiffRmAllVsKchAndLamFix1->Add(tKchAndLamFix1,-1);

  TH2* tDiffKchAndLamFix1VsKchAndLamFix2 = (TH2*)tKchAndLamFix1->Clone("tDiffKchAndLamFix1VsKchAndLamFix2");
    tDiffKchAndLamFix1VsKchAndLamFix2->Add(tKchAndLamFix2,-1);

  //-----------------------------------------
  TCanvas *tCanNoRmVsRmAll = new TCanvas("tCanNoRmVsRmAll","tCanNoRmVsRmAll");
    tCanNoRmVsRmAll->Divide(2,1);
  TCanvas *tCanNoRmVsRmKch = new TCanvas("tCanNoRmVsRmKch","tCanNoRmVsRmKch");
    tCanNoRmVsRmKch->Divide(2,1);
  TCanvas *tCanNoRmVsRmLam = new TCanvas("tCanNoRmVsRmLam","tCanNoRmVsRmLam");
    tCanNoRmVsRmLam->Divide(2,1);
  TCanvas *tCanRmKchVsRmAll = new TCanvas("tCanRmKchVsRmAll","tCanRmKchVsRmAll");
    tCanRmKchVsRmAll->Divide(2,1);
  TCanvas *tCanRmLamVsRmAll = new TCanvas("tCanRmLamVsRmAll","tCanRmLamVsRmAll");
    tCanRmLamVsRmAll->Divide(2,1);
  TCanvas *tCanNoRmVsKchFix = new TCanvas("tCanNoRmVsKchFix","tCanNoRmVsKchFix");
    tCanNoRmVsKchFix->Divide(2,1);
  TCanvas *tCanKchFixVsRmKch = new TCanvas("tCanKchFixVsRmKch","tCanKchFixVsRmKch");
    tCanKchFixVsRmKch->Divide(2,1);
 TCanvas *tCanRmKchVsKchFix = new TCanvas("tCanRmKchVsKchFix","tCanRmKchVsKchFix");
    tCanRmKchVsKchFix->Divide(2,1);

 TCanvas *tCanNoRmVsKchAndLamFix2 = new TCanvas("tCanNoRmVsKchAndLamFix2","tCanNoRmVsKchAndLamFix2");
    tCanNoRmVsKchAndLamFix2->Divide(2,1);
 TCanvas *tCanRmAllVsKchAndLamFix2 = new TCanvas("tCanRmAllVsKchAndLamFix2","tCanRmAllVsKchAndLamFix2");
    tCanRmAllVsKchAndLamFix2->Divide(2,1);

 TCanvas *tCanNoRmVsKchAndLamFix1 = new TCanvas("tCanNoRmVsKchAndLamFix1","tCanNoRmVsKchAndLamFix1");
    tCanNoRmVsKchAndLamFix1->Divide(2,1);
 TCanvas *tCanRmAllVsKchAndLamFix1 = new TCanvas("tCanRmAllVsKchAndLamFix1","tCanRmAllVsKchAndLamFix1");
    tCanRmAllVsKchAndLamFix1->Divide(2,1);

 TCanvas *tCanKchAndLamFix1VsKchAndLamFix2 = new TCanvas("tCanKchAndLamFix1VsKchAndLamFix2","tCanKchAndLamFix1VsKchAndLamFix2");
    tCanKchAndLamFix1VsKchAndLamFix2->Divide(2,1);


  //-------------------------------------------
  gStyle->SetOptStat(0);
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
  TPad* myPad2 = tCanNoRmVsRmKch->cd(1);
    myPad2->Divide(1,2);

  myPad2->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad2->cd(2);
    gPad->SetLogz();
    tRmKch->DrawCopy("colz");
    tTextRmKch->Draw();

  tCanNoRmVsRmKch->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsRmKch->DrawCopy("colz");
    tTextNoRmVsRmKch->Draw();

  //-----------------------------
  TPad* myPad3 = tCanNoRmVsRmLam->cd(1);
    myPad3->Divide(1,2);

  myPad3->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad3->cd(2);
    gPad->SetLogz();
    tRmLam->DrawCopy("colz");
    tTextRmLam->Draw();

  tCanNoRmVsRmLam->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsRmLam->DrawCopy("colz");
    tTextNoRmVsRmLam->Draw();

  //-------------------------------------------
  TPad* myPad4 = tCanRmKchVsRmAll->cd(1);
    myPad4->Divide(1,2);

  myPad4->cd(1);
    gPad->SetLogz();
    tRmKch->DrawCopy("colz");
    tTextRmKch->Draw();
  myPad4->cd(2);
    gPad->SetLogz();
    tRmAll->DrawCopy("colz");
    tTextRmAll->Draw();

  tCanRmKchVsRmAll->cd(2);
    gPad->SetLogz();
    tDiffRmKchVsRmAll->DrawCopy("colz");
    tTextRmKchVsRmAll->Draw();

  //-----------------------------
  TPad* myPad5 = tCanRmLamVsRmAll->cd(1);
    myPad5->Divide(1,2);

  myPad5->cd(1);
    gPad->SetLogz();
    tRmLam->DrawCopy("colz");
    tTextRmLam->Draw();
  myPad5->cd(2);
    gPad->SetLogz();
    tRmAll->DrawCopy("colz");
    tTextRmAll->Draw();

  tCanRmLamVsRmAll->cd(2);
    gPad->SetLogz();
    tDiffRmLamVsRmAll->DrawCopy("colz");
    tTextRmLamVsRmAll->Draw();

  //-----------------------------
  TPad* myPad6 = tCanNoRmVsKchFix->cd(1);
    myPad6->Divide(1,2);

  myPad6->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad6->cd(2);
    gPad->SetLogz();
    tKchFix->DrawCopy("colz");
    tTextKchFix->Draw();

  tCanNoRmVsKchFix->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsKchFix->DrawCopy("colz");
    tTextNoRmVsKchFix->Draw();

  //-----------------------------
  TPad* myPad7 = tCanKchFixVsRmKch->cd(1);
    myPad7->Divide(1,2);

  myPad7->cd(1);
    gPad->SetLogz();
    tKchFix->DrawCopy("colz");
    tTextKchFix->Draw();
  myPad7->cd(2);
    gPad->SetLogz();
    tRmKch->DrawCopy("colz");
    tTextRmKch->Draw();

  tCanKchFixVsRmKch->cd(2);
    gPad->SetLogz();
    tDiffKchFixVsRmKch->DrawCopy("colz");
    tTextKchFixVsRmKch->Draw();

  //-----------------------------
  TPad* myPad8 = tCanRmKchVsKchFix->cd(1);
    myPad8->Divide(1,2);

  myPad8->cd(1);
    gPad->SetLogz();
    tRmKch->DrawCopy("colz");
    tTextRmKch->Draw();
  myPad8->cd(2);
    gPad->SetLogz();
    tKchFix->DrawCopy("colz");
    tTextKchFix->Draw();

  tCanRmKchVsKchFix->cd(2);
    gPad->SetLogz();
    tDiffRmKchVsKchFix->DrawCopy("colz");
    tTextRmKchVsKchFix->Draw();

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

  //-----------------------------
  TPad* myPad11 = tCanNoRmVsKchAndLamFix1->cd(1);
    myPad11->Divide(1,2);

  myPad11->cd(1);
    gPad->SetLogz();
    tNoRm->DrawCopy("colz");
    tTextNoRm->Draw();
  myPad11->cd(2);
    gPad->SetLogz();
    tKchAndLamFix1->DrawCopy("colz");
    tTextKchAndLamFix1->Draw();

  tCanNoRmVsKchAndLamFix1->cd(2);
    gPad->SetLogz();
    tDiffNoRmVsKchAndLamFix1->DrawCopy("colz");
    tTextNoRmVsKchAndLamFix1->Draw();

  //-----------------------------
  TPad* myPad12 = tCanRmAllVsKchAndLamFix1->cd(1);
    myPad12->Divide(1,2);

  myPad12->cd(1);
    gPad->SetLogz();
    tRmAll->DrawCopy("colz");
    tTextRmAll->Draw();

  myPad12->cd(2);
    gPad->SetLogz();
    tKchAndLamFix1->DrawCopy("colz");
    tTextKchAndLamFix1->Draw();

  tCanRmAllVsKchAndLamFix1->cd(2);
    gPad->SetLogz();
    tDiffRmAllVsKchAndLamFix1->DrawCopy("colz");
    tTextRmAllVsKchAndLamFix1->Draw();

  //-----------------------------
  TPad* myPad13 = tCanKchAndLamFix1VsKchAndLamFix2->cd(1);
    myPad13->Divide(1,2);

  myPad13->cd(1);
    gPad->SetLogz();
    tKchAndLamFix1->DrawCopy("colz");
    tTextKchAndLamFix1->Draw();

  myPad13->cd(2);
    gPad->SetLogz();
    tKchAndLamFix2->DrawCopy("colz");
    tTextKchAndLamFix2->Draw();

  tCanKchAndLamFix1VsKchAndLamFix2->cd(2);
    gPad->SetLogz();
    tDiffKchAndLamFix1VsKchAndLamFix2->DrawCopy("colz");
    tTextKchAndLamFix1VsKchAndLamFix2->Draw();


  //-------------------------------------------
  if(bSaveFigures)
  {
    tCanNoRmVsRmAll->SaveAs(tSaveLocation+"NoRmVsRmAll.eps");
    tCanNoRmVsRmKch->SaveAs(tSaveLocation+"NoRmVsRmKch.eps");
    tCanNoRmVsRmLam->SaveAs(tSaveLocation+"NoRmVsRmLam.eps");
    tCanRmKchVsRmAll->SaveAs(tSaveLocation+"RmKchVsRmAll.eps");
    tCanRmLamVsRmAll->SaveAs(tSaveLocation+"RmLamVsRmAll.eps");
    tCanNoRmVsKchFix->SaveAs(tSaveLocation+"NoRmVsKchFix.eps");
    tCanKchFixVsRmKch->SaveAs(tSaveLocation+"KchFixVsRmKch.eps");
    tCanRmKchVsKchFix->SaveAs(tSaveLocation+"RmKchVsKchFix.eps");
    tCanNoRmVsKchAndLamFix2->SaveAs(tSaveLocation+"NoRmVsKchAndLamFix2.eps");
    tCanRmAllVsKchAndLamFix2->SaveAs(tSaveLocation+"RmAllVsKchAndLamFix2.eps");
    tCanNoRmVsKchAndLamFix1->SaveAs(tSaveLocation+"NoRmVsKchAndLamFix1.eps");
    tCanRmAllVsKchAndLamFix1->SaveAs(tSaveLocation+"RmAllVsKchAndLamFix1.eps");
    tCanKchAndLamFix1VsKchAndLamFix2->SaveAs(tSaveLocation+"KchAndLamFix1VsKchAndLamFix2.eps");
  }




}
