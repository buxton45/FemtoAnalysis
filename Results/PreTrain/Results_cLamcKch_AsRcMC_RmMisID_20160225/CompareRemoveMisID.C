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
    for(int j=1; j<=tNbinsY; j++)
    {
      double tNewContent = (1.0/tScale)*aHisto->GetBinContent(i,j);
      aHisto->SetBinContent(i,j,tNewContent);
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



void CompareRemoveMisID()
{

  TString FileDirectory = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_RmMisID_20160225/";

  TString FileNoRm = FileDirectory + "Results_cLamcKch_AsRcMC_20160224_Bp1.root";
  TString FileRmAll = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisID_20160225_Bp1.root";
  TString FileRmKch = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDKch_20160225_Bp1.root";
  TString FileRmLam = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDLam_20160225_Bp1.root";

  //--------------------------------------

  AnalysisType tAnalysisType;
    tAnalysisType = kLamKchP;
    //tAnalysisType = kALamKchP;
    //tAnalysisType = kLamKchM;
    //tAnalysisType = kALamKchM;

  KStarTrueVsRecType tKStarTrueVsRecType;
    //tKStarTrueVsRecType = kSame;
    tKStarTrueVsRecType = kRotSame;
    //tKStarTrueVsRecType = kMixed;
    //tKStarTrueVsRecType = kRotMixed;

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

  //----------------------------------------

  TH2* tNoRm = Get2dHisto(FileNoRm,tDirectoryName,tHistoName,tNewNameNoRm);
  TH2* tRmAll = Get2dHisto(FileRmAll,tDirectoryName,tHistoName,tNewNameRmAll);
  TH2* tRmKch = Get2dHisto(FileRmKch,tDirectoryName,tHistoName,tNewNameRmKch);
  TH2* tRmLam = Get2dHisto(FileRmLam,tDirectoryName,tHistoName,tNewNameRmLam);


//cout << "tNoRm->GetNbinsX() = " << tNoRm->GetNbinsX() << endl;
//cout << "tNoRm->GetNbinsY() = " << tNoRm->GetNbinsY() << endl;


/*
  NormalizeByTotalEntries(tNoRm);
  NormalizeByTotalEntries(tRmAll);
  NormalizeByTotalEntries(tRmKch);
  NormalizeByTotalEntries(tRmLam);
*/
/*
  NormalizeEachColumn(tNoRm);
  NormalizeEachColumn(tRmAll);
  NormalizeEachColumn(tRmKch);
  NormalizeEachColumn(tRmLam);
*/
/*
  NormalizeEachRow(tNoRm);
  NormalizeEachRow(tRmAll);
  NormalizeEachRow(tRmKch);
  NormalizeEachRow(tRmLam);
*/

  NormalizeCentralPeak(tNoRm);
  NormalizeCentralPeak(tRmAll);
  NormalizeCentralPeak(tRmKch);
  NormalizeCentralPeak(tRmLam);


  TCanvas *aCan1 = new TCanvas("aCan1","aCan1");
    aCan1->Divide(2,2);

  aCan1->cd(1);
    gPad->SetLogz();
  tNoRm->Draw("colz");

  aCan1->cd(2);
    gPad->SetLogz();
  tRmAll->Draw("colz");

  aCan1->cd(3);
    gPad->SetLogz();
  tRmKch->Draw("colz");

  aCan1->cd(4);
    gPad->SetLogz();
  tRmLam->Draw("colz");

}
