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



void CompareRemoveMisID2()
{

  TString FileDirectory = "~/Analysis/K0Lam/Results_cLamcKch_AsRcMC_RmMisID_20160225/";

  TString FileNoRm = FileDirectory + "Results_cLamcKch_AsRcMC_20160224_Bp1.root";
  TString FileRmAll = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisID_20160225_Bp1.root";
  TString FileRmKch = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDKch_20160225_Bp1.root";
  TString FileRmLam = FileDirectory + "Results_cLamcKch_AsRcMC_RmMisIDLam_20160225_Bp1.root";

  //--------------------------------------
  TString tFileName;
    //tFileName = FileNoRm;
    tFileName = FileRmAll;
    //tFileName = FileRmKch;
    //tFileName = FileRmLam;

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

  TString tNameSame = cModelKStarTrueVsRecSameBaseTag + tAnalysisBaseTag;
  TString tNameRotSame = cModelKStarTrueVsRecRotSameBaseTag + tAnalysisBaseTag;
  TString tNameMixed = cModelKStarTrueVsRecMixedBaseTag + tAnalysisBaseTag;
  TString tNameRotMixed = cModelKStarTrueVsRecRotMixedBaseTag + tAnalysisBaseTag;

  //----------------------------------------

  TH2* tSame = Get2dHisto(tFileName,tDirectoryName,tNameSame,tNameSame);
  TH2* tRotSame = Get2dHisto(tFileName,tDirectoryName,tNameRotSame,tNameRotSame);
  TH2* tMixed = Get2dHisto(tFileName,tDirectoryName,tNameMixed,tNameMixed);
  TH2* tRotMixed = Get2dHisto(tFileName,tDirectoryName,tNameRotMixed,tNameRotMixed);



/*
  NormalizeByTotalEntries(tSame);
  NormalizeByTotalEntries(tRotSame);
  NormalizeByTotalEntries(tMixed);
  NormalizeByTotalEntries(tRotMixed);
*/
/*
  NormalizeEachColumn(tSame);
  NormalizeEachColumn(tRotSame);
  NormalizeEachColumn(tMixed);
  NormalizeEachColumn(tRotMixed);
*/
/*
  NormalizeEachRow(tSame);
  NormalizeEachRow(tRotSame);
  NormalizeEachRow(tMixed);
  NormalizeEachRow(tRotMixed);
*/
/*
  NormalizeCentralPeak(tSame);
  NormalizeCentralPeak(tRotSame);
  NormalizeCentralPeak(tMixed);
  NormalizeCentralPeak(tRotMixed);
*/

  TCanvas *aCan1 = new TCanvas("aCan1","aCan1");
    aCan1->Divide(2,2);

  aCan1->cd(1);
    gPad->SetLogz();
  tSame->Draw("colz");

  aCan1->cd(2);
    gPad->SetLogz();
  tRotSame->Draw("colz");

  aCan1->cd(3);
    gPad->SetLogz();
  tMixed->Draw("colz");

  aCan1->cd(4);
    gPad->SetLogz();
  tRotMixed->Draw("colz");

}
