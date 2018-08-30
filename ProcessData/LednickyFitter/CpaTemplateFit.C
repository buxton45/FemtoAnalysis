#include "FitGeneratorAndDraw.h"
class FitGeneratorAndDraw;

#include "TFractionFitter.h"

vector<TString> tDefaultBins{"Primary", 
                             "#Lambda", "K^{0}_{S}", 
                             "#Sigma^{0}", "#Xi^{0}", "#Xi^{-}", 
                             "#Sigma^{*+}", "#Sigma^{*-}", "#Sigma^{*0}", "K^{*0}", "K^{*+}", 
                             "Other", "Fake", "Material"}; 


//________________________________________________________________________________________________________________
TH1F* Get1dHist(TString aFileLocation, TString aDirectoryName, TString aHistName, TString aNewName)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
  else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
  else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
  else assert(0);

  //TODO
  tFemtoListName = TString("LambdaKaon");

  tFemtoListName += TString("_femtolist");
  tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  aDirectoryName.ReplaceAll("0010","010");

  tFemtolist->SetOwner();

  TObjArray *tDir = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    tDir->SetOwner();

  //-----------------------------------------------------------------------------

  TH1F *tHisto = (TH1F*)tDir->FindObject(aHistName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "1dHisto NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHisto);
  //----------------------------------

  TH1F *ReturnHist = (TH1F*)tHisto->Clone(aNewName);
  ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}


  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtolist);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tFemtolist->Delete();
  delete tFemtolist;

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;

  //-------------------------------------------------------------------

  return (TH1F*)ReturnHist;
}

//________________________________________________________________________________________________________________
TH2F* Get2dHist(TString aFileLocation, TString aDirectoryName, TString aHistName, TString aNewName)
{
  TFile *tFile = TFile::Open(aFileLocation);
  TList *tFemtolist;
  TString tFemtoListName;
  TDirectoryFile *tDirFile;

  tDirFile = (TDirectoryFile*)tFile->Get("PWG2FEMTO");
  if(aDirectoryName.Contains("LamKch")) tFemtoListName = "cLamcKch";
  else if(aDirectoryName.Contains("LamK0")) tFemtoListName = "cLamK0";
  else if(aDirectoryName.Contains("XiKch")) tFemtoListName = "cXicKch";
  else assert(0);

  //TODO
  tFemtoListName = TString("LambdaKaon");

  tFemtoListName += TString("_femtolist");
  tFemtolist = (TList*)tDirFile->Get(tFemtoListName);
  aDirectoryName.ReplaceAll("0010","010");

  tFemtolist->SetOwner();

  TObjArray *tDir = (TObjArray*)tFemtolist->FindObject(aDirectoryName)->Clone();
    tDir->SetOwner();

  //-----------------------------------------------------------------------------

  TH2F *tHisto = (TH2F*)tDir->FindObject(aHistName);

  //-----make sure tHisto is retrieved
  if(!tHisto) {cout << "2dHisto NOT FOUND!!!:  Name:  " << aHistName << endl;}
  assert(tHisto);
  //----------------------------------

  TH2F *ReturnHist = (TH2F*)tHisto->Clone(aNewName);
  ReturnHist->SetDirectory(0);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!ReturnHist->GetSumw2N()) {ReturnHist->Sumw2();}


  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //This needs to be done, otherwise the other TObjArrays in TList are
  //thrown onto the stack (even after calling delete on the tFemtolist object)
  //which causes the RAM to be used up rapidly!
  //In short, TLists are stupid
  TIter next(tFemtolist);
  TObject *obj = nullptr;
  while((obj = next()))
  {
    TObjArray *arr = dynamic_cast<TObjArray*>(obj);
    if(arr) arr->Delete();
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  tFemtolist->Delete();
  delete tFemtolist;

  tDirFile->Close();
  delete tDirFile;

  tFile->Close();
  delete tFile;

  //-------------------------------------------------------------------

  return (TH2F*)ReturnHist;
}


//________________________________________________________________________________________________________________
TH1F* Get1dProjection(TH2F* a2dHist, TString aBinName)
{
  int tBinNumber = -1;
  for(int i=1; i<=a2dHist->GetNbinsY(); i++) if(aBinName.EqualTo(a2dHist->GetYaxis()->GetBinLabel(i))) tBinNumber=i;
  assert(tBinNumber > 0);

  TString tNewName = TString::Format("%s CPA", aBinName.Data());
  TH1F* tReturnHist = (TH1F*)a2dHist->ProjectionX("", tBinNumber, tBinNumber);

  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1F* Get1dProjection(TH2F* a2dHist, int aDefaultBin)
{
  TString tBinName = tDefaultBins[aDefaultBin];
  return Get1dProjection(a2dHist, tBinName);
}


//________________________________________________________________________________________________________________
void DoTemplateFit(TH1F* aData, TH2F* a2dMC, int tNRes=3)
{
/*
  int tNHist=0;
  if(tNRes==3) tNHist = 7;
  if(tNRes==10) tNHist = 12;
*/

  //First, grab all MC hisograms
  TH1F* tPrimary  = Get1dProjection(a2dMC, 0);
  TH1F* tLambda   = Get1dProjection(a2dMC, 1);
  TH1F* tK0s      = Get1dProjection(a2dMC, 2);
  TH1F* tSig0     = Get1dProjection(a2dMC, 3);
  TH1F* tXi0      = Get1dProjection(a2dMC, 4);
  TH1F* tXiC      = Get1dProjection(a2dMC, 5);
  TH1F* tSigStP   = Get1dProjection(a2dMC, 6);
  TH1F* tSigStM   = Get1dProjection(a2dMC, 7);
  TH1F* tSigSt0   = Get1dProjection(a2dMC, 8);
  TH1F* tKSt0     = Get1dProjection(a2dMC, 9);
  TH1F* tKStP     = Get1dProjection(a2dMC, 10);
  TH1F* tOther    = Get1dProjection(a2dMC, 11);
  TH1F* tFake     = Get1dProjection(a2dMC, 12);
  TH1F* tMaterial = Get1dProjection(a2dMC, 13);

  assert(tPrimary->GetXaxis()->GetBinWidth(1)==aData->GetXaxis()->GetBinWidth(1));
  //---------------------------------------------
  TObjArray* tMCArr = new TObjArray();
  tMCArr->Add(tPrimary);
  tMCArr->Add(tSig0);
  tMCArr->Add(tXi0);
  tMCArr->Add(tXiC);
  if(tNRes==10)
  {
    tMCArr->Add(tSigStP);
    tMCArr->Add(tSigStM);
    tMCArr->Add(tSigSt0);
    tMCArr->Add(tKSt0);
    tMCArr->Add(tKStP);
  }
  else
  {
    tOther->Add(tSigStP);
    tOther->Add(tSigStM);
    tOther->Add(tSigSt0);
    tOther->Add(tKSt0);
    tOther->Add(tKStP);
  }
  tMCArr->Add(tOther);
  tMCArr->Add(tFake);
  tMCArr->Add(tMaterial);
  //---------------------------------------------
  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();
  gStyle->SetOptStat(0);

  TFractionFitter* tFracFit = new TFractionFitter(aData, tMCArr);
  for(int i=0; i<tMCArr->GetEntries(); i++) tFracFit->Constrain(0, 0., 1.);  //Constrain all parameters between 0 and 1
  tFracFit->SetRangeX(1, aData->GetNbinsX());  //set the number of bins to fit
  int tStatus = tFracFit->Fit();
  cout << "Fit status: " << tStatus << endl;
  if(tStatus==0)
  {
    TH1F* tResult = (TH1F*)tFracFit->GetPlot();
    aData->Draw("Ep");
    tResult->Draw("same");
  }

}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------

  TString tResultsDate_Data = "20180505";
  TString tResultsDate_MC   = "20180830";

  AnalysisType tAnType = kLamK0;
  ParticleType tParticleType = kLam; //kLam, kALam, kK0
  CentralityType tCentType = k0010;

  bool SaveImages = false;
  TString tSaveFileType = "eps";
  TString tSaveDir = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/Comments/Laura/20180117/Figures/";

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);

  TString tDirectoryBase_Data = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_Data.Data());
  TString tFileLocationBase_Data = TString::Format("%sResults_%s_%s",tDirectoryBase_Data.Data(),tGeneralAnTypeName.Data(),tResultsDate_Data.Data());
  TString tFileLocation_Data_FemtoMinus = TString::Format("%s_FemtoMinus.root", tFileLocationBase_Data.Data());
  TString tFileLocation_Data_FemtoPlus  = TString::Format("%s_FemtoPlus.root", tFileLocationBase_Data.Data());

  TString tDirectoryBase_MC = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate_MC.Data());
  TString tFileLocationBase_MC = TString::Format("%sResults_%s_%s",tDirectoryBase_MC.Data(),tGeneralAnTypeName.Data(),tResultsDate_MC.Data());
  TString tFileLocation_MC_FemtoMinus = TString::Format("%s_FemtoMinus.root", tFileLocationBase_MC.Data());
  TString tFileLocation_MC_FemtoPlus  = TString::Format("%s_FemtoPlus.root", tFileLocationBase_MC.Data());

  //TODO
  tFileLocation_MC_FemtoMinus = TString("/home/jesse/Analysis/FemtoAnalysis/RunAnalysis/Test/AnalysisResults.root");
//-----------------------------------------------------------------------------

  TString t1dHistName = TString::Format("CosPointingAngle_%s_Pass", cParticleTags[tParticleType]);
  TString t2dHistName = TString::Format("CosPointingAnglewParentInfo_%s_Pass", cParticleTags[tParticleType]);
  TString tDirName = TString::Format("%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]);

//-------------------------------------------------------------------------------
  TH1F* tData_FemtoMinus = Get1dHist(tFileLocation_Data_FemtoMinus, tDirName, t1dHistName, t1dHistName);
  TH1F* tData_FemtoPlus  = Get1dHist(tFileLocation_Data_FemtoPlus,  tDirName, t1dHistName, t1dHistName);
  TH1F* tData = (TH1F*)tData_FemtoMinus->Clone(TString::Format("Data_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]));
    tData->Add(tData_FemtoPlus);

  TH2F* tMC_FemtoMinus = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistName, t2dHistName);
//  TH2F* tMC_FemtoPlus  = Get2dHist(tFileLocation_MC_FemtoPlus, tDirName, t2dHistName, t2dHistName);
  TH2F* tMC = (TH2F*)tMC_FemtoMinus->Clone(TString::Format("MC_%s%s", cAnalysisBaseTags[tAnType], cCentralityTags[tCentType]));
//    tMC->Add(tMC_FemtoPlus);

  TH2F* tTest2d = Get2dHist(tFileLocation_MC_FemtoMinus, tDirName, t2dHistName, t2dHistName);


  DoTemplateFit(tData, tMC, 3);

//-------------------------------------------------------------------------------

  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
