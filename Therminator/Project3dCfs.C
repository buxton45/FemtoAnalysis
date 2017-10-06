#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

//________________________________________________________________________________________________________________
TH1D* GetFull(TString aFileLocationCfs, AnalysisType aAnType)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfFull%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetPrimaryOnly(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  int tIndex1 = GetParticleIndexInPidInfo(aType1) + 1;
  int tIndex2 = GetParticleIndexInPidInfo(aType2) + 1;

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]), tIndex1, tIndex1, tIndex2, tIndex2);
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]), tIndex1, tIndex1, tIndex2, tIndex2);

  TString tReturnName = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetSecondaryOnly(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  int tIndex1 = GetParticleIndexInPidInfo(aType1) + 1;
  int tIndex2 = GetParticleIndexInPidInfo(aType2) + 1;

  for(int i=1; i<=tNum3d->GetNbinsX(); i++)
  {
    for(int j=1; j<=tNum3d->GetNbinsY(); j++)
    {
      if(i==tIndex1 || j==tIndex2)
      {
        for(int k=1; k<tNum3d->GetNbinsZ(); k++)
        {
          tNum3d->SetBinContent(i, j, k, 0.);
          tDen3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetAtLeastOneSecondaryInPair(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  int tIndex1 = GetParticleIndexInPidInfo(aType1) + 1;
  int tIndex2 = GetParticleIndexInPidInfo(aType2) + 1;

  for(int i=1; i<=tNum3d->GetNbinsX(); i++)
  {
    for(int j=1; j<=tNum3d->GetNbinsY(); j++)
    {
      if(i!=tIndex1 || j!=tIndex2)
      {
        for(int k=1; k<tNum3d->GetNbinsZ(); k++)
        {
          tNum3d->SetBinContent(i, j, k, 0.);
          tDen3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetWithoutSigmaSt(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tNum3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1==kPDGSigStP || tType1==kPDGASigStM ||
       tType1==kPDGSigStM || tType1==kPDGASigStP ||
       tType1==kPDGSigSt0 || tType1==kPDGSigSt0)
    {
      for(int j=1; j<=tNum3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tNum3d->GetNbinsZ(); k++)
        {
          tNum3d->SetBinContent(i, j, k, 0.);
          tDen3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetSigmaStOnly(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  ParticlePDGType tType1;
//  ParticlePDGType tType2;
  for(int i=1; i<=tNum3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    if(tType1 != kPDGSigStP && tType1 != kPDGASigStM &&
       tType1 != kPDGSigStM && tType1 != kPDGASigStP &&
       tType1 != kPDGSigSt0 && tType1 != kPDGSigSt0)
    {
      for(int j=1; j<=tNum3d->GetNbinsY(); j++)
      {
//        tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
        for(int k=1; k<tNum3d->GetNbinsZ(); k++)
        {
          tNum3d->SetBinContent(i, j, k, 0.);
          tDen3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
}

//________________________________________________________________________________________________________________
TH1D* GetPrimaryAndShortDecays(TString aFileLocationCfs, AnalysisType aAnType, ParticlePDGType aType1, ParticlePDGType aType2)
{
  TH3D* tNum3d = Get3dHisto(aFileLocationCfs, TString::Format("Num3d%s", cAnalysisBaseTags[aAnType]));
  TH3D* tDen3d = Get3dHisto(aFileLocationCfs, TString::Format("Den3d%s", cAnalysisBaseTags[aAnType]));

  int tIndex1 = GetParticleIndexInPidInfo(aType1) + 1;
  int tIndex2 = GetParticleIndexInPidInfo(aType2) + 1;

  ParticlePDGType tType1, tType2;
  for(int i=1; i<=tNum3d->GetNbinsX(); i++)
  {
    tType1 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(i-1));
    for(int j=1; j<=tNum3d->GetNbinsY(); j++)
    {
      tType2 = static_cast<ParticlePDGType>(GetParticlePidFromIndex(j-1));
      if((i != tIndex1 || j != tIndex2) && !IncludeAsPrimary(tType1, tType2, 5.0))
      {
        for(int k=1; k<tNum3d->GetNbinsZ(); k++)
        {
          tNum3d->SetBinContent(i, j, k, 0.);
          tDen3d->SetBinContent(i, j, k, 0.);
        }
      }
    }
  }

  TH1D* tNum = tNum3d->ProjectionZ(TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                   1, tNum3d->GetNbinsX(), 
                                   1, tNum3d->GetNbinsY());
  TH1D* tDen = tDen3d->ProjectionZ(TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]), 
                                   1, tDen3d->GetNbinsX(), 
                                   1, tDen3d->GetNbinsY());

  TString tReturnName = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TH1D* tCf = BuildCf(tNum, tDen, tReturnName);

  return tCf;
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
//-----------------------------------------------------------------------------


  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationCfs = tDirectory + "CorrelationFunctions_10MixedEvNum.root";

/*
  TString tDirectory = "/home/jesse/Analysis/ReducedTherminator2Events/test/";
  TString tFileLocationCfs = tDirectory + "testCorrelationFunctions_5MixedEvNum.root";
*/
  AnalysisType tAnType = kLamKchP;
  ParticlePDGType tType1 = kPDGLam;
  ParticlePDGType tType2 = kPDGKchP;

//-------------------------------------------------------------------------------
  TH1D* tCfFullProject = GetFull(tFileLocationCfs, tAnType);
    tCfFullProject->SetMarkerStyle(20);
    tCfFullProject->SetMarkerColor(1);

  TH1D* tCfFull = Get1dHisto(tFileLocationCfs, TString::Format("CfFull%s", cAnalysisBaseTags[tAnType]));
    tCfFull->SetMarkerStyle(20);
    tCfFull->SetMarkerColor(2);

  TCanvas *tCanFullA = new TCanvas("tCanFullA", "tCanFullA");
  tCanFullA->cd();
  tCfFullProject->Draw();
  tCfFull->Draw("same");

  TCanvas *tCanFullB = new TCanvas("tCanFullB", "tCanFullB");
  tCanFullB->Divide(2,1);
  tCanFullB->cd(1);
  tCfFullProject->Draw();
  tCanFullB->cd(2);
  tCfFull->Draw();
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
  TH1D* tCfPrimaryOnlyProject = GetPrimaryOnly(tFileLocationCfs, tAnType, tType1, tType2);
    tCfPrimaryOnlyProject->SetMarkerStyle(20);
    tCfPrimaryOnlyProject->SetMarkerColor(1);

  TH1D* tCfPrimOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[tAnType]));
    tCfPrimOnly->SetMarkerStyle(20);
    tCfPrimOnly->SetMarkerColor(2);

  TCanvas *tCanPrimA = new TCanvas("tCanPrimA", "tCanPrimA");
  tCanPrimA->cd();
  tCfPrimaryOnlyProject->Draw();
  tCfPrimOnly->Draw("same");

  TCanvas *tCanPrimB = new TCanvas("tCanPrimB", "tCanPrimB");
  tCanPrimB->Divide(2,1);
  tCanPrimB->cd(1);
  tCfPrimaryOnlyProject->Draw();
  tCanPrimB->cd(2);
  tCfPrimOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfSecondaryOnlyProject = GetSecondaryOnly(tFileLocationCfs, tAnType, tType1, tType2);
    tCfSecondaryOnlyProject->SetMarkerStyle(20);
    tCfSecondaryOnlyProject->SetMarkerColor(1);

  TH1D* tCfSecondaryOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[tAnType]));
    tCfSecondaryOnly->SetMarkerStyle(20);
    tCfSecondaryOnly->SetMarkerColor(2);

  TCanvas *tCanSecondaryA = new TCanvas("tCanSecondaryA", "tCanSecondaryA");
  tCanSecondaryA->cd();
  tCfSecondaryOnlyProject->Draw();
  tCfSecondaryOnly->Draw("same");

  TCanvas *tCanSecondaryB = new TCanvas("tCanSecondaryB", "tCanSecondaryB");
  tCanSecondaryB->Divide(2,1);
  tCanSecondaryB->cd(1);
  tCfSecondaryOnlyProject->Draw();
  tCanSecondaryB->cd(2);
  tCfSecondaryOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfWithoutSigmaStProject = GetWithoutSigmaSt(tFileLocationCfs, tAnType, tType1, tType2);
    tCfWithoutSigmaStProject->SetMarkerStyle(20);
    tCfWithoutSigmaStProject->SetMarkerColor(1);

  TH1D* tCfWithoutSigmaSt = Get1dHisto(tFileLocationCfs, TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[tAnType]));
    tCfWithoutSigmaSt->SetMarkerStyle(20);
    tCfWithoutSigmaSt->SetMarkerColor(2);

  TCanvas *tCanWithoutSigmaStA = new TCanvas("tCanWithoutSigmaStA", "tCanWithoutSigmaStA");
  tCanWithoutSigmaStA->cd();
  tCfWithoutSigmaStProject->Draw();
  tCfWithoutSigmaSt->Draw("same");

  TCanvas *tCanWithoutSigmaStB = new TCanvas("tCanWithoutSigmaStB", "tCanWithoutSigmaStB");
  tCanWithoutSigmaStB->Divide(2,1);
  tCanWithoutSigmaStB->cd(1);
  tCfWithoutSigmaStProject->Draw();
  tCanWithoutSigmaStB->cd(2);
  tCfWithoutSigmaSt->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfSigmaStOnlyProject = GetSigmaStOnly(tFileLocationCfs, tAnType, tType1, tType2);
    tCfSigmaStOnlyProject->SetMarkerStyle(20);
    tCfSigmaStOnlyProject->SetMarkerColor(1);

  TH1D* tCfSigmaStOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[tAnType]));
    tCfSigmaStOnly->SetMarkerStyle(20);
    tCfSigmaStOnly->SetMarkerColor(2);

  TCanvas *tCanSigmaStOnlyA = new TCanvas("tCanSigmaStOnlyA", "tCanSigmaStOnlyA");
  tCanSigmaStOnlyA->cd();
  tCfSigmaStOnlyProject->Draw();
  tCfSigmaStOnly->Draw("same");

  TCanvas *tCanSigmaStOnlyB = new TCanvas("tCanSigmaStOnlyB", "tCanSigmaStOnlyB");
  tCanSigmaStOnlyB->Divide(2,1);
  tCanSigmaStOnlyB->cd(1);
  tCfSigmaStOnlyProject->Draw();
  tCanSigmaStOnlyB->cd(2);
  tCfSigmaStOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfPrimaryAndShortDecaysProject = GetPrimaryAndShortDecays(tFileLocationCfs, tAnType, tType1, tType2);
    tCfPrimaryAndShortDecaysProject->SetMarkerStyle(20);
    tCfPrimaryAndShortDecaysProject->SetMarkerColor(1);

  TH1D* tCfPrimaryAndShortDecays = Get1dHisto(tFileLocationCfs, TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[tAnType]));
    tCfPrimaryAndShortDecays->SetMarkerStyle(20);
    tCfPrimaryAndShortDecays->SetMarkerColor(2);

  TCanvas *tCanPrimaryAndShortDecaysA = new TCanvas("tCanPrimaryAndShortDecaysA", "tCanPrimaryAndShortDecaysA");
  tCanPrimaryAndShortDecaysA->cd();
  tCfPrimaryAndShortDecaysProject->Draw();
  tCfPrimaryAndShortDecays->Draw("same");

  TCanvas *tCanPrimaryAndShortDecaysB = new TCanvas("tCanPrimaryAndShortDecaysB", "tCanPrimaryAndShortDecaysB");
  tCanPrimaryAndShortDecaysB->Divide(2,1);
  tCanPrimaryAndShortDecaysB->cd(1);
  tCfPrimaryAndShortDecaysProject->Draw();
  tCanPrimaryAndShortDecaysB->cd(2);
  tCfPrimaryAndShortDecays->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfAtLeastOneSecondaryInPairProject = GetAtLeastOneSecondaryInPair(tFileLocationCfs, tAnType, tType1, tType2);
    tCfAtLeastOneSecondaryInPairProject->SetMarkerStyle(20);
    tCfAtLeastOneSecondaryInPairProject->SetMarkerColor(1);

  TCanvas *tCanAtLeastOneSecondaryInPairA = new TCanvas("tCanAtLeastOneSecondaryInPairA", "tCanAtLeastOneSecondaryInPairA");
  tCanAtLeastOneSecondaryInPairA->cd();
  tCfAtLeastOneSecondaryInPairProject->Draw();
//-------------------------------------------------------------------------------


  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}


