/* ThermFlowAnalysis.cxx */

#include "ThermFlowAnalysis.h"

#ifdef __ROOT__
ClassImp(ThermFlowAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
ThermFlowAnalysis::ThermFlowAnalysis(int aNpTBins, double apTBinSize, int aPID) :
  fPID(aPID),
  fNpTBins(aNpTBins),
  fpTBinSize(apTBinSize),

  fEtaA(1.0),
  fEtaB(-1.0),
  fEtaOI(0.8),

  fNEvTot(0),
  fNEv_pTBins(fNpTBins, 0),

  fEns_Res_v2(0.),
  fEns_Res_v2_Sq(0.),
  fVarEns_Res_v2(0.),
  fEns_v2(fNpTBins, 0),
  fEns_v2_Sq(fNpTBins, 0),
  fVarEns_v2(fNpTBins, 0),

  fEns_Res_v3(0.),
  fEns_Res_v3_Sq(0.),
  fVarEns_Res_v3(0.),
  fEns_v3(fNpTBins, 0),
  fEns_v3_Sq(fNpTBins, 0),
  fVarEns_v3(fNpTBins, 0),

  fEns_pT(fNpTBins, 0),
  fEns_pT_Sq(fNpTBins, 0),
  fVarEns_pT(fNpTBins, 0),

  fGraphV2(nullptr),
  fGraphV3(nullptr)

{

}



//________________________________________________________________________________________________________________
ThermFlowAnalysis::~ThermFlowAnalysis()
{
  delete fGraphV2;
  delete fGraphV3;
}


//________________________________________________________________________________________________________________
void ThermFlowAnalysis::BuildVnEPIngredients(ThermEvent &aEvent, double aHarmonic)
{
  complex<double> tImI (0., 1.);
  complex<double> tNull (0., 0.);

  const vector<ThermParticle> tAllPartColl = aEvent.GetAllParticlesCollection();
  unsigned int tAllMult = tAllPartColl.size();
  double tM_QnA=0., tM_QnB=0.;
  complex<double> tQnA (0., 0.);
  complex<double> tQnB (0., 0.);


  vector<ThermParticle> tPOIColl;
  if(fPID==0) tPOIColl = aEvent.GetAllParticlesCollection();
  else tPOIColl = aEvent.GetGoodParticleCollectionwConjCastAsThermParticle(fPID);
  unsigned int tPOIMult = tPOIColl.size();
  vector<complex<double>> tQn(fNpTBins, tNull);
  vector<double> tM_Qn(fNpTBins, 0.);
  vector<double> tInd_pT(fNpTBins, 0.);
  vector<double> tInd_pT_Sq(fNpTBins, 0.);
  vector<double> tInd_vn(fNpTBins, 0.);
  vector<double> tInd_vn_Sq(fNpTBins, 0.);

  double tInd_Res=0., tInd_Res_Sq=0.;

  //First, build reference flow using all particles
  for(unsigned int iPart=0; iPart<tAllMult; iPart++)
  {
    if(tAllPartColl[iPart].GetEtaP() >= fEtaA)
    {
      tQnA += exp(tImI*aHarmonic*tAllPartColl[iPart].GetPhiP());
      tM_QnA++;
    }
    else if(tAllPartColl[iPart].GetEtaP() <= fEtaB)
    {
      tQnB += exp(tImI*aHarmonic*tAllPartColl[iPart].GetPhiP());
      tM_QnB++;
    }
  }

  for(unsigned int iPart=0; iPart<tPOIMult; iPart++)
  {
    if(abs(tPOIColl[iPart].GetEtaP()) < fEtaOI && (fPID==0 || abs(tPOIColl[iPart].GetPID())==fPID))
    {
      for(int iBin=0; iBin<fNpTBins; iBin++)
      {
        if( (tPOIColl[iPart].GetPt() > iBin*fpTBinSize) && (tPOIColl[iPart].GetPt() <= (iBin+1)*fpTBinSize) )
        {
          tQn[iBin] += exp(tImI*aHarmonic*tPOIColl[iPart].GetPhiP());
          tInd_pT[iBin] += tPOIColl[iPart].GetPt();
          tInd_pT_Sq[iBin] += pow(tPOIColl[iPart].GetPt(), 2);
          tM_Qn[iBin]++;
          break;
        }
      }
    }
  }

  for(int iBin=0; iBin<fNpTBins; iBin++)
  {
    if(tM_Qn[iBin] > 0)
    {
      tQn[iBin] /= tM_Qn[iBin];
      tInd_pT[iBin] /= tM_Qn[iBin];
      tInd_pT_Sq[iBin] /= tM_Qn[iBin];
    }
  }
  tQnA /= tM_QnA;
  tQnB /= tM_QnB;

  tInd_Res = real((tQnB/abs(tQnB))*(conj(tQnA)/abs(tQnA)));
  tInd_Res_Sq = pow(tInd_Res, 2);

  for(int iBin=0; iBin<fNpTBins; iBin++)
  {
    tInd_vn[iBin] = real(tQn[iBin]*conj(tQnA)/abs(tQnA));
    tInd_vn_Sq[iBin] = pow(tInd_vn[iBin], 2);
  }


  //----------------- Add to ensembles --------------------
  if(aHarmonic==2.)
  {
    fEns_Res_v2 += tInd_Res;
    fEns_Res_v2_Sq += tInd_Res_Sq;
  }
  else if(aHarmonic==3.)
  {
    fEns_Res_v3 += tInd_Res;
    fEns_Res_v3_Sq += tInd_Res_Sq;
  }
  else assert(0);

  for(int iBin=0; iBin<fNpTBins; iBin++)
  {
    if(aHarmonic==2.)
    {
      fEns_v2[iBin] += tInd_vn[iBin];
      fEns_v2_Sq[iBin] += tInd_vn_Sq[iBin];

      fEns_pT[iBin] += tInd_pT[iBin];
      fEns_pT_Sq[iBin] += tInd_pT_Sq[iBin];

      if(tM_Qn[iBin] >0) fNEv_pTBins[iBin]++;
    }
    else if(aHarmonic==3.)
    {
      fEns_v3[iBin] += tInd_vn[iBin];
      fEns_v3_Sq[iBin] += tInd_vn_Sq[iBin];
    }
    else assert(0);
  }

}

//________________________________________________________________________________________________________________
void ThermFlowAnalysis::BuildVnEPIngredients(ThermEvent &aEvent)
{
  fNEvTot++;
  BuildVnEPIngredients(aEvent, 2.0);
  BuildVnEPIngredients(aEvent, 3.0);
}

//________________________________________________________________________________________________________________
void ThermFlowAnalysis::BuildVnGraphs()
{
  TString tParticleName;
  if(fPID==0) tParticleName = TString("UnIdent");
  else tParticleName = GetParticleNamev2(fPID);

  double tv2EP[fNpTBins], tv2EP_Err[fNpTBins];
  double tv3EP[fNpTBins], tv3EP_Err[fNpTBins];
  double tpT[fNpTBins], tpT_Err[fNpTBins];

  if(fEns_Res_v2 < 0) cout << "WARNING!!!!!!!!" << endl << "fEns_Res_v2<0 so no resolution correction used! (PID = " << fPID << ")" << endl;
  if(fEns_Res_v3 < 0) cout << "WARNING!!!!!!!!" << endl << "fEns_Res_v3<0 so no resolution correction used! (PID = " << fPID << ")" << endl;

  double tVar_v2EP=0., tVar_v3EP=0.;
  for(int i=0; i<fNpTBins; i++)
  {
    if(fEns_Res_v2 > 0)
    {
      tv2EP[i] = fEns_v2[i]/sqrt(fEns_Res_v2);
      tVar_v2EP = pow(tv2EP[i], 2)*(fVarEns_v2[i]/pow(fEns_v2[i], 2) + 0.25*fVarEns_Res_v2/pow(fEns_Res_v2, 2));
    }
    else
    {
      tv2EP[i] = fEns_v2[i];
      tVar_v2EP = fVarEns_v2[i];
    }
    tv2EP_Err[i] = sqrt(tVar_v2EP);
    //-----------------------------
    if(fEns_Res_v3 > 0)
    {
      tv3EP[i] = fEns_v3[i]/sqrt(fEns_Res_v3);
      tVar_v3EP = pow(tv3EP[i], 2)*(fVarEns_v3[i]/pow(fEns_v3[i], 2) + 0.25*fVarEns_Res_v3/pow(fEns_Res_v3, 2));
    }
    else
    {
      tv3EP[i] = fEns_v3[i];
      tVar_v3EP = fVarEns_v3[i];
    }
    tv3EP_Err[i] = sqrt(tVar_v3EP);
    //-----------------------------
    tpT[i] = fEns_pT[i];
    tpT_Err[i] = sqrt(fVarEns_pT[i]);
  }

/*
  for(int i=0; i<fNpTBins; i++)
  {
    cout << "i     = " << i << endl;
    cout << "tpT   = " << tpT[i] << endl;
    cout << "tv2EP = " << tv2EP[i] << endl;
    cout << "tv3EP = " << tv3EP[i] << endl << endl;
  }
*/

  int tColor = 1;
  if(fPID==321) tColor = kRed;
  else if(fPID==311) tColor = kYellow-3;
  else if(fPID==3122) tColor = kBlue+1;

  fGraphV2 = new TGraphErrors(fNpTBins, tpT, tv2EP, tpT_Err, tv2EP_Err);
    fGraphV2->SetMarkerSize(0.75);
    fGraphV2->SetMarkerColor(tColor);
    fGraphV2->SetMarkerStyle(20);
    fGraphV2->SetLineColor(tColor);
    fGraphV2->SetTitle(TString::Format("v_{2}{EP, |#Delta#eta| > 2.0} vs. p_{T} (|#eta|<0.8) (%s)", tParticleName.Data()));
    fGraphV2->SetName(TString::Format("v2_%s", tParticleName.Data()));

  fGraphV3 = new TGraphErrors(fNpTBins, tpT, tv3EP, tpT_Err, tv3EP_Err);
    fGraphV3->SetMarkerSize(0.75);
    fGraphV3->SetMarkerColor(tColor);
    fGraphV3->SetMarkerStyle(24);
    fGraphV3->SetLineColor(tColor);
    fGraphV3->SetTitle(TString::Format("v_{3}{EP, |#Delta#eta| > 2.0} vs. p_{T} (|#eta|<0.8) (%s)", tParticleName.Data()));	
    fGraphV3->SetName(TString::Format("v3_%s", tParticleName.Data()));

  fGraphV2->GetXaxis()->SetTitle("<p_{T}>");
  fGraphV2->GetYaxis()->SetTitle("v_{n}{EP}");

  fGraphV3->GetXaxis()->SetTitle("<p_{T}>");
  fGraphV3->GetYaxis()->SetTitle("v_{n}{EP}");
}

//________________________________________________________________________________________________________________
TCanvas* ThermFlowAnalysis::DrawFlowHarmonics()
{
  if(!fGraphV2 || !fGraphV3) BuildVnGraphs();

  TString tCanName;
  if(fPID==0) tCanName = TString("tFlowCan_UnIdent");
  else tCanName = TString::Format("tFlowCan_%s", GetParticleNamev2(fPID).Data());
  TCanvas* tFlowCan = new TCanvas(tCanName, tCanName);
  tFlowCan->cd();

  fGraphV2->Draw("AP");
  fGraphV3->Draw("Psame");
  tFlowCan->Update();

  return tFlowCan;
}

//________________________________________________________________________________________________________________
TObjArray* ThermFlowAnalysis::GetVnGraphs()
{
  if(!fGraphV2 || !fGraphV3) BuildVnGraphs();

  TObjArray* tGraphs = new TObjArray();
  tGraphs->Add(fGraphV2);
  tGraphs->Add(fGraphV3);

  return tGraphs;
}

//________________________________________________________________________________________________________________
void ThermFlowAnalysis::SaveGraphs(TFile* aFile)
{
  assert(aFile->IsOpen());

  fGraphV2->Write();
  fGraphV3->Write();
}

//________________________________________________________________________________________________________________
void ThermFlowAnalysis::Finalize()
{
  cout << "Total #events in flow calculations: " << fNEvTot << endl;
  cout << "PID = " << fPID << endl << endl;

  fEns_Res_v2 /= fNEvTot;
  fEns_Res_v2_Sq /= fNEvTot;

  fEns_Res_v3 /= fNEvTot;
  fEns_Res_v3_Sq /= fNEvTot;

  fVarEns_Res_v2 = (fEns_Res_v2_Sq - pow(fEns_Res_v2, 2))/fNEvTot;
  fVarEns_Res_v3 = (fEns_Res_v3_Sq - pow(fEns_Res_v3, 2))/fNEvTot;


  for(int iBin=0; iBin<fNpTBins; iBin++)
  {
    if(fNEv_pTBins[iBin] > 0)
    {
      fEns_v2[iBin] /= fNEv_pTBins[iBin];
      fEns_v2_Sq[iBin] /= fNEv_pTBins[iBin];

      fEns_v3[iBin] /= fNEv_pTBins[iBin];
      fEns_v3_Sq[iBin] /= fNEv_pTBins[iBin];

      fEns_pT[iBin] /= fNEv_pTBins[iBin];
      fEns_pT_Sq[iBin] /= fNEv_pTBins[iBin];

      //---------

      fVarEns_v2[iBin] = (fEns_v2_Sq[iBin] - pow(fEns_v2[iBin], 2))/fNEv_pTBins[iBin]; //Not sure these should be divided by fNEv_pTBins[iBin]
      fVarEns_v3[iBin] = (fEns_v3_Sq[iBin] - pow(fEns_v3[iBin], 2))/fNEv_pTBins[iBin];
      fVarEns_pT[iBin] = (fEns_pT_Sq[iBin] - pow(fEns_pT[iBin], 2))/fNEv_pTBins[iBin];
    }
  }

  //----------------------------
  BuildVnGraphs();
  DrawFlowHarmonics();
}

