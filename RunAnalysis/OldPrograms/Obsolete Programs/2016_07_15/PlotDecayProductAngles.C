#include <cmath>
#include "TF1.h"
#include "TCanvas.h"
#include "TAxis.h"

double LabAngle(double *x, double *par)
{
  double scale = 3.14159265359/180.;
/*
  double tanTheta = sin(scale*x[0])/(par[0]*(cos(scale*x[0])+par[1]/par[2]));
  double Theta = atan(tanTheta);
*/
  double Num = sin(scale*x[0]);
  double Den = (par[0]*(cos(scale*x[0])+par[1]/par[2]));
  double Theta = atan2(Num,Den);

  return (1./scale)*Theta;
}


TObjArray* GetLabAngles(double aMassMother, double aMomentumMother, double aMassDaughter1, double aMassDaughter2)
{
  double Gamma = sqrt(pow(aMomentumMother,2)+pow(aMassMother,2))/aMassMother;
  double Beta = aMomentumMother/(sqrt(pow(aMomentumMother,2)+pow(aMassMother,2)));

  double PStarSquared = (1/(4*pow(aMassMother,2)))*((pow(aMassMother,2)-pow((aMassDaughter1+aMassDaughter2),2))*(pow(aMassMother,2)-pow((aMassDaughter1-aMassDaughter2),2)));
  double PStar = sqrt(PStarSquared);

  double E1Star = (pow(aMassMother,2)+pow(aMassDaughter1,2)-pow(aMassDaughter2,2))/(2*aMassMother);
  double E2Star = (pow(aMassMother,2)-pow(aMassDaughter1,2)+pow(aMassDaughter2,2))/(2*aMassMother);

  double Beta1Star = PStar/E1Star;
  double Beta2Star = PStar/E2Star;

  TF1 *LabAngle1 = new TF1("LabAngle1",LabAngle,0,180,3);
    LabAngle1->SetParameter(0,Gamma);
    LabAngle1->SetParameter(1,Beta);
    LabAngle1->SetParameter(2,Beta1Star);

  TF1 *LabAngle2 = new TF1("LabAngle2",LabAngle,-180,0,3);
    LabAngle2->SetParameter(0,Gamma);
    LabAngle2->SetParameter(1,Beta);
    LabAngle2->SetParameter(2,Beta2Star);

  TObjArray* aReturnArray = new TObjArray();
    aReturnArray->Add(LabAngle1);
    aReturnArray->Add(LabAngle2);

  return aReturnArray;

}


void PlotDecayProductAngles()
{
static const double PionMass = 0.13956995,
                    KaonMass = 0.493677,
                  ProtonMass = 0.938272013,
                  LambdaMass = 1.115683;
  //------------------------------------------

  double MassMother = LambdaMass;
  double MassDaughter1 = ProtonMass;
    TString Daughter1 = "Proton Daughter";
  double MassDaughter2 = PionMass;
    TString Daughter2 = "Pion Daughter";

  TString aSaveName = "PlotDecayProductAngles_Lamda.pdf";


  const int nMomMother = 9;
  double MomentumMotherArray[nMomMother] = {0.,.25,.5,.75,1.,2.,3.,4.,5.};

  //----------------------------------
  TCanvas* PlottingCanvas = new TCanvas("PlottingCanvas","Angular Distribution of Daughters");
    PlottingCanvas->Divide(1,2);
    gStyle->SetOptStat(0);

  PlottingCanvas->cd(1);
    TH1F* AidHisto1 = new TH1F();
    TAxis* xax1 = AidHisto1->GetXaxis();
      xax1->SetLimits(0,180);
      xax1->SetTitle("#Theta_{CM}");
    TAxis* yax1 = AidHisto1->GetYaxis();
      yax1->SetRangeUser(0,180);
      yax1->SetTitle("#Theta_{LAB}");
    AidHisto1->SetTitle(Daughter1);
    AidHisto1->Draw();
    TLegend* leg1 = new TLegend(0.15,0.45,0.25,0.85);
      leg1->SetHeader("P_{mother}");


  PlottingCanvas->cd(2);
    TH1F* AidHisto2 = new TH1F();
    TAxis* xax2 = AidHisto2->GetXaxis();
      xax2->SetLimits(-180,0);
      xax2->SetTitle("#Theta_{CM}");
    TAxis* yax2 = AidHisto2->GetYaxis();
      yax2->SetRangeUser(-180,0);
      yax2->SetTitle("#Theta_{LAB}");
    AidHisto2->SetTitle(Daughter2);
    AidHisto2->Draw();
  //----------------------------------

  for(int i=0; i<nMomMother; i++)
  {
    TObjArray* LabAngles = GetLabAngles(MassMother,MomentumMotherArray[i],MassDaughter1,MassDaughter2);

    PlottingCanvas->cd(1);
    TF1* LabAngle1 = LabAngles->At(0);
    LabAngle1->SetLineColor(i+1);
    //LabAngle1->SetLineStyle(i+1);
    LabAngle1->Draw("SAME");

    TString aString = "";
    aString += MomentumMotherArray[i];
    leg1->AddEntry(LabAngle1, aString, "l");

    PlottingCanvas->cd(2);
    TF1* LabAngle2 = LabAngles->At(1);
    LabAngle2->SetLineColor(i+1);
    //LabAngle2->SetLineStyle(i+1);
    LabAngle2->Draw("SAME");
  }
  PlottingCanvas->cd(1);
  leg1->Draw();

  PlottingCanvas->SaveAs(aSaveName);
}
