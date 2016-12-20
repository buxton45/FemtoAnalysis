/*
 *  KStarCF.cxx
 *
 */


#ifndef KSTARCF_CXX
#define KSTARCF_CXX

#include <AliFemtoCorrFctn.h>

//
#include "KStarCF.h"

#include <iostream>
#include <sstream>

#ifdef __ROOT__
ClassImp(KStarCF)
#endif

KStarCF::KStarCF(): AliFemtoCorrFctn() {}

KStarCF::KStarCF(char *title, const int& nbins, const float KStarLo, const float KStarHi, AliFemtoAnalysis *analysis):
  AliFemtoCorrFctn(),
  _numerator(0),
  _denominator(0),
  _minv(0),
  _qinv(0),
  _minv_m(0),
  _qinv_m(0)
{
  std::cout << "[KStarCF::KStarCF]\n";
  std::cout << "  Setting Analysis: " << analysis << "\n";
  SetAnalysis(analysis);
  std::stringstream ss;

  ss << title << "_Num";
  _numerator = new TH1D(ss.str().c_str(), title, nbins, KStarLo, KStarHi);
  _numerator->Sumw2();

  ss.str(std::string());
  ss.clear();
  ss << title << "_Den";
  _denominator = new TH1D(ss.str().c_str(), title, nbins, KStarLo, KStarHi);
  _denominator->Sumw2();

  float mmin = 0, mmax = 4.0;
  float nbins_minv = 800.;
  
  ss.str(std::string());
  ss.clear();
  ss << title << "_minv";
  _minv = new TH1D(ss.str().c_str(), "m_{inv};m_{inv} (GeV)", nbins_minv, mmin,mmax);
  _minv->Sumw2();

  ss.str(std::string());
  ss.clear();
  ss << title << "_minv_m";
  _minv_m = new TH1D(ss.str().c_str(), "m_{inv} (mixed events);m_{inv} (GeV)", nbins_minv, mmin,mmax);

  ss.str(std::string());
  ss.clear();
  ss << title << "_qinv";
  _qinv = new TH1D(ss.str().c_str(), "q_{inv};q_{inv} (GeV)", nbins, mmin,mmax);

  ss.str(std::string());
  ss.clear();
  ss << title << "_qinv_m";
  _qinv_m = new TH1D(ss.str().c_str(), "q_{inv} (mixed events);q_{inv} (GeV)", nbins, mmin,mmax);
}


AliFemtoString
KStarCF::Report()
{
  std::stringstream ss;
  ss << "[KStarCF::Report]\n";
  ss << "  # in numerator : " << _numerator->GetEntries() << "\n";
  ss << "  # in denominator : " << _denominator->GetEntries() << "\n";
  AliFemtoString res = ss.str();
  return res;
}

void
KStarCF::Finish()
{

}

TList*
KStarCF::GetOutputList()
{
  TList *olist = new TList();
  olist->Add(_numerator);
  olist->Add(_denominator);
  olist->Add(_minv);
  olist->Add(_qinv);
  olist->Add(_minv_m);
  olist->Add(_qinv_m);
  return olist;
}


void
KStarCF::AddRealPair(AliFemtoPair* aPair)
{
  if (fPairCut && !fPairCut->Pass(aPair)) {
    return;
  }
  double kstar = aPair->KStar();
  _numerator->Fill(kstar);

  _minv->Fill(aPair->MInv());
  _qinv->Fill(aPair->QInv());
}

void
KStarCF::AddMixedPair(AliFemtoPair* aPair)
{
  if (fPairCut && !fPairCut->Pass(aPair)) {
    return;
  }
  double kstar = aPair->KStar();
  _denominator->Fill(kstar);

  _minv_m->Fill(aPair->MInv());
  _qinv_m->Fill(aPair->QInv());

}

/*
void
KStarCF::EventBegin(const AliFemtoEvent* // aEvent
)
{
}

void
KStarCF::EventEnd(const AliFemtoEvent* // aEvent
)
{
}
// */

#endif /* KSTARCF_CXX */
