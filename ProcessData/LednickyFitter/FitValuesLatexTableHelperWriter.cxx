/* FitValuesLatexTableHelperWriter.cxx */

#include "FitValuesLatexTableHelperWriter.h"

#ifdef __ROOT__
ClassImp(FitValuesLatexTableHelperWriter)
#endif




//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriter::FitValuesLatexTableHelperWriter() : 
  FitValuesWriter(),
  fResType(kInclude3Residuals)
{

}





//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriter::~FitValuesLatexTableHelperWriter()
{
  //no-op
}


//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID(TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  TString tReturnFitInfoTString;
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) tReturnFitInfoTString = GetFitInfoTStringFromTwoLetterID_LamKch(aTwoLetterID, aResType, tResPrimMaxDecayType);
  else if(aAnType==kLamK0 || aAnType==kALamK0)   tReturnFitInfoTString = GetFitInfoTStringFromTwoLetterID_LamK0(aTwoLetterID, aResType, tResPrimMaxDecayType);

  return tReturnFitInfoTString;
}

//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID_LamKch(TString aTwoLetterID, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  TString tCommon1 = "_MomResCrctn";
  TString tCommon2 = TString::Format("%s_UsingXiDataAndCoulombOnly", cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);
  if(aResType==kIncludeNoResiduals) tCommon2 = TString("");

  TString tReturnFitInfoTString = "";

  if     (aTwoLetterID.EqualTo("Aa")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ba")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ca")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Da")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ea")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ab")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Db")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Eb")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ac")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bc")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cc")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Dc")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ec")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ad")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Bd")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Cd")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Dd")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ed")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s_StavCf_ShareLam_Dualie_ShareLam_ShareRadii", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  //--------------------------------------------------------------
  else assert(0);

  return tReturnFitInfoTString;
}

//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID_LamK0(TString aTwoLetterID, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  TString tCommon1 = "_MomResCrctn";
  TString tCommon2 = TString::Format("%s_UsingXiDataAndCoulombOnly", cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);
  if(aResType==kIncludeNoResiduals) tCommon2 = TString("");

  TString tReturnFitInfoTString = "";

  if     (aTwoLetterID.EqualTo("Aa")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnPolynomial%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ab")) tReturnFitInfoTString = TString::Format("%s_NonFlatBgdCrctnLinear%s%s", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else if(aTwoLetterID.EqualTo("Ac")) tReturnFitInfoTString = TString::Format("%s%s%s_StavCf", tCommon1.Data(), cIncludeResidualsTypeTags[aResType], tCommon2.Data());
  else assert(0);

  tReturnFitInfoTString += TString("_SingleLamParam");

  return tReturnFitInfoTString;
}


//________________________________________________________________________________________________________________
TString FitValuesLatexTableHelperWriter::GetLatexTableOverallLabel(TString aTwoLetterID)
{
  TString tReturnLabel = "";

  if     (aTwoLetterID.EqualTo("Aa")) tReturnLabel = TString("FitGen_PolyNFB_NoLamShare");
  else if(aTwoLetterID.EqualTo("Ba")) tReturnLabel = TString("FitGen_PolyNFB_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Ca")) tReturnLabel = TString("Dualie_PolyNFB_NoLamShare");
  else if(aTwoLetterID.EqualTo("Da")) tReturnLabel = TString("Dualie_PolyNFB_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Ea")) tReturnLabel = TString("Dualie_PolyNFB_ShareSingleLam");
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ab")) tReturnLabel = TString("FitGen_LinrNFB_NoLamShare");
  else if(aTwoLetterID.EqualTo("Bb")) tReturnLabel = TString("FitGen_LinrNFB_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Cb")) tReturnLabel = TString("Dualie_LinrNFB_NoLamShare");
  else if(aTwoLetterID.EqualTo("Db")) tReturnLabel = TString("Dualie_LinrNFB_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Eb")) tReturnLabel = TString("Dualie_LinrNFB_ShareSingleLam");
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ac")) tReturnLabel = TString("FitGen_NoNFB_StavCf_NoLamShare");
  else if(aTwoLetterID.EqualTo("Bc")) tReturnLabel = TString("FitGen_NoNFB_StavCf_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Cc")) tReturnLabel = TString("Dualie_NoNFB_StavCf_NoLamShare");
  else if(aTwoLetterID.EqualTo("Dc")) tReturnLabel = TString("Dualie_NoNFB_StavCf_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Ec")) tReturnLabel = TString("Dualie_NoNFB_StavCf_ShareSingleLam");
  //--------------------------------------------------------------
  else if(aTwoLetterID.EqualTo("Ad")) tReturnLabel = TString("FitGen_LinrNFB_StavCf_NoLamShare");
  else if(aTwoLetterID.EqualTo("Bd")) tReturnLabel = TString("FitGen_LinrNFB_StavCf_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Cd")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_NoLamShare");
  else if(aTwoLetterID.EqualTo("Dd")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_ShareLamConj");
  else if(aTwoLetterID.EqualTo("Ed")) tReturnLabel = TString("Dualie_LinrNFB_StavCf_ShareSingleLam");
  //--------------------------------------------------------------
  else assert(0);

  return tReturnLabel;
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  TString tFitInfoTString = GetFitInfoTStringFromTwoLetterID(aTwoLetterID, aAnType, aResType, tResPrimMaxDecayType);

  FitParameter *tPar_lam0010, *tPar_lam1030, *tPar_lam3050;
  FitParameter *tPar_R0010, *tPar_R1030, *tPar_R3050;
  FitParameter *tPar_Ref0, *tPar_Imf0, *tPar_d0;

  tPar_lam0010 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k0010, kLambda);
  tPar_lam1030 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k1030, kLambda);
  tPar_lam3050 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k3050, kLambda);

  tPar_R0010 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k0010, kRadius);
  tPar_R1030 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k1030, kRadius);
  tPar_R3050 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k3050, kRadius);

  tPar_Ref0 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k0010, kRef0);
  tPar_Imf0 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k0010, kImf0);
  tPar_d0 = GetFitParameter(aMasterFileLocation, tFitInfoTString, aAnType, k0010, kd0);


  //------------------------------------------------------------------
  TString tAnalysisTag;
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) tAnalysisTag = cAnalysisBaseTags[aAnType];
  else if(aAnType==kLamK0) tAnalysisTag = TString("LamKs");
  else if(aAnType==kALamK0) tAnalysisTag = TString("ALamKs");
  else assert(0);

  aOut << TString::Format("\\newarray\\%s%s", aTwoLetterID.Data(), tAnalysisTag.Data()) << endl;
  aOut << TString::Format("\\readarray{%s%s}{", aTwoLetterID.Data(), tAnalysisTag.Data()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam0010->GetFitValue(), tPar_R0010->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam1030->GetFitValue(), tPar_R1030->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & ", tPar_lam3050->GetFitValue(), tPar_R3050->GetFitValue()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f}", tPar_Ref0->GetFitValue(), tPar_Imf0->GetFitValue(), tPar_d0->GetFitValue()) << endl << endl;


}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP)
  {
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kLamKchP,  aResType, tResPrimMaxDecayType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kALamKchM, aResType, tResPrimMaxDecayType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kLamKchM,  aResType, tResPrimMaxDecayType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kALamKchP, aResType, tResPrimMaxDecayType);
  }
  else if(aAnType==kLamK0 || aAnType==kALamK0)
  {
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kLamK0,  aResType, tResPrimMaxDecayType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aTwoLetterID, kALamK0, aResType, tResPrimMaxDecayType);
  }
  else assert(0);
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelperHeader(ostream &aOut)
{
  aOut << "\%NOTE: Arrays seem to only be able to be named with alphabetical characters" << endl;
  aOut << "\%      No '_' or event numbers" << endl;
  aOut << endl;
  aOut << "\% ----------Shorthand definitions----------" << endl;
  aOut << "\% A = FitGen_NoLamShare" << endl;
  aOut << "\% B = FitGen_ShareLamConj" << endl;
  aOut << endl;
  aOut << "\% C = Dualie_NoLamShare" << endl;
  aOut << "\% D = Dualie_ShareLamConj" << endl;
  aOut << "\% E = Dualie_ShareSingleLam" << endl;
  aOut << endl;
  aOut << "\%   ------------------" << endl;
  aOut << "\% a = PolyNFB" << endl;
  aOut << "\% b = LinrNFB" << endl;
  aOut << "\% c = LinrNFB_StavCf" << endl;
  aOut << "\% d = NoNFB_StavCf" << endl;
  aOut << endl;
  aOut << "\% ----------How arrays are stored----------" << endl;
  aOut << "\% aArray = {lam0010(1), R0010(2)," << endl;
  aOut << "\%           lam1030(3), R1030(4)," << endl;
  aOut << "\%           lam3050(5), R3050(6)," << endl;
  aOut << "\%           ReF0(7), ImF0(8), d0(9)}" << endl;
  aOut << endl;
  aOut << endl;
  aOut << "\%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriter::WriteLatexTableHelper(TString aHelperBaseLocation, TString aMasterFileLocation, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType)
{
  assert(aAnType==kLamKchP || aAnType==kLamK0);
  TString tPairDesc = "";
  if     (aAnType==kLamKchP) tPairDesc = TString("_cLamcKch");
  else if(aAnType==kLamK0) tPairDesc = TString("_cLamK0");

  TString tHelperLocation = TString::Format("%s%s%s%s.tex", aHelperBaseLocation.Data(), tPairDesc.Data(), cIncludeResidualsTypeTags[aResType], cResPrimMaxDecayTypeTags[tResPrimMaxDecayType]);

  std::ofstream tOut;
  tOut.open(tHelperLocation);
  WriteLatexTableHelperHeader(tOut);

  vector<vector<TString> > tAllTwoLetterID = {{"Aa", "Ba", "Ca", "Da", "Ea"},
                                              {"Ab", "Bb", "Cb", "Db", "Eb"},
                                              {"Ac", "Bc", "Cc", "Dc", "Ec"}};
  if(aAnType==kLamK0 || aAnType==kALamK0) tAllTwoLetterID = vector<vector<TString> > {{"Aa"}, {"Ab"}, {"Ac"}};
  //-------------------------------------------
  TString tTwoLetterID = "";
  for(unsigned int i=0; i<tAllTwoLetterID.size(); i++)
  {
    for(unsigned int j=0; j<tAllTwoLetterID[i].size(); j++)
    {
      tTwoLetterID = tAllTwoLetterID[i][j];
      tOut << TString::Format("%% --------------- %s = %s%s ---------------", tTwoLetterID.Data(), GetLatexTableOverallLabel(tTwoLetterID).Data(), cIncludeResidualsTypeTags[aResType]) << endl;
      WriteLatexTableHelperSection(tOut, aMasterFileLocation, tTwoLetterID, aAnType, aResType, tResPrimMaxDecayType);
    }
    tOut << "\%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;
  }

  tOut.close();
}

