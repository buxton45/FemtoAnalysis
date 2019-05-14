///////////////////////////////////////////////////////////////////////////
// Types_LambdaValues.cxx:                                               //
///////////////////////////////////////////////////////////////////////////


#include "Types_LambdaValues.h"
#include <cassert>

//NOTE:  Changing MaxDecayLengthPrimary only changes the primary lambda values
//       (and the "other" lambda values).
//       i.e. for a given number of residuals included, only the first 6 elements
//       of the arrays will differ


//------------ WHEN INCLUDING NO RESIDUALS ------------------------------------//
const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay0[85] = {
/*kLamK0=*/   1.000,   /*kALamK0=*/   1.000,
/*kLamKchP=*/ 1.000,   /*kALamKchM=*/ 1.000,  /*kLamKchM=*/ 1.000, /*kALamKchP=*/ 1.000,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.000, /*kResASig0KchM=*/   0.000, /*kResSig0KchM=*/   0.000, /*kResASig0KchP=*/  0.000,
/*kResXi0KchP=*/    0.000, /*kResAXi0KchM=*/    0.000, /*kResXi0KchM=*/    0.000, /*kResAXi0KchP=*/   0.000,
/*kResXiCKchP=*/    0.000, /*kResAXiCKchM=*/    0.000, /*kResXiCKchM=*/    0.000, /*kResAXiCKchP=*/   0.000,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.000, /*kResASig0K0=*/   0.000,
/*kResXi0K0=*/    0.000, /*kResAXi0K0=*/    0.000,
/*kResXiCK0=*/    0.000, /*kResAXiCK0=*/    0.000,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay4[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay5[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay6[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay10[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay100[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};

const double cAnalysisLambdaFactors_NoRes_MaxPrimDecay15[85] = {
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
0.,0.,0.,0.,0.
};


const double *cAnalysisLambdaFactors_NoRes[7] = {cAnalysisLambdaFactors_NoRes_MaxPrimDecay0, 
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay4,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay5,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay6,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay10,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay100,
                                                 cAnalysisLambdaFactors_NoRes_MaxPrimDecay15};

//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=0 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay0[85] = {
/*kLamK0=*/   0.138,   /*kALamK0=*/   0.140,
/*kLamKchP=*/ 0.126,   /*kALamKchM=*/ 0.128,  /*kLamKchM=*/ 0.125, /*kALamKchP=*/ 0.129,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.113, /*kResASig0KchM=*/   0.111, /*kResSig0KchM=*/   0.112, /*kResASig0KchP=*/  0.112,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.039, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.051, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.052, /*kResASigStMKchM=*/ 0.050, /*kResSigStPKchM=*/ 0.052, /*kResASigStMKchP=*/ 0.050,
/*kResSigStMKchP=*/ 0.047, /*kResASigStPKchM=*/ 0.048, /*kResSigStMKchM=*/ 0.047, /*kResASigStPKchP=*/ 0.049,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.043, /*kResSigSt0KchM=*/ 0.047, /*kResASigSt0KchP=*/ 0.044,

/*kResLamKSt0=*/  0.045, /*kResALamAKSt0=*/  0.046, /*kResLamAKSt0=*/  0.045, /*kResALamKSt0=*/  0.045,
/*kResSig0KSt0=*/ 0.040, /*kResASig0AKSt0=*/ 0.040, /*kResSig0AKSt0=*/ 0.040, /*kResASig0KSt0=*/ 0.040,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.123, /*kResASig0K0=*/   0.122,
/*kResXi0K0=*/    0.043, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.055, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.057, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.051, /*kResASigStPK0=*/ 0.053,
/*kResSigSt0K0=*/ 0.052, /*kResASigSt0K0=*/ 0.048,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.009, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=4 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay4[85] = {
/*kLamK0=*/   0.187,   /*kALamK0=*/   0.188,
/*kLamKchP=*/ 0.174,   /*kALamKchM=*/ 0.175,  /*kLamKchM=*/ 0.173, /*kALamKchP=*/ 0.175,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.112, /*kResASig0KchM=*/   0.111, /*kResSig0KchM=*/   0.111, /*kResASig0KchP=*/  0.112,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.052, /*kResASigStMKchM=*/ 0.049, /*kResSigStPKchM=*/ 0.051, /*kResASigStMKchP=*/ 0.050,
/*kResSigStMKchP=*/ 0.047, /*kResASigStPKchM=*/ 0.048, /*kResSigStMKchM=*/ 0.046, /*kResASigStPKchP=*/ 0.048,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.043, /*kResSigSt0KchM=*/ 0.046, /*kResASigSt0KchP=*/ 0.043,

/*kResLamKSt0=*/  0.044, /*kResALamAKSt0=*/  0.045, /*kResLamAKSt0=*/  0.044, /*kResALamKSt0=*/  0.045,
/*kResSig0KSt0=*/ 0.040, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.040, /*kResASig0KSt0=*/ 0.039,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.122, /*kResASig0K0=*/   0.121,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.055, /*kResAXiCK0=*/    0.050,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.056, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.051, /*kResASigStPK0=*/ 0.053,
/*kResSigSt0K0=*/ 0.051, /*kResASigSt0K0=*/ 0.047,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.007, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.009, /*kResAXiCAKSt0ToALamK0=*/  0.008
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=5 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay5[85] = {
/*kLamK0=*/   0.244,   /*kALamK0=*/   0.246,
/*kLamKchP=*/ 0.211,   /*kALamKchM=*/ 0.212,  /*kLamKchM=*/ 0.211, /*kALamKchP=*/ 0.212,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.111, /*kResASig0KchM=*/   0.110, /*kResSig0KchM=*/   0.110, /*kResASig0KchP=*/  0.111,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.050, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.049, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.051, /*kResASigStMKchM=*/ 0.049, /*kResSigStPKchM=*/ 0.051, /*kResASigStMKchP=*/ 0.050,
/*kResSigStMKchP=*/ 0.046, /*kResASigStPKchM=*/ 0.048, /*kResSigStMKchM=*/ 0.046, /*kResASigStPKchP=*/ 0.048,
/*kResSigSt0KchP=*/ 0.047, /*kResASigSt0KchM=*/ 0.043, /*kResSigSt0KchM=*/ 0.046, /*kResASigSt0KchP=*/ 0.043,

/*kResLamKSt0=*/  0.044, /*kResALamAKSt0=*/  0.045, /*kResLamAKSt0=*/  0.044, /*kResALamKSt0=*/  0.045,
/*kResSig0KSt0=*/ 0.039, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.039, /*kResASig0KSt0=*/ 0.039,
/*kResXi0KSt0=*/  0.014, /*kResAXi0AKSt0=*/  0.013, /*kResXi0AKSt0=*/  0.014, /*kResAXi0KSt0=*/  0.013,
/*kResXiCKSt0=*/  0.018, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.018, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.121, /*kResASig0K0=*/   0.121,
/*kResXi0K0=*/    0.042, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.054, /*kResAXiCK0=*/    0.050,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.056, /*kResASigStMK0=*/ 0.054,
/*kResSigStMK0=*/ 0.050, /*kResASigStPK0=*/ 0.052,
/*kResSigSt0K0=*/ 0.051, /*kResASigSt0K0=*/ 0.047,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.022,
/*kResSig0KSt0ToLamK0=*/ 0.019, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=6 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay6[85] = {
/*kLamK0=*/   0.333,   /*kALamK0=*/   0.337,
/*kLamKchP=*/ 0.304,   /*kALamKchM=*/ 0.308,  /*kLamKchM=*/ 0.305, /*kALamKchP=*/ 0.307,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.109, /*kResASig0KchM=*/   0.109, /*kResSig0KchM=*/   0.108, /*kResASig0KchP=*/  0.110,
/*kResXi0KchP=*/    0.038, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.038, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.049, /*kResAXiCKchM=*/    0.045, /*kResXiCKchM=*/    0.049, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.051, /*kResASigStMKchM=*/ 0.049, /*kResSigStPKchM=*/ 0.050, /*kResASigStMKchP=*/ 0.049,
/*kResSigStMKchP=*/ 0.046, /*kResASigStPKchM=*/ 0.047, /*kResSigStMKchM=*/ 0.045, /*kResASigStPKchP=*/ 0.048,
/*kResSigSt0KchP=*/ 0.046, /*kResASigSt0KchM=*/ 0.042, /*kResSigSt0KchM=*/ 0.045, /*kResASigSt0KchP=*/ 0.043,

/*kResLamKSt0=*/  0.043, /*kResALamAKSt0=*/  0.045, /*kResLamAKSt0=*/  0.043, /*kResALamKSt0=*/  0.045,
/*kResSig0KSt0=*/ 0.039, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.039, /*kResASig0KSt0=*/ 0.039,
/*kResXi0KSt0=*/  0.013, /*kResAXi0AKSt0=*/  0.012, /*kResXi0AKSt0=*/  0.013, /*kResAXi0KSt0=*/  0.012,
/*kResXiCKSt0=*/  0.017, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.017, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.119, /*kResASig0K0=*/   0.119,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.050,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.055, /*kResASigStMK0=*/ 0.053,
/*kResSigStMK0=*/ 0.050, /*kResASigStPK0=*/ 0.052,
/*kResSigSt0K0=*/ 0.050, /*kResASigSt0K0=*/ 0.046,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.021,
/*kResSig0KSt0ToLamK0=*/ 0.018, /*kResASig0AKSt0ToALamK0=*/ 0.019,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=10 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay10[85] = {
/*kLamK0=*/   0.359,   /*kALamK0=*/   0.362,
/*kLamKchP=*/ 0.330,   /*kALamKchM=*/ 0.332,  /*kLamKchM=*/ 0.330, /*kALamKchP=*/ 0.332,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.109, /*kResASig0KchM=*/   0.109, /*kResSig0KchM=*/   0.108, /*kResASig0KchP=*/  0.110,
/*kResXi0KchP=*/    0.038, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.049, /*kResAXiCKchM=*/    0.045, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.050, /*kResASigStMKchM=*/ 0.048, /*kResSigStPKchM=*/ 0.050, /*kResASigStMKchP=*/ 0.049,
/*kResSigStMKchP=*/ 0.045, /*kResASigStPKchM=*/ 0.047, /*kResSigStMKchM=*/ 0.045, /*kResASigStPKchP=*/ 0.048,
/*kResSigSt0KchP=*/ 0.045, /*kResASigSt0KchM=*/ 0.042, /*kResSigSt0KchM=*/ 0.045, /*kResASigSt0KchP=*/ 0.043,

/*kResLamKSt0=*/  0.043, /*kResALamAKSt0=*/  0.044, /*kResLamAKSt0=*/  0.043, /*kResALamKSt0=*/  0.044,
/*kResSig0KSt0=*/ 0.038, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.038, /*kResASig0KSt0=*/ 0.039,
/*kResXi0KSt0=*/  0.013, /*kResAXi0AKSt0=*/  0.012, /*kResXi0AKSt0=*/  0.013, /*kResAXi0KSt0=*/  0.012,
/*kResXiCKSt0=*/  0.017, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.017, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.118, /*kResASig0K0=*/   0.119,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.055, /*kResASigStMK0=*/ 0.053,
/*kResSigStMK0=*/ 0.049, /*kResASigStPK0=*/ 0.052,
/*kResSigSt0K0=*/ 0.049, /*kResASigSt0K0=*/ 0.046,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.021,
/*kResSig0KSt0ToLamK0=*/ 0.018, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=100 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay100[85] = {
/*kLamK0=*/   0.384,   /*kALamK0=*/   0.387,
/*kLamKchP=*/ 0.369,   /*kALamKchM=*/ 0.373,  /*kLamKchM=*/ 0.370, /*kALamKchP=*/ 0.372,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.108, /*kResASig0KchM=*/   0.108, /*kResSig0KchM=*/   0.107, /*kResASig0KchP=*/  0.109,
/*kResXi0KchP=*/    0.037, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.048, /*kResAXiCKchM=*/    0.045, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.050, /*kResASigStMKchM=*/ 0.048, /*kResSigStPKchM=*/ 0.049, /*kResASigStMKchP=*/ 0.049,
/*kResSigStMKchP=*/ 0.045, /*kResASigStPKchM=*/ 0.047, /*kResSigStMKchM=*/ 0.045, /*kResASigStPKchP=*/ 0.047,
/*kResSigSt0KchP=*/ 0.045, /*kResASigSt0KchM=*/ 0.042, /*kResSigSt0KchM=*/ 0.045, /*kResASigSt0KchP=*/ 0.042,

/*kResLamKSt0=*/  0.043, /*kResALamAKSt0=*/  0.044, /*kResLamAKSt0=*/  0.043, /*kResALamKSt0=*/  0.044,
/*kResSig0KSt0=*/ 0.038, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.038, /*kResASig0KSt0=*/ 0.038,
/*kResXi0KSt0=*/  0.013, /*kResAXi0AKSt0=*/  0.012, /*kResXi0AKSt0=*/  0.013, /*kResAXi0KSt0=*/  0.012,
/*kResXiCKSt0=*/  0.017, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.017, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.118, /*kResASig0K0=*/   0.119,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.054, /*kResASigStMK0=*/ 0.053,
/*kResSigStMK0=*/ 0.049, /*kResASigStPK0=*/ 0.052,
/*kResSigSt0K0=*/ 0.049, /*kResASigSt0K0=*/ 0.046,

/*kResLamKSt0ToLamK0=*/  0.020, /*kResALamAKSt0ToALamK0=*/  0.021,
/*kResSig0KSt0ToLamK0=*/ 0.018, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};

//------------ WHEN INCLUDING ALL 10 RESIDUALS and MaxDecayLengthPrimary=15 ------------------------------------//
const double cAnalysisLambdaFactors_10Res_MaxPrimDecay15[85] = {
/*kLamK0=*/   0.365,   /*kALamK0=*/   0.368,
/*kLamKchP=*/ 0.336,   /*kALamKchM=*/ 0.339,  /*kLamKchM=*/ 0.337, /*kALamKchP=*/ 0.338,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.109, /*kResASig0KchM=*/   0.109, /*kResSig0KchM=*/   0.108, /*kResASig0KchP=*/  0.110,
/*kResXi0KchP=*/    0.038, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.049, /*kResAXiCKchM=*/    0.045, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.050, /*kResASigStMKchM=*/ 0.048, /*kResSigStPKchM=*/ 0.050, /*kResASigStMKchP=*/ 0.049,
/*kResSigStMKchP=*/ 0.045, /*kResASigStPKchM=*/ 0.047, /*kResSigStMKchM=*/ 0.045, /*kResASigStPKchP=*/ 0.048,
/*kResSigSt0KchP=*/ 0.045, /*kResASigSt0KchM=*/ 0.042, /*kResSigSt0KchM=*/ 0.045, /*kResASigSt0KchP=*/ 0.043,

/*kResLamKSt0=*/  0.043, /*kResALamAKSt0=*/  0.044, /*kResLamAKSt0=*/  0.043, /*kResALamKSt0=*/  0.044,
/*kResSig0KSt0=*/ 0.038, /*kResASig0AKSt0=*/ 0.039, /*kResSig0AKSt0=*/ 0.038, /*kResASig0KSt0=*/ 0.039,
/*kResXi0KSt0=*/  0.013, /*kResAXi0AKSt0=*/  0.012, /*kResXi0AKSt0=*/  0.013, /*kResAXi0KSt0=*/  0.012,
/*kResXiCKSt0=*/  0.017, /*kResAXiCAKSt0=*/  0.016, /*kResXiCAKSt0=*/  0.017, /*kResAXiCKSt0=*/  0.016,

/*kResSig0K0=*/   0.118, /*kResASig0K0=*/   0.119,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.055, /*kResASigStMK0=*/ 0.053,
/*kResSigStMK0=*/ 0.049, /*kResASigStPK0=*/ 0.052,
/*kResSigSt0K0=*/ 0.049, /*kResASigSt0K0=*/ 0.046,

/*kResLamKSt0ToLamK0=*/  0.021, /*kResALamAKSt0ToALamK0=*/  0.021,
/*kResSig0KSt0ToLamK0=*/ 0.018, /*kResASig0AKSt0ToALamK0=*/ 0.018,
/*kResXi0KSt0ToLamK0=*/  0.006, /*kResAXi0AKSt0ToALamK0=*/  0.006,
/*kResXiCKSt0ToLamK0=*/  0.008, /*kResAXiCAKSt0ToALamK0=*/  0.008
};


const double *cAnalysisLambdaFactors_10Res[7] = {cAnalysisLambdaFactors_10Res_MaxPrimDecay0, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay4, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay5, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay6, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay10, 
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay100,
                                                 cAnalysisLambdaFactors_10Res_MaxPrimDecay15};

//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------



//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=0 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay0[85] = {
/*kLamK0=*/   0.142,   /*kALamK0=*/   0.142,
/*kLamKchP=*/ 0.130,   /*kALamKchM=*/ 0.128,  /*kLamKchM=*/ 0.129, /*kALamKchP=*/ 0.129,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.116, /*kResASig0KchM=*/   0.112, /*kResSig0KchM=*/   0.115, /*kResASig0KchP=*/  0.113,
/*kResXi0KchP=*/    0.040, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.040, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.052, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.127, /*kResASig0K0=*/   0.123,
/*kResXi0K0=*/    0.044, /*kResAXi0K0=*/    0.040,
/*kResXiCK0=*/    0.057, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=4 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay4[85] = {
/*kLamK0=*/   0.193,   /*kALamK0=*/   0.190,
/*kLamKchP=*/ 0.179,   /*kALamKchM=*/ 0.175,  /*kLamKchM=*/ 0.178, /*kALamKchP=*/ 0.176,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.115, /*kResASig0KchM=*/   0.111, /*kResSig0KchM=*/   0.114, /*kResASig0KchP=*/  0.112,
/*kResXi0KchP=*/    0.040, /*kResAXi0KchM=*/    0.036, /*kResXi0KchM=*/    0.039, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.052, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.051, /*kResAXiCKchP=*/   0.047,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.126, /*kResASig0K0=*/   0.123,
/*kResXi0K0=*/    0.043, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.056, /*kResAXiCK0=*/    0.051,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=5 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay5[85] = {
/*kLamK0=*/   0.271,   /*kALamK0=*/   0.269,
/*kLamKchP=*/ 0.259,   /*kALamKchM=*/ 0.256,  /*kLamKchM=*/ 0.259, /*kALamKchP=*/ 0.257,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.113, /*kResASig0KchM=*/   0.110, /*kResSig0KchM=*/   0.112, /*kResASig0KchP=*/  0.111,
/*kResXi0KchP=*/    0.039, /*kResAXi0KchM=*/    0.035, /*kResXi0KchM=*/    0.039, /*kResAXi0KchP=*/   0.036,
/*kResXiCKchP=*/    0.051, /*kResAXiCKchM=*/    0.046, /*kResXiCKchM=*/    0.050, /*kResAXiCKchP=*/   0.046,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.124, /*kResASig0K0=*/   0.121,
/*kResXi0K0=*/    0.043, /*kResAXi0K0=*/    0.039,
/*kResXiCK0=*/    0.056, /*kResAXiCK0=*/    0.050,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=6 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay6[85] = {
/*kLamK0=*/   0.506,   /*kALamK0=*/   0.509,
/*kLamKchP=*/ 0.485,   /*kALamKchM=*/ 0.485,  /*kLamKchM=*/ 0.484, /*kALamKchP=*/ 0.486,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.108, /*kResASig0KchM=*/   0.107, /*kResSig0KchM=*/   0.107, /*kResASig0KchP=*/  0.108,
/*kResXi0KchP=*/    0.038, /*kResAXi0KchM=*/    0.034, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.049, /*kResAXiCKchM=*/    0.044, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.118, /*kResASig0K0=*/   0.118,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=10 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay10[85] = {
/*kLamK0=*/   0.531,   /*kALamK0=*/   0.532,
/*kLamKchP=*/ 0.509,   /*kALamKchM=*/ 0.509,  /*kLamKchM=*/ 0.509, /*kALamKchP=*/ 0.510,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.108, /*kResASig0KchM=*/   0.107, /*kResSig0KchM=*/   0.107, /*kResASig0KchP=*/  0.108,
/*kResXi0KchP=*/    0.037, /*kResAXi0KchM=*/    0.034, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.048, /*kResAXiCKchM=*/    0.044, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.118, /*kResASig0K0=*/   0.118,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=100 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay100[85] = {
/*kLamK0=*/   0.555,   /*kALamK0=*/   0.557,
/*kLamKchP=*/ 0.547,   /*kALamKchM=*/ 0.548,  /*kLamKchM=*/ 0.547, /*kALamKchP=*/ 0.548,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.107, /*kResASig0KchM=*/   0.106, /*kResSig0KchM=*/   0.106, /*kResASig0KchP=*/  0.107,
/*kResXi0KchP=*/    0.037, /*kResAXi0KchM=*/    0.034, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.034,
/*kResXiCKchP=*/    0.048, /*kResAXiCKchM=*/    0.044, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.044,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.117, /*kResASig0K0=*/   0.117,
/*kResXi0K0=*/    0.040, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.052, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};

//------------ WHEN INCLUDING 3 RESIDUALS and MaxDecayLengthPrimary=15 ------------------------------------//
const double cAnalysisLambdaFactors_3Res_MaxPrimDecay15[85] = {
/*kLamK0=*/   0.537,   /*kALamK0=*/   0.538,
/*kLamKchP=*/ 0.516,   /*kALamKchM=*/ 0.515,  /*kLamKchM=*/ 0.515, /*kALamKchP=*/ 0.516,
/*kXiKchP=*/  1.000,   /*kAXiKchM=*/  1.000,  /*kXiKchM=*/  1.000, /*kAXiKchP=*/  1.000,
/*kXiK0=*/    1.000,   /*kAXiK0=*/    1.000,
/*kLamLam=*/  1.000,   /*kALamALam=*/ 1.000,  /*kLamALam=*/ 1.000,
/*kLamPiP=*/  1.000,   /*kALamPiM=*/  1.000,  /*kLamPiM=*/  1.000, /*kALamPiP=*/  1.000,

//----- Residual Types -----
/*kResSig0KchP=*/   0.108, /*kResASig0KchM=*/   0.107, /*kResSig0KchM=*/   0.107, /*kResASig0KchP=*/  0.107,
/*kResXi0KchP=*/    0.037, /*kResAXi0KchM=*/    0.034, /*kResXi0KchM=*/    0.037, /*kResAXi0KchP=*/   0.035,
/*kResXiCKchP=*/    0.048, /*kResAXiCKchM=*/    0.044, /*kResXiCKchM=*/    0.048, /*kResAXiCKchP=*/   0.045,
/*kResOmegaKchP=*/  0.000, /*kResAOmegaKchM=*/  0.000, /*kResOmegaKchM=*/  0.000, /*kResAOmegaKchP=*/ 0.000,
/*kResSigStPKchP=*/ 0.000, /*kResASigStMKchM=*/ 0.000, /*kResSigStPKchM=*/ 0.000, /*kResASigStMKchP=*/ 0.000,
/*kResSigStMKchP=*/ 0.000, /*kResASigStPKchM=*/ 0.000, /*kResSigStMKchM=*/ 0.000, /*kResASigStPKchP=*/ 0.000,
/*kResSigSt0KchP=*/ 0.000, /*kResASigSt0KchM=*/ 0.000, /*kResSigSt0KchM=*/ 0.000, /*kResASigSt0KchP=*/ 0.000,

/*kResLamKSt0=*/  0.000, /*kResALamAKSt0=*/  0.000, /*kResLamAKSt0=*/  0.000, /*kResALamKSt0=*/  0.000,
/*kResSig0KSt0=*/ 0.000, /*kResASig0AKSt0=*/ 0.000, /*kResSig0AKSt0=*/ 0.000, /*kResASig0KSt0=*/ 0.000,
/*kResXi0KSt0=*/  0.000, /*kResAXi0AKSt0=*/  0.000, /*kResXi0AKSt0=*/  0.000, /*kResAXi0KSt0=*/  0.000,
/*kResXiCKSt0=*/  0.000, /*kResAXiCAKSt0=*/  0.000, /*kResXiCAKSt0=*/  0.000, /*kResAXiCKSt0=*/  0.000,

/*kResSig0K0=*/   0.117, /*kResASig0K0=*/   0.117,
/*kResXi0K0=*/    0.041, /*kResAXi0K0=*/    0.038,
/*kResXiCK0=*/    0.053, /*kResAXiCK0=*/    0.049,
/*kResOmegaK0=*/  0.000, /*kResAOmegaK0=*/  0.000,
/*kResSigStPK0=*/ 0.000, /*kResASigStMK0=*/ 0.000,
/*kResSigStMK0=*/ 0.000, /*kResASigStPK0=*/ 0.000,
/*kResSigSt0K0=*/ 0.000, /*kResASigSt0K0=*/ 0.000,

/*kResLamKSt0ToLamK0=*/  0.000, /*kResALamAKSt0ToALamK0=*/  0.000,
/*kResSig0KSt0ToLamK0=*/ 0.000, /*kResASig0AKSt0ToALamK0=*/ 0.000,
/*kResXi0KSt0ToLamK0=*/  0.000, /*kResAXi0AKSt0ToALamK0=*/  0.000,
/*kResXiCKSt0ToLamK0=*/  0.000, /*kResAXiCAKSt0ToALamK0=*/  0.000
};


const double *cAnalysisLambdaFactors_3Res[7] = {cAnalysisLambdaFactors_3Res_MaxPrimDecay0, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay4, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay5, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay6, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay10, 
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay100,
                                                cAnalysisLambdaFactors_3Res_MaxPrimDecay15};


//----------------------------------------------------------------------------------------------------------------
//****************************************************************************************************************
//----------------------------------------------------------------------------------------------------------------

const double** cAnalysisLambdaFactorsArr[3] = {cAnalysisLambdaFactors_NoRes, cAnalysisLambdaFactors_10Res, cAnalysisLambdaFactors_3Res};

