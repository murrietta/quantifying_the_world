/*
Analyze this data set using multiple imputation.
Use PROC MI to discover the missing values patterns and to decide what MI options to use. (Assume no need for transformations.)
Use PROC MI to create multiple imputed data sets.
Use PROC REG to analyze the multiple data sets while outputting information to be used in MIANALYZE.
Use PROC MIANALYZE to summarize the imputed analyses.
Compare these results to the listwise deletion results.
*/

/* reading in the data */
data carmpg;
   infile 'C:\Users\m\Documents\SMU\qtw\hw01\carmpgdata_2.txt' dlm='09'x dsd truncover firstobs=2;
   input Auto $ MPG CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
 /* proc print data=carmpg; run; */
/* The professor explained a more simple basic model but since we were given more variables we will use
   what was given to us */
TITLE 'Predicting MPG';
ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\initial_model.rtf";
PROC REG DATA = carmpg;
  MODEL MPG = CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
RUN;
QUIT;
ods rtf close;
/* look at some histograms of the data */
proc univariate data=carmpg;
histogram;
run;
proc corr data=carmpg COV plots(maxpoints=NONE)=matrix(histogram);
   var MPG CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
   ods select MatrixPlot;
run;
/* look at the missing data patterns to figure out which MI method to use, we use
   NIMPUTE=0 to not perform any imputation but just provide the missing data patterns*/
ODS SELECT MISSPATTERN;

ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\missing_pattern.rtf";
PROC MI DATA=carmpg NIMPUTE=0 simple seed = 35399;
*em itprint outem=outem;
class CYLINDERS ENG_TYPE;
VAR MPG CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
FCS;
RUN;
ods rtf close;
ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\ods_multiple_imputation_10.rtf";
PROC MI DATA=carmpg NIMPUTE=10 out = miout seed = 35399;
class CYLINDERS ENG_TYPE;
VAR MPG CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
*MCMC initial=em (bootstrap=100) displayinit;
FCS;
RUN;
ods rtf close;
ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\ods_multiple_imputation_5.rtf";
PROC MI DATA=carmpg NIMPUTE=5 out = miout seed = 35399;
class CYLINDERS ENG_TYPE;
VAR MPG CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
*MCMC initial=em (bootstrap=100) displayinit;
FCS;
RUN;
ods rtf close;
/*proc print data=miout; run;*/

/* Try proc reg with the imputed data, create output dataset to use in MIANALYZE */
ods graphics on;
ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\ods_imputed_models.rtf";
PROC REG DATA = miout outest=outreg covout plots(label)=(CooksD RStudentByLeverage DFFITS DFBETAS);
   MODEL MPG = CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
   by _Imputation_;
RUN;
quit;
ods rtf close;
proc print data=outreg; run;
/* this will output the combined estimate from all imputation runs, we should
   get the original estimates from the first (non-imputed) proc reg and compare
   side-by-side like in the video*/
ods rtf file="C:\Users\m\Documents\SMU\qtw\hw01\ods_mi_analyze_10.rtf";
PROC MIANALYZE data = outreg;
    MODELEFFECTS Intercept CYLINDERS SIZE HP WEIGHT ACCEL ENG_TYPE;
RUN;
ods rtf close;
/*in class breakout work-----------------------------------------------------------------------*/
data carmpg2;
   infile 'C:\Users\m\Documents\SMU\qtw\week02\group2.txt' dlm=',' dsd truncover firstobs=2;
   input MPG cylinders Displacement HP Weight Acceleration Year Origin Name $;
proc print data=carmpg2;run;
ODS SELECT MISSPATTERN;
PROC MI DATA=carmpg2 NIMPUTE=0;
VAR MPG cylinders Displacement HP Weight Acceleration Year Origin;
RUN;
ODS SELECT MISSPATTERN;
PROC MI DATA=carmpg2 NIMPUTE=0 out = mimpg;
class Origin cylinders Year;
VAR MPG Displacement HP Weight Acceleration Origin cylinders Year;
FCS;
RUN;
/*--------------------------------------------------------------------------------------------*/
