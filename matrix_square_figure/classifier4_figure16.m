clc;
clear;
close all;
warning off;

%ED-ELM
%case1_session1
matrix_A=[469 154 100;257 502 178;174 244 622];
matrix_B=[677 216 104;122 440 315;101 244 481];
matrix_C=[411 38 87;451 822 313;38 40 500];
matrix_D=[263 167 154;9 400 11;628 333 735];
matrix_E=[509 10 143;11 607 12;380 283 745];
matrix_F=[452 7 329;7 459 12;441 434 559];
matrix_G=[700 346 443;190 533 12;10 21 445];
matrix_H=[542 176 267;265 700 152;93 24 481];

%case1_session2
matrix_A_1=[239 99 120;534 735 67;127 66 713];
matrix_B_1=[527 92 117;155 468 228;218 340 555];
matrix_C_1=[313 243 189;6 650 9;581 7 702];
matrix_D_1=[471 3 2;423 467 377;6 430 521];
matrix_E_1=[658 268 325;233 629 10;9 3 565];
matrix_F_1=[426 266 223;166 501 93;308 133 584];
matrix_G_1=[359 225 30;491 633 159;50 42 711];
matrix_H_1=[669 15 21;23 754 479;208 131 400];

%case2_session1
matrix_I=[526 313 151;2 120 5;52 2 279];
matrix_J=[489 322 228;5 109 7;86 4 200];
matrix_K=[400 60 17;110 335 96;70 40 322];
matrix_L=[397 52 62;14 300 44;169 83 329];
matrix_M=[452 27 43;87 297 178;41 111 214];
matrix_N=[300 145 24;109 211 61;171 79 350];

%case2_session2
matrix_I_1=[400 144 45;104 240 90;76 51 300];
matrix_J_1=[371 40 45;55 263 202;154 132 188];
matrix_K_1=[470 14 138;20 320 19;90 101 278];
matrix_L_1=[435 19 75;86 376 127;59 40 233];
matrix_M_1=[436 16 10;42 258 171;102 161 254];
matrix_N_1=[498 12 12;69 249 282;13 174 141];

%»­Í¼
%case1_session1
ED_ELM_case1_session1=matrix_A+matrix_B+matrix_C+matrix_D+matrix_E+matrix_F+matrix_G+matrix_H;
ED_ELM_case1_session1=ED_ELM_case1_session1./7200*100;
name={'L','M','H'};
matrixplot(ED_ELM_case1_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
caxis([0,100]);

%case1_session2
ED_ELM_case1_session2=matrix_A_1+matrix_B_1+matrix_C_1+matrix_D_1+matrix_E_1+matrix_F_1+matrix_G_1+matrix_H_1;
ED_ELM_case1_session2=ED_ELM_case1_session2./7200*100;
name={'L','M','H'};
matrixplot(ED_ELM_case1_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
caxis([0,100]);

%case2_session1
ED_ELM_case2_session1=matrix_I+matrix_J+matrix_K+matrix_L+matrix_M+matrix_N;
ED_ELM_case2_session1(:,1)=ED_ELM_case2_session1(:,1)./3480*100;
ED_ELM_case2_session1(:,2:3)=ED_ELM_case2_session1(:,2:3)./2610*100;
name={'L','M','H'};
matrixplot(ED_ELM_case2_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
caxis([0,100]);

%case2_session2
ED_ELM_case2_session2=matrix_I_1+matrix_J_1+matrix_K_1+matrix_L_1+matrix_M_1+matrix_N_1;
ED_ELM_case2_session2(:,1)=ED_ELM_case2_session2(:,1)./3480*100;
ED_ELM_case2_session2(:,2:3)=ED_ELM_case2_session2(:,2:3)./2610*100;
name={'L','M','H'};
matrixplot(ED_ELM_case2_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
caxis([0,100]);

% %ELM
% %case1_session1
% matrix_A=[525 330 196;318 321 123;57 249 581];
% matrix_B=[659 243 126;178 172 298;63 485 476];
% matrix_C=[497 438 248;142 300 254;261 162 398];
% matrix_D=[397 408 415;282 199 277;221 293 208];
% matrix_E=[767 235 117;97 285 209;36 380 574];
% matrix_F=[448 354 273;220 174 275;232 372 352];
% matrix_G=[646 334 155;123 174 399;131 392 346];
% matrix_H=[732 132 127;84 257 380;84 511 393];
% 
% %case1_session2
% matrix_A_1=[471 303 237;240 226 341;189 371 322];
% matrix_B_1=[567 299 198;131 211 262;202 390 440];
% matrix_C_1=[501 432 183;209 230 333;190 238 384];
% matrix_D_1=[405 324 275;240 321 250;255 255 375];
% matrix_E_1=[587 221 263;156 291 336;157 388 301];
% matrix_F_1=[545 326 227;253 227 174;102 347 499];
% matrix_G_1=[478 332 214;255 209 241;167 359 445];
% matrix_H_1=[370 228 339;208 310 322;322 362 239];
% 
% %case2_session1
% matrix_I=[403 153 200;54 72 72;123 210 163];
% matrix_J=[365 336 289;75 43 55;140 56 91];
% matrix_K=[394 187 135;88 86 148;98 162 152];
% matrix_L=[418 200 158;75 84 114;87 151 163];
% matrix_M=[401 195 197;72 101 55;107 139 183];
% matrix_N=[326 283 165;137 44 69;117 108 201];
% 
% %case2_session2
% matrix_I_1=[392 161 191;63 95 90;125 179 154];
% matrix_J_1=[404 229 147;81 106 68;95 100 220];
% matrix_K_1=[485 166 84;76 91 82;19 178 269];
% matrix_L_1=[467 123 133;91 81 50;22 231 252];
% matrix_M_1=[404 318 116;84 67 95;92 50 224];
% matrix_N_1=[370 211 217;90 121 127;120 103 91];
% 
% %»­Í¼
% %case1_session1
% ELM_case1_session1=matrix_A+matrix_B+matrix_C+matrix_D+matrix_E+matrix_F+matrix_G+matrix_H;
% ELM_case1_session1=ELM_case1_session1./7200*100;
% name={'L','M','H'};
% matrixplot(ELM_case1_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case1_session2
% ELM_case1_session2=matrix_A_1+matrix_B_1+matrix_C_1+matrix_D_1+matrix_E_1+matrix_F_1+matrix_G_1+matrix_H_1;
% ELM_case1_session2=ELM_case1_session2./7200*100;
% name={'L','M','H'};
% matrixplot(ELM_case1_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session1
% ELM_case2_session1=matrix_I+matrix_J+matrix_K+matrix_L+matrix_M+matrix_N;
% ELM_case2_session1(:,1)=ELM_case2_session1(:,1)./3480*100;
% ELM_case2_session1(:,2:3)=ELM_case2_session1(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(ELM_case2_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session2
% ELM_case2_session2=matrix_I_1+matrix_J_1+matrix_K_1+matrix_L_1+matrix_M_1+matrix_N_1;
% ELM_case2_session2(:,1)=ELM_case2_session2(:,1)./3480*100;
% ELM_case2_session2(:,2:3)=ELM_case2_session2(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(ELM_case2_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);

% %LSSVM
% %case1_session1
% matrix_A=[459 290 191;279 352 243;162 258 466];
% matrix_B=[584 242 93;290 203 304;26 455 503];
% matrix_C=[423 351 216;202 433 366;275 116 318];
% matrix_D=[286 241 265;414 472 510;200 187 125];
% matrix_E=[754 181 105;132 355 184;14 364 611];
% matrix_F=[394 241 198;277 346 400;229 313 302];
% matrix_G=[524 290 111;200 269 534;176 341 255];
% matrix_H=[811 48 46;58 334 469;31 518 385];
% 
% %case1_session2
% matrix_A_1=[386 268 205;295 248 386;219 384 309];
% matrix_B_1=[604 275 131;139 341 326;157 284 443];
% matrix_C_1=[477 345 166;262 287 413;161 268 321];
% matrix_D_1=[352 238 264;278 416 301;270 246 335];
% matrix_E_1=[596 171 231;189 337 405;115 392 264];
% matrix_F_1=[569 274 194;244 261 179;87 365 527];
% matrix_G_1=[467 300 163;245 249 264;188 351 473];
% matrix_H_1=[335 192 327;291 394 395;274 314 178];
% 
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\LSSVM\case2_session1
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\LSSVM\case2_session2
% 
% %case2_session1
% matrix_I=m1_1;
% matrix_J=m1_2;
% matrix_K=m1_3;
% matrix_L=m1_4;
% matrix_M=m1_5;
% matrix_N=m1_6;
% 
% %case2_session2
% matrix_I_1=m1;
% matrix_J_1=m2;
% matrix_K_1=m3;
% matrix_L_1=m4;
% matrix_M_1=m5;
% matrix_N_1=m6;
% 
% %»­Í¼
% %case1_session1
% LSSVM_case1_session1=matrix_A+matrix_B+matrix_C+matrix_D+matrix_E+matrix_F+matrix_G+matrix_H;
% LSSVM_case1_session1=LSSVM_case1_session1./7200*100;
% name={'L','M','H'};
% matrixplot(LSSVM_case1_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case1_session2
% LSSVM_case1_session2=matrix_A_1+matrix_B_1+matrix_C_1+matrix_D_1+matrix_E_1+matrix_F_1+matrix_G_1+matrix_H_1;
% LSSVM_case1_session2=LSSVM_case1_session2./7200*100;
% name={'L','M','H'};
% matrixplot(LSSVM_case1_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session1
% LSSVM_case2_session1=matrix_I+matrix_J+matrix_K+matrix_L+matrix_M+matrix_N;
% LSSVM_case2_session1(:,1)=LSSVM_case2_session1(:,1)./3480*100;
% LSSVM_case2_session1(:,2:3)=LSSVM_case2_session1(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(LSSVM_case2_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session2
% LSSVM_case2_session2=matrix_I_1+matrix_J_1+matrix_K_1+matrix_L_1+matrix_M_1+matrix_N_1;
% LSSVM_case2_session2(:,1)=LSSVM_case2_session2(:,1)./3480*100;
% LSSVM_case2_session2(:,2:3)=LSSVM_case2_session2(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(LSSVM_case2_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);

% %SAE
% %case1_session1
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\SAE\case1_session1
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\SAE\case1_session2
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\SAE\case2_session1
% load F:\matlab\trial_procedure\study_1\matrix_square_figure\SAE\case2_session2
% 
% matrix_A=m11;
% matrix_B=m12;
% matrix_C=m13;
% matrix_D=m14;
% matrix_E=m15;
% matrix_F=m16;
% matrix_G=m17;
% matrix_H=m18;
% 
% %case1_session2
% matrix_A_1=m21;
% matrix_B_1=m22;
% matrix_C_1=m23;
% matrix_D_1=m24;
% matrix_E_1=m25;
% matrix_F_1=m26;
% matrix_G_1=m27;
% matrix_H_1=m28;
% 
% %case2_session1
% matrix_I=m31;
% matrix_J=m32;
% matrix_K=m33;
% matrix_L=m34;
% matrix_M=m35;
% matrix_N=m36;
% 
% %case2_session2
% matrix_I_1=m41;
% matrix_J_1=m42;
% matrix_K_1=m43;
% matrix_L_1=m44;
% matrix_M_1=m45;
% matrix_N_1=m46;
% 
% %»­Í¼
% %case1_session1
% SAE_case1_session1=matrix_A+matrix_B+matrix_C+matrix_D+matrix_E+matrix_F+matrix_G+matrix_H;
% SAE_case1_session1=SAE_case1_session1./7200*100;
% name={'L','M','H'};
% matrixplot(SAE_case1_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case1_session2
% SAE_case1_session2=matrix_A_1+matrix_B_1+matrix_C_1+matrix_D_1+matrix_E_1+matrix_F_1+matrix_G_1+matrix_H_1;
% SAE_case1_session2=SAE_case1_session2./7200*100;
% name={'L','M','H'};
% matrixplot(SAE_case1_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session1
% SAE_case2_session1=matrix_I+matrix_J+matrix_K+matrix_L+matrix_M+matrix_N;
% SAE_case2_session1(:,1)=SAE_case2_session1(:,1)./3480*100;
% SAE_case2_session1(:,2:3)=SAE_case2_session1(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(SAE_case2_session1,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
% %case2_session2
% SAE_case2_session2=matrix_I_1+matrix_J_1+matrix_K_1+matrix_L_1+matrix_M_1+matrix_N_1;
% SAE_case2_session2(:,1)=SAE_case2_session2(:,1)./3480*100;
% SAE_case2_session2(:,2:3)=SAE_case2_session2(:,2:3)./2610*100;
% name={'L','M','H'};
% matrixplot(SAE_case2_session2,'XVarNames',name,'YVarNames',name,'ColorBar','on');
% caxis([0,100]);
% 
