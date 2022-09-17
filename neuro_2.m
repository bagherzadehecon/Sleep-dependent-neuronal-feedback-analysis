clc
clear 
close all
%% part(2)
[state1,t1,X1] =  FeatureExtraction('ST7011J0-PSG.edf','ST7011JP-Hypnogram_annotations.txt');
[state2,t2,X2] =  FeatureExtraction('ST7022J0-PSG.edf','ST7022JM-Hypnogram_annotations.txt');
[state3,t3,X3] =  FeatureExtraction('ST7041J0-PSG.edf','ST7041JO-Hypnogram_annotations.txt');
[state4,t4,X4] =  FeatureExtraction('ST7052J0-PSG.edf','ST7052JA-Hypnogram_annotations.txt');
[state5,t5,X5] =  FeatureExtraction('ST7061J0-PSG.edf','ST7061JR-Hypnogram_annotations.txt');

%% part(3)
[coeff1,score1,latent1] = pca(X1);
 
cumulative=zeros(1,10);
for j=1:10
    cumulative(j)=sum(latent1(1:j));
end
figure(1)
cumulative=cumulative/sum(latent1)*100;
plot(cumulative)

xlabel('N')
ylabel('Percentage of Var.')
title('subject1')

%{
    0: W
    1: 1
    2: 2
    3: 3
    4: 4
    5: M
    6: R
%}

v_W=find(state1==0);
v_1=find(state1==1);
v_2=find(state1==2);
v_3=find(state1==3);
v_4=find(state1==4);
v_R=find(state1==6);

m_W=score1(v_W,1:3);
m_1=score1(v_1,1:3);
m_2=score1(v_2,1:3);
m_3=score1(v_3,1:3);
m_4=score1(v_4,1:3);
m_R=score1(v_R,1:3);

figure(2)
plot3(m_W(:,1),m_W(:,2),m_W(:,3),'.');
hold on
plot3(m_1(:,1),m_1(:,2),m_1(:,3),'.');
hold on
plot3(m_2(:,1),m_2(:,2),m_2(:,3),'.');
hold on
plot3(m_3(:,1),m_3(:,2),m_3(:,3),'.');
hold on
plot3(m_4(:,1),m_4(:,2),m_4(:,3),'.');
hold on
plot3(m_R(:,1),m_R(:,2),m_R(:,3),'.');
legend('W','1','2','3','4','R');
title('subject1')

% second
[coeff2,score2,latent2] = pca(X2);
    
cumulative=zeros(1,10);
for j=1:10
    cumulative(j)=sum(latent2(1:j));
end
figure(3)
cumulative=cumulative/sum(latent2)*100;
plot(cumulative)

xlabel('N')
ylabel('Percentage of Var.')
title('subject2')

%{
    0: W
    1: 1
    2: 2
    3: 3
    4: 4
    5: M
    6: R
%}

v_W=find(state2==0);
v_1=find(state2==1);
v_2=find(state2==2);
v_3=find(state2==3);
v_4=find(state2==4);
v_R=find(state2==6);

m_W=score2(v_W,1:3);
m_1=score2(v_1,1:3);
m_2=score2(v_2,1:3);
m_3=score2(v_3,1:3);
m_4=score2(v_4,1:3);
m_R=score2(v_R,1:3);

figure(4)
plot3(m_W(:,1),m_W(:,2),m_W(:,3),'.');
hold on
plot3(m_1(:,1),m_1(:,2),m_1(:,3),'.');
hold on
plot3(m_2(:,1),m_2(:,2),m_2(:,3),'.');
hold on
plot3(m_3(:,1),m_3(:,2),m_3(:,3),'.');
hold on
plot3(m_4(:,1),m_4(:,2),m_4(:,3),'.');
hold on
plot3(m_R(:,1),m_R(:,2),m_R(:,3),'.');
legend('W','1','2','3','4','R');
title('subject2')

%third
[coeff3,score3,latent3] = pca(X3);
    
cumulative=zeros(1,10);
for j=1:10
    cumulative(j)=sum(latent3(1:j));
end
figure(5)
cumulative=cumulative/sum(latent3)*100;
plot(cumulative)

xlabel('N')
ylabel('Percentage of Var.')
title('subject3')

%{
    0: W
    1: 1
    2: 2
    3: 3
    4: 4
    5: M
    6: R
%}

v_W=find(state3==0);
v_1=find(state3==1);
v_2=find(state3==2);
v_3=find(state3==3);
v_4=find(state3==4);
v_R=find(state3==6);

m_W=score3(v_W,1:3);
m_1=score3(v_1,1:3);
m_2=score3(v_2,1:3);
m_3=score3(v_3,1:3);
m_4=score3(v_4,1:3);
m_R=score3(v_R,1:3);

figure(6)
plot3(m_W(:,1),m_W(:,2),m_W(:,3),'.');
hold on
plot3(m_1(:,1),m_1(:,2),m_1(:,3),'.');
hold on
plot3(m_2(:,1),m_2(:,2),m_2(:,3),'.');
hold on
plot3(m_3(:,1),m_3(:,2),m_3(:,3),'.');
hold on
plot3(m_4(:,1),m_4(:,2),m_4(:,3),'.');
hold on
plot3(m_R(:,1),m_R(:,2),m_R(:,3),'.');
legend('W','1','2','3','4','R');
title('subject3')

%forth
[coeff4,score4,latent4] = pca(X4);
    
cumulative=zeros(1,10);
for j=1:10
    cumulative(j)=sum(latent4(1:j));
end
figure(7)
cumulative=cumulative/sum(latent4)*100;
plot(cumulative)

xlabel('N')
ylabel('Percentage of Var.')
title('subject3')

%{
    0: W
    1: 1
    2: 2
    3: 3
    4: 4
    5: M
    6: R
%}

v_W=find(state4==0);
v_1=find(state4==1);
v_2=find(state4==2);
v_3=find(state4==3);
v_4=find(state4==4);
v_R=find(state4==6);

m_W=score4(v_W,1:3);
m_1=score4(v_1,1:3);
m_2=score4(v_2,1:3);
m_3=score4(v_3,1:3);
m_4=score4(v_4,1:3);
m_R=score4(v_R,1:3);

figure(8)
plot3(m_W(:,1),m_W(:,2),m_W(:,3),'.');
hold on
plot3(m_1(:,1),m_1(:,2),m_1(:,3),'.');
hold on
plot3(m_2(:,1),m_2(:,2),m_2(:,3),'.');
hold on
plot3(m_3(:,1),m_3(:,2),m_3(:,3),'.');
hold on
plot3(m_4(:,1),m_4(:,2),m_4(:,3),'.');
hold on
plot3(m_R(:,1),m_R(:,2),m_R(:,3),'.');
legend('W','1','2','3','4','R');
title('subject4')

%fifth
[coeff5,score5,latent5] = pca(X5);
    
cumulative=zeros(1,10);
for j=1:10
    cumulative(j)=sum(latent5(1:j));
end
figure(9)
cumulative=cumulative/sum(latent5)*100;
plot(cumulative)

xlabel('N')
ylabel('Percentage of Var.')
title('subject5')

%{
    0: W
    1: 1
    2: 2
    3: 3
    4: 4
    5: M
    6: R
%}

v_W=find(state5==0);
v_1=find(state5==1);
v_2=find(state5==2);
v_3=find(state5==3);
v_4=find(state5==4);
v_R=find(state5==6);

m_W=score5(v_W,1:3);
m_1=score5(v_1,1:3);
m_2=score5(v_2,1:3);
m_3=score5(v_3,1:3);
m_4=score5(v_4,1:3);
m_R=score5(v_R,1:3);

figure(10)
plot3(m_W(:,1),m_W(:,2),m_W(:,3),'.');
hold on
plot3(m_1(:,1),m_1(:,2),m_1(:,3),'.');
hold on
plot3(m_2(:,1),m_2(:,2),m_2(:,3),'.');
hold on
plot3(m_3(:,1),m_3(:,2),m_3(:,3),'.');
hold on
plot3(m_4(:,1),m_4(:,2),m_4(:,3),'.');
hold on
plot3(m_R(:,1),m_R(:,2),m_R(:,3),'.');
legend('W','1','2','3','4','R');
title('subject5')



figure(11)

subplot(1,3,1)
stem (coeff1(1:10,1))
title('first component subj1')
subplot(1,3,2)
stem (coeff1(1:10,2))
title('second component subj1')
subplot(1,3,3)
stem (coeff1(1:10,3))
title('third component subj1')


figure(12)
title('subject 2')
subplot(1,3,1)
stem (coeff2(1:10,1))
title('first component subj2')
subplot(1,3,2)
stem (coeff2(1:10,2))
title('second component subj2')
subplot(1,3,3)
stem (coeff2(1:10,3))
title('third component subj2')

figure(13)
title('subject 3')
subplot(1,3,1)
stem (coeff3(1:10,1))
title('first component subj3')
subplot(1,3,2)
stem (coeff3(1:10,2))
title('second component subj3')
subplot(1,3,3)
stem (coeff3(1:10,3))
title('third component subj3')

figure(14)
title('subject 4')
subplot(1,3,1)
stem (coeff4(1:10,1))
title('first component subj4')
subplot(1,3,2)
stem (coeff4(1:10,2))
title('second component subj4')
subplot(1,3,3)
stem (coeff4(1:10,3))
title('third component subj4')

figure(15)
title('subject 5')
subplot(1,3,1)
stem (coeff5(1:10,1))
title('first component subj5')
subplot(1,3,2)
stem (coeff5(1:10,2))
title('second component subj5')
subplot(1,3,3)
stem (coeff5(1:10,3))
title('third component subj5')
%% part(4-1)
num_bin=find(state1>=1&state1<=4);
y=state1(num_bin)';
X_1=X1(num_bin,:);

lm1 = fitlm(X_1,y);

pre=lm1.Fitted;
y1=find(y==1);
y2=find(y==2);
y3=find(y==3);
y4=find(y==4);

h1=pre(y1);
h2=pre(y2);
h3=pre(y3);
h4=pre(y4);

figure(1)

histogram(h1,0:0.05:5,'normalization','pdf');
hold on
histogram(h2,0:0.05:5,'normalization','pdf');
hold on
histogram(h3,0:0.05:5,'normalization','pdf');
hold on
histogram(h4,0:0.05:5,'normalization','pdf');
legend('1','2','3','4')
title('subject1')

%second
num_bin=find(state2>=1&state2<=4);
y=state2(num_bin)';
X_1=X2(num_bin,:);

lm2 = fitlm(X_1,y);

pre=lm2.Fitted;
y1=find(y==1);
y2=find(y==2);
y3=find(y==3);
y4=find(y==4);

h1=pre(y1);
h2=pre(y2);
h3=pre(y3);
h4=pre(y4);

figure(2)

histogram(h1,0:0.05:5,'normalization','pdf');
hold on
histogram(h2,0:0.05:5,'normalization','pdf');
hold on
histogram(h3,0:0.05:5,'normalization','pdf');
hold on
histogram(h4,0:0.05:5,'normalization','pdf');
legend('1','2','3','4')
title('subject2')

%third
num_bin=find(state3>=1&state3<=4);
y=state3(num_bin)';
X_1=X3(num_bin,:);

lm3 = fitlm(X_1,y);

pre=lm3.Fitted;
y1=find(y==1);
y2=find(y==2);
y3=find(y==3);
y4=find(y==4);

h1=pre(y1);
h2=pre(y2);
h3=pre(y3);
h4=pre(y4);

figure(3)

histogram(h1,0:0.05:5,'normalization','pdf');
hold on
histogram(h2,0:0.05:5,'normalization','pdf');
hold on
histogram(h3,0:0.05:5,'normalization','pdf');
hold on
histogram(h4,0:0.05:5,'normalization','pdf');
legend('1','2','3','4')
title('subject3')

%fourth
num_bin=find(state4>=1&state4<=4);
y=state4(num_bin)';
X_1=X4(num_bin,:);

lm4 = fitlm(X_1,y);

pre=lm4.Fitted;
y1=find(y==1);
y2=find(y==2);
y3=find(y==3);
y4=find(y==4);

h1=pre(y1);
h2=pre(y2);
h3=pre(y3);
h4=pre(y4);

figure(4)

histogram(h1,0:0.05:5,'normalization','pdf');
hold on
histogram(h2,0:0.05:5,'normalization','pdf');
hold on
histogram(h3,0:0.05:5,'normalization','pdf');
hold on
histogram(h4,0:0.05:5,'normalization','pdf');
legend('1','2','3','4')
title('subject4')

%fifth
num_bin=find(state5>=1&state5<=4);
y=state5(num_bin)';
X_1=X5(num_bin,:);

lm5 = fitlm(X_1,y);

pre=lm5.Fitted;
y1=find(y==1);
y2=find(y==2);
y3=find(y==3);
y4=find(y==4);

h1=pre(y1);
h2=pre(y2);
h3=pre(y3);
h4=pre(y4);

figure(5)

histogram(h1,0:0.05:5,'normalization','pdf');
hold on
histogram(h2,0:0.05:5,'normalization','pdf');
hold on
histogram(h3,0:0.05:5,'normalization','pdf');
hold on
histogram(h4,0:0.05:5,'normalization','pdf');
legend('1','2','3','4')
title('subject5')
% we can see R-Squarred and p-value in lm 1-5.
lm1
lm2
lm3
lm4
lm5
%% part(4-2)
num_bin_new=find(state1==0|state1==6);
y_new=state1(num_bin_new)';
X_1_2=X1(num_bin_new,:);

ypred = predict(lm1,X_1_2);

% 0==>W

y0=find(y_new==0);

% 6==>R

y6=find(y_new==6);

h0=ypred(y0);
h6=ypred(y6);

figure(1)
histogram(h0,0:0.05:5,'Normalization','pdf');
hold on
histogram(h6,0:0.05:5,'Normalization','pdf');


legend('W','REM')
title('subject1')

%second
num_bin_new=find(state2==0|state2==6);
y_new=state2(num_bin_new)';
X_1_2=X2(num_bin_new,:);

ypred = predict(lm2,X_1_2);

% 0==>W

y0=find(y_new==0);

% 6==>R

y6=find(y_new==6);

h0=ypred(y0);
h6=ypred(y6);

figure(2)
histogram(h0,0:0.05:5,'Normalization','pdf');
hold on
histogram(h6,0:0.05:5,'Normalization','pdf');


legend('W','REM')


title('subject2')

%third
num_bin_new=find(state3==0|state3==6);
y_new=state3(num_bin_new)';
X_1_2=X3(num_bin_new,:);

ypred = predict(lm3,X_1_2);

% 0==>W

y0=find(y_new==0);

% 6==>R

y6=find(y_new==6);

h0=ypred(y0);
h6=ypred(y6);

figure(3)
histogram(h0,0:0.05:5,'Normalization','pdf');
hold on
histogram(h6,0:0.05:5,'Normalization','pdf');


legend('W','REM')


title('subject3')

%fourth
num_bin_new=find(state4==0|state4==6);
y_new=state4(num_bin_new)';
X_1_2=X4(num_bin_new,:);

ypred = predict(lm4,X_1_2);

% 0==>W

y0=find(y_new==0);

% 6==>R

y6=find(y_new==6);

h0=ypred(y0);
h6=ypred(y6);

figure(4)
histogram(h0,0:0.05:5,'Normalization','pdf');
hold on
histogram(h6,0:0.05:5,'Normalization','pdf');


legend('W','REM')


title('subject4')

%fifth
num_bin_new=find(state5==0|state5==6);
y_new=state5(num_bin_new)';
X_1_2=X5(num_bin_new,:);

ypred = predict(lm5,X_1_2);

% 0==>W

y0=find(y_new==0);

% 6==>R

y6=find(y_new==6);

h0=ypred(y0);
h6=ypred(y6);

figure(5)
histogram(h0,0:0.05:5,'Normalization','pdf');
hold on
histogram(h6,0:0.05:5,'Normalization','pdf');


legend('W','REM')


title('subject5')

%% part(5) 
% in this part we have used multisvm in order to train a system for
% clastering our data for states 1 to 4 .
% the accuracy mean was near 80 percent and the var was enough low.
state1new = state1(1:2800);
vec1=state1new==1;
vec11=find(vec1);

vec2=state1new==2;
vec22=find(vec2);

vec3=state1new==3;
vec33=find(vec3);

vec4=state1new==4;
vec44=find(vec4);

a1= sum(vec1);
a2= sum(vec2);
a3= sum(vec3);
a4= sum(vec4);
a = a1+a2+a3+a4;
Xnew1 = X1(1:2800,:);
Y1=zeros(a,10);
Y1(1:a1,:)=Xnew1(vec11,:);
Y1(a1+1:a1+a2,:)=Xnew1(vec22,:);
Y1(a1+a2+1:a1+a2+a3,:)=Xnew1(vec33,:);
Y1(a1+a2+a3+1:a,:)=Xnew1(vec44,:);

u1=zeros(a,1);
u1(1:a1)=1;
u1(a1+1:a1+a2)=2;
u1(a1+a2+1:a1+a2+a3)=3;
u1(a1+a2+a3+1:a)=4;
J=J_value2(Y1);

x1=(J(1,:)>0.01);
x2=(J(2,:)>0.01);
x3=(J(3,:)>0.01);
x4=(J(4,:)>0.01);
x5=(J(5,:)>0.01);
x6=(J(6,:)>0.01);
x_f=x1|x2|x3|x4|x5|x6;
num_col=find(x_f);
X_new_n=Y1(:,num_col);
categories = [1 ; 2 ; 3 ; 4];

X_n=X_new_n;
vecsPerCat = getVecsPerCat(X_n, u1, categories);
foldSizes = computeFoldSizes(vecsPerCat,10);
[X_sorted, u1_sorted] = randSortAndGroup(X_n, u1, categories);

accuracy_f=zeros(1,10);

for roundNumber = 1 : 10

[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, u1_sorted, categories, vecsPerCat, foldSizes, roundNumber);
[pediction] = Multi_SVM( X_train,X_val,y_train );
cmat = confusionmat(y_val,pediction);
acc = 100*sum(diag(cmat))./sum(cmat(:));
accuracy_f(roundNumber)=acc;

end
figure(1)
stem(1:10,accuracy_f)
ylabel('accuracy in percentage')
title('subject1 clastering accuracy')
var(accuracy_f)
mean(accuracy_f)

% subj2
state1new = state2(1:2773);
vec1=state1new==1;
vec11=find(vec1);

vec2=state1new==2;
vec22=find(vec2);

vec3=state1new==3;
vec33=find(vec3);

vec4=state1new==4;
vec44=find(vec4);

a1= sum(vec1);
a2= sum(vec2);
a3= sum(vec3);
a4= sum(vec4);
a = a1+a2+a3+a4;
Xnew1 = X2(1:2773,:);
Y1=zeros(a,10);
Y1(1:a1,:)=Xnew1(vec11,:);
Y1(a1+1:a1+a2,:)=Xnew1(vec22,:);
Y1(a1+a2+1:a1+a2+a3,:)=Xnew1(vec33,:);
Y1(a1+a2+a3+1:a,:)=Xnew1(vec44,:);

u1=zeros(a,1);
u1(1:a1)=1;
u1(a1+1:a1+a2)=2;
u1(a1+a2+1:a1+a2+a3)=3;
u1(a1+a2+a3+1:a)=4;
J=J_value2(Y1);

x1=(J(1,:)>0.01);
x2=(J(2,:)>0.01);
x3=(J(3,:)>0.01);
x4=(J(4,:)>0.01);
x5=(J(5,:)>0.01);
x6=(J(6,:)>0.01);
x_f=x1|x2|x3|x4|x5|x6;
num_col=find(x_f);
X_new_n=Y1(:,num_col);
categories = [1 ; 2 ; 3 ; 4];

X_n=X_new_n;
vecsPerCat = getVecsPerCat(X_n, u1, categories);
foldSizes = computeFoldSizes(vecsPerCat,10);
[X_sorted, u1_sorted] = randSortAndGroup(X_n, u1, categories);

accuracy_f=zeros(1,10);

for roundNumber = 1 : 10

[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, u1_sorted, categories, vecsPerCat, foldSizes, roundNumber);
[pediction] = Multi_SVM( X_train,X_val,y_train );
cmat = confusionmat(y_val,pediction);
acc = 100*sum(diag(cmat))./sum(cmat(:));
accuracy_f(roundNumber)=acc;

end
figure(2)
stem(1:10,accuracy_f)
ylabel('accuracy in percentage')
title('subject2 clastering accuracy')
var(accuracy_f)
mean(accuracy_f)


% subj3
state1new = state3(1:2800);
vec1=state1new==1;
vec11=find(vec1);

vec2=state1new==2;
vec22=find(vec2);

vec3=state1new==3;
vec33=find(vec3);

vec4=state1new==4;
vec44=find(vec4);

a1= sum(vec1);
a2= sum(vec2);
a3= sum(vec3);
a4= sum(vec4);
a = a1+a2+a3+a4;
Xnew1 = X3(1:2800,:);
Y1=zeros(a,10);
Y1(1:a1,:)=Xnew1(vec11,:);
Y1(a1+1:a1+a2,:)=Xnew1(vec22,:);
Y1(a1+a2+1:a1+a2+a3,:)=Xnew1(vec33,:);
Y1(a1+a2+a3+1:a,:)=Xnew1(vec44,:);

u1=zeros(a,1);
u1(1:a1)=1;
u1(a1+1:a1+a2)=2;
u1(a1+a2+1:a1+a2+a3)=3;
u1(a1+a2+a3+1:a)=4;
J=J_value2(Y1);

x1=(J(1,:)>0.01);
x2=(J(2,:)>0.01);
x3=(J(3,:)>0.01);
x4=(J(4,:)>0.01);
x5=(J(5,:)>0.01);
x6=(J(6,:)>0.01);
x_f=x1|x2|x3|x4|x5|x6;
num_col=find(x_f);
X_new_n=Y1(:,num_col);
categories = [1 ; 2 ; 3 ; 4];

X_n=X_new_n;
vecsPerCat = getVecsPerCat(X_n, u1, categories);
foldSizes = computeFoldSizes(vecsPerCat,10);
[X_sorted, u1_sorted] = randSortAndGroup(X_n, u1, categories);

accuracy_f=zeros(1,10);

for roundNumber = 1 : 10

[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, u1_sorted, categories, vecsPerCat, foldSizes, roundNumber);
[pediction] = Multi_SVM( X_train,X_val,y_train );
cmat = confusionmat(y_val,pediction);
acc = 100*sum(diag(cmat))./sum(cmat(:));
accuracy_f(roundNumber)=acc;

end
figure(3)
stem(1:10,accuracy_f)
ylabel('accuracy in percentage')
title('subject3 clastering accuracy')
var(accuracy_f)
mean(accuracy_f)


% subj4
state1new = state4(1:2800);
vec1=state1new==1;
vec11=find(vec1);

vec2=state1new==2;
vec22=find(vec2);

vec3=state1new==3;
vec33=find(vec3);

vec4=state1new==4;
vec44=find(vec4);

a1= sum(vec1);
a2= sum(vec2);
a3= sum(vec3);
a4= sum(vec4);
a = a1+a2+a3+a4;
Xnew1 = X4(1:2800,:);
Y1=zeros(a,10);
Y1(1:a1,:)=Xnew1(vec11,:);
Y1(a1+1:a1+a2,:)=Xnew1(vec22,:);
Y1(a1+a2+1:a1+a2+a3,:)=Xnew1(vec33,:);
Y1(a1+a2+a3+1:a,:)=Xnew1(vec44,:);

u1=zeros(a,1);
u1(1:a1)=1;
u1(a1+1:a1+a2)=2;
u1(a1+a2+1:a1+a2+a3)=3;
u1(a1+a2+a3+1:a)=4;
J=J_value2(Y1);

x1=(J(1,:)>0.01);
x2=(J(2,:)>0.01);
x3=(J(3,:)>0.01);
x4=(J(4,:)>0.01);
x5=(J(5,:)>0.01);
x6=(J(6,:)>0.01);
x_f=x1|x2|x3|x4|x5|x6;
num_col=find(x_f);
X_new_n=Y1(:,num_col);
categories = [1 ; 2 ; 3 ; 4];

X_n=X_new_n;
vecsPerCat = getVecsPerCat(X_n, u1, categories);
foldSizes = computeFoldSizes(vecsPerCat,10);
[X_sorted, u1_sorted] = randSortAndGroup(X_n, u1, categories);

accuracy_f=zeros(1,10);

for roundNumber = 1 : 10

[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, u1_sorted, categories, vecsPerCat, foldSizes, roundNumber);
[pediction] = Multi_SVM( X_train,X_val,y_train );
cmat = confusionmat(y_val,pediction);
acc = 100*sum(diag(cmat))./sum(cmat(:));
accuracy_f(roundNumber)=acc;

end
figure(4)
stem(1:10,accuracy_f)
ylabel('accuracy in percentage')
title('subject4 clastering accuracy')
var(accuracy_f)
mean(accuracy_f)

% subj5 this subj does not have state 4.So we just worked on state 1 to 3.
state1new = state5(1:2800);
vec1=state1new==1;
vec11=find(vec1);

vec2=state1new==2;
vec22=find(vec2);

vec3=state1new==3;
vec33=find(vec3);

vec4=state1new==4;
vec44=find(vec4);

a1= sum(vec1);
a2= sum(vec2);
a3= sum(vec3);
a4= sum(vec4);
a = a1+a2+a3+a4;
Xnew1 = X5(1:2800,:);
Y1=zeros(a,10);
Y1(1:a1,:)=Xnew1(vec11,:);
Y1(a1+1:a1+a2,:)=Xnew1(vec22,:);
Y1(a1+a2+1:a,:)=Xnew1(vec33,:);


u1=zeros(a,1);
u1(1:a1)=1;
u1(a1+1:a1+a2)=2;
u1(a1+a2+1:a1+a2+a3)=3;

J=J_value2(Y1);

x1=(J(1,:)>0.01);
x2=(J(2,:)>0.01);
x3=(J(3,:)>0.01);
x4=(J(4,:)>0.01);
x5=(J(5,:)>0.01);
x6=(J(6,:)>0.01);
x_f=x1|x2|x3|x4|x5|x6;
num_col=find(x_f);
X_new_n=Y1(:,num_col);
categories = [1 ; 2 ; 3 ];

X_n=X_new_n;
vecsPerCat = getVecsPerCat(X_n, u1, categories);
foldSizes = computeFoldSizes(vecsPerCat,10);
[X_sorted, u1_sorted] = randSortAndGroup(X_n, u1, categories);

accuracy_f=zeros(1,10);

for roundNumber = 1 : 10

[X_train, y_train, X_val, y_val] = getFoldVectors(X_sorted, u1_sorted, categories, vecsPerCat, foldSizes, roundNumber);
[pediction] = Multi_SVM( X_train,X_val,y_train );
cmat = confusionmat(y_val,pediction);
acc = 100*sum(diag(cmat))./sum(cmat(:));
accuracy_f(roundNumber)=acc;

end
figure(5)
stem(1:10,accuracy_f)
ylabel('accuracy in percentage')
title('subject5 clastering accuracy')
var(accuracy_f)
mean(accuracy_f)

