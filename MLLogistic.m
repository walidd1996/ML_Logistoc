clc
clear 
close all

%excel = datastore('house_data_complete.csv','TreatAsMissing','NA',.....
   %'MissingValue',0,'ReadSize',21613);
    % excel=importdata('house_data_complete.csv');
excel=importdata('house_data_complete.csv');
h1=excel.data(:,1:2);
h2=excel.textdata(2:end ,3:17);
   

alpha=0.001;

for i= 1:15
    if iscell(h2(:,i))
      h3(:,i)= str2double(h2(:,i));
         
end
end
h4=[h3 h1];
for i=1:size(h4,2)
    h4(:,i)=h4(:,i)/max(h4(:,i));
end



m=floor(0.6*length(h4(:,1)));
n=floor(0.2*length(h4(:,1)));

train_linear2=h4(1:m,2:5);
train_linear=train_linear2/max(train_linear2);
train_linear = [ones(m,1) train_linear];




y_train2=h4(1:m,1);
y_train=y_train2./max(y_train2);

alpha=0.001;
theata_linear=zeros(size(train_linear,2),1);

cv_linear = h4(m+1:m+n,2:5);
cv_linear = cv_linear/max(cv_linear);
cv_linear = [ones(n,1) cv_linear];

j_linear_train=[];
j_linear_cv=[];

y_cv=h4(m+1:m+n,1);
y_cv=y_cv/max(y_cv);

 %%%% linear%%
for i=1:5000
    h=1./(1+exp(-train_linear*theata_linear));
    
    sum_cost =(train_linear'*(h-y_train))./m;
    theata_linear=theata_linear-alpha*(sum_cost);   
    h = train_linear*theata_linear;
    j_linear_train(i)=mean((-y_train.*log10(h))-((1-y_train).*log10(1-h)))/2;
    
end
figure('name','linear_train')
i=1:5000;
plot(i, j_linear_train);


for i=1:5000 
    h=1./(1+exp(-cv_linear*theata_linear));
   sum_cost =(cv_linear'*(h-y_cv))./n;
      theata_linear=theata_linear-alpha*(sum_cost);  
   j_linear_cv(i)=mean((-y_cv.*log10(h))-((1-y_cv).*log10(1-h)))/2;
    
end
figure('name','linear_cv')
i=1:5000;
plot(i, j_linear_cv);

 %%%quad
 
 train_quad2=h4(1:m,6:9);
 train_quad= [ones(m,1)  train_quad2 train_quad2.^2 ];

 theata_quad=zeros(size(train_quad,2),1);

cv_q = h4(m+1:m+n,6:9);

cv_quad= [ones(n,1) cv_q cv_q.^2];

j_quad_train=[];
j_quad_cv=[];

for i=1:5000
    h=1./(1+exp(-train_quad*theata_quad));
    
    sum_cost =(train_quad'*(h-y_train))./m;
    theata_quad=theata_quad-alpha*(sum_cost);   
    h = train_quad*theata_quad;
    j_quad_train(i)=mean((-y_train.*log10(h))-((1-y_train).*log10(1-h)))/2;
end
    i=1:5000;
figure('name','quad_train')

plot(i,j_quad_train)
    

%%%%%%%%%%%%%


for i=1:5000  
     h=1./(1+exp(-cv_quad*theata_quad));
    
    sum_cost =(cv_quad'*(h-y_cv))./n;
    theata_quad=theata_quad-alpha*(sum_cost);   
    h = cv_quad*theata_quad;
    j_quad_cv(i)=mean((-y_cv.*log10(h))-((1-y_cv).*log10(1-h)))/2;
end

i=1:5000;
figure('name','quad_cv')

plot(i,j_quad_cv)

 
%%%% cube
train_cube2=h4(1:m,10:13);

train_cube= [ones(m,1)  train_cube2  train_cube2.^2 train_cube2.^3 ];

 theata_cube=zeros(size(train_cube,2),1);

 cv_cube2=h4(m+1:m+n ,10:13);

cv_cube= [ones(n,1) cv_cube2 cv_cube2.^2 cv_cube2.^3];
j_cube_train=[];
j_cube_cv=[];

for i=1:5000
    
    h=1./(1+exp(-train_cube*theata_cube));
    
    sum_cost =(train_cube'*(h-y_train))./m;
    theata_cube=theata_cube-alpha*(sum_cost);   
    h = train_cube*theata_cube;
    j_cube_train(i)=mean((-y_train.*log10(h))-((1-y_train).*log10(1-h)))/2;
     
end

for i=1:5000  
     h=1./(1+exp(-cv_cube*theata_cube));
    
    sum_cost =(cv_cube'*(h-y_cv))./n;
    theata_cube=theata_cube-alpha*(sum_cost);   
    h = cv_cube*theata_cube;
    j_cube_cv(i)=mean((-y_cv.*log10(h))-((1-y_cv).*log10(1-h)))/2;
     
end
i=1:5000;
figure('name','cube_cv')

plot(i,j_cube_cv)
figure('name','cube_train')

 plot(i,j_cube_train)
 
 
 
 
 %%%qad%%%
 
train_qad2=h4(1:m,14:17);

train_qad= [ones(m,1)  train_qad2  train_qad2.^2 train_qad2.^3   train_qad2.^4 ];

 theata_qad=zeros(size(train_qad,2),1);

 cv_qad2=h4(m+1:m+n ,14:17);

cv_qad= [ones(n,1) cv_qad2 cv_qad2.^2 cv_qad2.^3 cv_qad2.^4];
j_qad_train=[];
j_qad_cv=[];

 
 for i=1:5000
    
    h=1./(1+exp(-train_qad*theata_qad));
    
    sum_cost =(train_qad'*(h-y_train))./m;
    theata_qad=theata_qad-alpha*(sum_cost);   
    h = train_qad*theata_qad;
    j_qad_train(i)=mean((-y_train.*log10(h))-((1-y_train).*log10(1-h)))/2;
     
 end

 

for i=1:5000  
     h=1./(1+exp(-cv_qad*theata_qad));
    
    sum_cost =(cv_qad'*(h-y_cv))./n;
    theata_qad=theata_qad-alpha*(sum_cost);   
    h = cv_qad*theata_qad;
    j_qad_cv(i)=mean((-y_cv.*log10(h))-((1-y_cv).*log10(1-h)))/2;
     
   
end
i=1:5000;
figure('name','qad_cv')

plot(i,j_qad_cv)
figure('name','qad_train')

 plot(i,j_qad_train)
 
 j_test=[];
 
 if j_linear_train(4999)-j_linear_cv<0.05
     
  test_data_linear2=h4(m+n+1:m+n+n,2:5);
  %test_data_linear=[ones(n,1),test_data_linear2];
  %y_test=h4(m+n+1:m+n+n,1);
     
   % h = theata_linear*test_data_linear;
    % j_test=mean((h-y_test).^2);  
 end
 
 

 if j_quad_train(4999)-j_quad_cv<0.05
     
  test_data_quad2=h4(m+n+1:end,6:9);
 % test_data_quad=[ones(n,1),test_data_quad2];
  %y_test=h4(m+n+1:m+n+n,1);
     
   % h = test_data_quad*theata_quad;
   %  j_test=mean((h-y_test).^2);  
 end
 
 if j_cube_train(4999)-j_cube_cv<0.05
     
  test_data_cube=h4(m+n+1:m+n+n,10:13);
 % y_test=h4(m+n+1:m+n+n,1);
     
  %  h = test_data_cube*theata_cube;
   %  j_test=mean((h-y_test).^2);  
 end
 
 
 if j_qad_train(4999)-j_qad_cv<0.05
     
  %test_data_cube=h4(m+n+1:m+n+n,14:17);
  %y_test=h4(m+n+1:m+n+n,1);
     
    %h = test_data_qad*theata_qad;
   %  j_test=mean((h-y_test).^2);  
 end
 
 
  
   
 
 
 
 
 
 
 
 
 
 
 
 
 
