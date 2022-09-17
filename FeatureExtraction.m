function [state,t,X] =  FeatureExtraction(edfaddress,hypaddress)
Data = AnnotExtract(hypaddress);
[hdr, record1] = edfread(edfaddress);

% how many  10 second window

%filtering
%h = BPF(1001,0.1,48,100);
%record_T = zeros(5,length(record));
%for kk=1:5
 %   record_T(kk,:) = transpose(record(kk,:));
  %  record_T(kk,:) = FilterDFT(record_T(kk,:),h);
   % record(kk,:)= transpose(record_T(kk,:));
%end

%record_T = FilterDFT(record_T,h);
%record=transpose(record_T);
%d2 = designfilt('bandpassfir','FilterOrder',500, ...
 %  'CutoffFrequency1',0.001,'CutoffFrequency2',48, ...
  %'SampleRate',100);
%[b2,a2]=tf(d2);
% deleting useless data.
record=zeros(5,2850000);
[b2,a2]=butter(6,0.96);
for kk=1:5
record1(kk,:) = filter(b2,a2,record1(kk,:));
record(kk,1:2850000)=record1(kk,1:2850000);
end

s=size(record);
l=(s(2)/100)/10;
X=zeros(l,10);
Fs=100;

%{
# row
    1   Cz
    2   Oz
`   3   EOG
    4   EMG
%}

window=1000;

% Cz
% delta=1-4, theta=4-8, alpha=8-13, beta=13-30
for j=1:l

    w_length=((j-1)*window+1):j*window;
    sig_window=record(1,w_length);
    [Pxx,F] = periodogram(sig_window,rectwin(length(sig_window)),length(sig_window),Fs);
    p_delta = bandpower(Pxx,F,[1 4],'psd');
    p_theta = bandpower(Pxx,F,[4 8],'psd');
    p_alpha = bandpower(Pxx,F,[8 13],'psd');
    p_beta  =  bandpower(Pxx,F,[13 30],'psd');
    
    X(j,1:4)=[p_delta p_theta p_alpha p_beta];
    

end

% Oz
% delta=1-4, theta=4-8, alpha=8-13, beta=13-30
for j=1:l

    w_length=((j-1)*window+1):j*window;
    sig_window=record(2,w_length);
    [Pxx,F] = periodogram(sig_window,rectwin(length(sig_window)),length(sig_window),Fs);
    p_delta = bandpower(Pxx,F,[1 4],'psd');
    p_theta = bandpower(Pxx,F,[4 8],'psd');
    p_alpha = bandpower(Pxx,F,[8 13],'psd');
    p_beta  =  bandpower(Pxx,F,[13 30],'psd');
    
    X(j,5:8)=[p_delta p_theta p_alpha p_beta];
    

end

% EOG  

for j=1:l
    
    w_length=((j-1)*window+1):j*window;
    sig_window=record(3,w_length);
    [Pxx,F] = periodogram(sig_window,rectwin(length(sig_window)),length(sig_window),Fs);
    p_EOG = bandpower(Pxx,F,'psd');
    X(j,9)=p_EOG;
    
end
 
% EMG
for j=1:l
    
    w_length=((j-1)*window+1):j*window;
    sig_window=record(4,w_length);
    [Pxx,F] = periodogram(sig_window,rectwin(length(sig_window)),length(sig_window),Fs);
    p_EMG = bandpower(Pxx,F,'psd');
    X(j,10)=p_EMG;
    
end


% t vector

s1=max(Data(1,:));
t=0:10:s1;
s2=size(Data(1,:));
state=zeros(1,length(t));
for j=1:s2(2)-1
    
    a1=(Data(1,j)/10)+1;
    a2=(Data(1,j+1)/10);
    state(1,a1:a2)=Data(2,j);
    
end

% determine the dimention
if length(t)<l
    
    X=X(1:length(t),:);
else
    t=t(1:l);
    state=state(1:l);
end



end

