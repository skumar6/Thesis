%playing with original audio .wav file
[Y, FS]= audioread('C:/Users/Sanjeev/Desktop/soundClips/1472152629747.wav');
len = length(Y);
time = 1/FS* (1:len);
nfreq = FS/2;

disp(len);
plot(Y);
figure;




envelope = abs(hilbert(Y));

% subplot(3,1,1);
plot(envelope);
title('hilbert envelope');
xlabel('time');
ylabel('amplitude');
disp(length(envelope));
figure;

windowWidth = 2500;
kernel = ones(windowWidth,1) / windowWidth;
out = filter(kernel, 1, envelope);

% subplot(3,1,2);
plot(out);
title 'moving average'
xlabel('time');
ylabel('amplitude');
hth = 0.25;
lth = 0.1;
figure;

peaks = [];
disp(length(out));
disp(out(100));
for i = 1:len
  if  hth==0.25 && out(i) > hth
      disp(i);
     peaks = [peaks, i-2500];
     hth = 2;
     lth = .15;
  end
  
  if lth==0.15 && out(i) <lth
       disp(i);
    peaks = [peaks, i];
     hth = .25;
     lth = -1;
  end
end
filelen = (1:len); 

% subplot(3,1,3);
plot(Y);
title('audio signal');
xlabel('time');
ylabel('amplitude');

hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), Y(i), '.g');
    end   
end
hold off;

featureMetric = [];
i = 1;
s=1;
 
while i<=length(peaks)
    features=[];
    secchunk = Y(peaks(i):peaks(i+1)); 
    i = i+2;
    seclen = length(secchunk);
    t = 1/FS*(1:seclen);
    sec_fft = abs (fft(secchunk));
    sec_fft = sec_fft(1 : (seclen/2) );
    str = ['sample no.  ', num2str(s)];
    s=s+1;
    figure;
    
    subplot(2,1,1);
    plot(secchunk);
    xlabel('time');
    ylabel('amplitude');
    title(str);
    
    subplot(2,1,2);
    plot(sec_fft);
    title('fft')
    xlabel('frequency');
    ylabel('magnitude');
    
   
    features = [features, min(secchunk), max(secchunk), mean(secchunk), var(secchunk),  min(sec_fft), max(sec_fft), mean(sec_fft), var(sec_fft),1];
  %  numstr = [num2str(onsets(i,1)),' -- ', num2str(onsets(i,2))];
   % disp(numstr);
    featureMetric = [featureMetric; features];
end








