name = 'tsy26';  %original inertail sensor data file from app
%it ll plot diffenr sensor values
%Before running this script it is must to run motionDataProcessing.java
% on the same data file we are working over here.

ps1= 'C:/Users/Sanjeev/Desktop/plotHelpers/';
ps2= '4plots.txt';
fileName = [ps1 name ps2];

fig1 = figure;
sps1 =  'C:/Users/Sanjeev/Desktop/plotHelpers/';
sps2=  '4plots.txt';
smoothenFile = [sps1 name sps2];

y = dlmread(fileName, ' ' ); % Read the original file
sy = dlmread(smoothenFile,''); % read smoothen file written by motionDataProcessing.java

peaksFile = 'C:/Users/Sanjeev/Desktop/plotHelpers/peaks.txt';  % read peaks written by motionDataProcessing.java
peaks = dlmread(peaksFile, ' ');

% timeStmapFile = 'C:/Users/Sanjeev/Desktop/plotHelpers/classifyYesTime.txt';
% timeStamps = dlmread(timeStmapFile, ' ');

len=size(y,1);
slen = size(sy,1);
filelen = (1:slen);


subplot(2,1,1);
plot(y(:,3));
hold on;
title('rms');
for i=1:slen
    if any(peaks==i)
        plot(filelen(i), y(i,3), '*r');
    end   
end
hold off;

yy = smooth(y(:,3), 11);
subplot(2,1,2);
plot(yy);
title('smoothen-rms');
hold on;
for i=1:slen
    if any(peaks==i)
        plot(filelen(i), yy(i), '*r');
    end   
end
hold off;

saveas(fig1,'C:/Users/Sanjeev/Desktop/thesisWriting/movingwidowrms.png')

fig2=figure;

subplot(2,1,1);
plot(1:slen, sy(:,3),'-');
hold on 
for i=1:slen
    if any(peaks==i)
        plot(filelen(i), sy(i,3), '*r');
    end   
end
title('acc-x')
xlabel('sample(fq MS128)');
ylabel('accx-value');
hold off;

subplot(2,1,2);
plot(1:slen, sy(:,4), '-');
hold on;
for i=1:slen
    if any(peaks==i)
        plot(filelen(i), sy(i,4), '*r');
        %hold on;
    end   
end
title('MovingWindow(windoe size 10)');
xlabel('sample(fq MS128)');
ylabel('accx-filtered');
hold off;
saveas(fig2,'C:/Users/Sanjeev/Desktop/thesisWriting/movingwidowaccx.png')

figure();
peaksVal=[];
subplot(3,1,1);
plot(1:len, y(:,4),'.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,4), '*r');
    end   
end
title('accelerometerX');
xlabel('sample no. (frequency MS128)');
ylabel('accx-value');
hold off;



subplot(3,1,2);
plot(1:len, y(:,5),'.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,5), '*r');
    end   
end
title('accelerometerY');
xlabel('sample no. (frequency MS128)');
ylabel('accy-value');
hold off;



subplot(3,1,3);
plot(1:len, y(:,6),'.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,6), '*r');
    end   
end
title('accelerometerZ');
xlabel('sample no. (frequency MS128)');
ylabel('accz-value');
hold off;

figure;
subplot(3,1,1);
plot(1:len, y(:,7),'.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,7), '*r');
    end   
end
title('gyroscopeX');
xlabel('sample no.(frequency MS128)');
ylabel('gyrox-value');
hold off;

subplot(3,1,2);
plot(1:len, y(:,8),'.-');
hold on;
for i=1:slen
    if any(peaks==i)
        plot(filelen(i), y(i,8), '*r');
    end   
end
title('gyroscopeY');
xlabel('sample no. (frequency MS128)');
ylabel('gyroy-value');
hold off;
 
subplot(3,1,3);
plot(1:len, y(:,9),'.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,9), '*r');
    end   
end
title('gyroscopeZ');
xlabel('sample no. (frequency MS128)');
ylabel('gyroz-value');
hold off;

figure;
subplot(3,1,1);
plot(1:len, y(:,3), '.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,3), '*r');
    end
end
title 'accelerometer-rms'
xlabel 'sample no. (frequency MS128)';
ylabel 'accrms-value';
hold off;

subplot(2,1,2);
plot(1:len, y(:,10), '.-');
hold on;
for i=1:len
    if any(peaks==i)
        plot(filelen(i), y(i,10), '*r');
    end
end
title 'gyro-rms'
xlabel 'sample no. (frequency MS128)';
ylabel 'gyro-value';
hold off;




