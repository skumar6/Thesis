%%plot for time and accx spikes, time comaparision 
name = 'tsy4';
ps1= 'C:/Users/Sanjeev/Desktop/plotHelpers/';
ps2= '.txt';
fileName = [ps1 name ps2];

y = dlmread(fileName, ' ' );
timeStmapFile = 'C:/Users/Sanjeev/Desktop/plotHelpers/classifyYesTime.txt';
timeStamp = dlmread(timeStmapFile, ' ');
timeStamps = transpose(timeStamp);
slen=size(y,1);
filelen = (1:slen); 
dummy = zeros(slen, 1);
audioDummy = zeros();



for i= 1:slen
   if(any(timeStamps == y(i,2)))
       dummy(i)=1.5;  
       %disp(i);
   end
end
figure;



plot(y(:,2), y(:,4), '-');
title('plot and timestmap');
xlabel('time');
ylabel('amplitude');
hold on;

plot(y(:,2), dummy(:,1), 'r-');

hold on;

%writing audio times manuaally
 set = [1.473224465239E12 1.473224475373E12 1.473224489983E12];

 dummyA = zeros(1,(y(slen,2)-y(1,2))+1);
 count = 0;
 for i = y(1,2):y(slen,2)
     count = count+1;
     if(any(set == i))
         dummyA(count)= 1.5;
    end
 end


fils=(1:y(slen,2)-y(1,2)-1);
c=0;
for i = y(1,2):y(slen,2)
    c=c+1;
    fils(1,c) = i;
end
plot(fils, dummyA, '-g');
legend('Acc-x-Motion', 'timeStamp-motion','timeStamp-audio', 'Location','southwest')
hold off;



