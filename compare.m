%plot audioplots, but before running this script audioDataProcessing.java
%must run.

[Y, FS]= audioread('C:/Users/Sanjeev/Desktop/soundClips/1490915164844.wav');
pointersFile = 'C:/Users/Sanjeev/Desktop/audioJava/pointers.txt';
movingWinFile= 'C:/Users/Sanjeev/Desktop/audioJava/MovingAvg.txt';
sampleTextFile = 'C:/Users/Sanjeev/Desktop/audioJava/sampleText.txt';
hilbertFile = 'C:/Users/Sanjeev/Desktop/audioJava/hilbertOutput.txt';
timeSeriesChunksFile = 'C:/Users/Sanjeev/Desktop/audioJava/timeSeriesChunks.txt';
fftChunkFile = 'C:/Users/Sanjeev/Desktop/audioJava/fftChunks.txt';


fftChunkstcell = createCell(fftChunkFile);
timeSeriesChunksdcell= createCell(timeSeriesChunksFile);
hilbertreadcell = createCell(hilbertFile);
javareadcell = createCell(sampleTextFile);
pointerscell = createCell(pointersFile);
movingAvgcell = createCell(movingWinFile);

pointers= transpose(cell2mat(pointerscell));

subplot(3,1,1);
plot(cell2mat(hilbertreadcell));
xlabel('HibertTransform');
ylabel('amplitude');
subplot(3,1,2);
plot(cell2mat(movingAvgcell));
xlabel('Moving Average');
ylabel('amplitude');

subplot(3,1,3);
originalSignal = transpose(cell2mat(javareadcell));
audiolen = length(originalSignal);
filelen = (1:audiolen); 
plot(originalSignal);
hold on;
for i=1:audiolen
    if any(pointers==i)
        plot(filelen(i), originalSignal(i), '.r');
    end   
end
xlabel('originalsignal with start-end points');
ylabel('amplitude');
hold off;








