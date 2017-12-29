%将视频转换为单张图片
clear
clc

file_name = '30.mp4';        %视频所在文件夹
obj = VideoReader(file_name);     %读取视频文件

numFrames = obj.NumberOfFrames;   %视频总的帧数 
i=1;
for k = 1: numFrames
    frame = read(obj,k);
    %imshow(frame);                
%     gray_frame = rgb2gray(frame); %若每一帧为彩色图片，转换为灰度图
    imshow(frame);                %显示每一帧图片
    %保存每一帧图片
    d=i+3*(i-1);
    if k==d
    %if k==1:2:2951
        imwrite(frame,strcat('F:\猪脸识别\智能控制理论与方法-研究生\database\train_img\30\',num2str(k),'.jpg'),'jpg');
        i=i+1;
    end
%     imwrite(gray_frame,strcat('C:\Users\cuining\Desktop\猪脸识别\train_img\2\',num2str(k),'.jpg'),'jpg');
    
end

