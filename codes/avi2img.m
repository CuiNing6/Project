%����Ƶת��Ϊ����ͼƬ
clear
clc

file_name = '30.mp4';        %��Ƶ�����ļ���
obj = VideoReader(file_name);     %��ȡ��Ƶ�ļ�

numFrames = obj.NumberOfFrames;   %��Ƶ�ܵ�֡�� 
i=1;
for k = 1: numFrames
    frame = read(obj,k);
    %imshow(frame);                
%     gray_frame = rgb2gray(frame); %��ÿһ֡Ϊ��ɫͼƬ��ת��Ϊ�Ҷ�ͼ
    imshow(frame);                %��ʾÿһ֡ͼƬ
    %����ÿһ֡ͼƬ
    d=i+3*(i-1);
    if k==d
    %if k==1:2:2951
        imwrite(frame,strcat('F:\����ʶ��\���ܿ��������뷽��-�о���\database\train_img\30\',num2str(k),'.jpg'),'jpg');
        i=i+1;
    end
%     imwrite(gray_frame,strcat('C:\Users\cuining\Desktop\����ʶ��\train_img\2\',num2str(k),'.jpg'),'jpg');
    
end

