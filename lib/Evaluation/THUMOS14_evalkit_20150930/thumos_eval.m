files=dir('action_detection_*');
maps=zeros(1,length(files));

for i=1:length(files)
    [pr_all,ap_all,map]=TH14evalDet(files(i).name,'annotation','test',0.5);
    fprintf('file: %s map %1.3f\n',files(i).name,map);
    maps(i)=map;
end
