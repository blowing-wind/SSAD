files=dir('action_detection_*');
file=files(1).name;
ious=[0.5,0.3,0.4,0.6,0.7];
maps=zeros(1,length(ious));

for i=1:length(ious)
    [pr_all,ap_all,map]=TH14evalDet(file,'annotation','test',ious(i));
    fprintf('file: %s iou %1.1f map %1.3f\n',file,ious(i),map);
    maps(i)=map;
end
