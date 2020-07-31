
%%
%样本降维

%%
%加载数据

%加载模板

mask = load_nii('E:\蒋皆恢\帕金森诊断\数据\aal_95_79_95_69_RL.img');
m = mask.img;
m(m>90) = 0;
m(m>0) = 1;
mm = reshape(m,1,517845);

%PD数据

path = uigetdir;
path_img=strcat(path,'\');   
DIR_img=dir(strcat(path_img,'*.img')); %获取该子文件夹中所有nii格式的图像
img_name=cell(length(DIR_img),1);
for k = 1:length(DIR_img)
img_name{k,1}=DIR_img(k).name;
end
img_name=sort_nat(img_name);
number = length(img_name); 

PDdata = [];

for i = 1:number
    name = img_name{i};
    full_name = strcat(path_img,'\',name);
    im = load_nii(full_name);
    im_data = double(im.img);
    
    i
    %套模板
    ddata = reshape(im_data,1,517845) .* mm;
    ddata = ddata(ddata>0);
    
    PDdata(i,:) = ddata;
    
end

%正常组数据

path = uigetdir;
path_img=strcat(path,'\');   
DIR_img=dir(strcat(path_img,'*.img')); %获取该子文件夹中所有nii格式的图像
img_name=cell(length(DIR_img),1);
for k = 1:length(DIR_img)
img_name{k,1}=DIR_img(k).name;
end
img_name=sort_nat(img_name);
number = length(img_name); 

NCdata = [];

for i = 1:number
    name = img_name{i};
    full_name = strcat(path_img,'\',name);
    im = load_nii(full_name);
    im_data = double(im.img);
    
    %套模板
    i
    ddata = reshape(im_data,1,517845) .* mm;
    ddata = ddata(ddata>0);
    NCdata(i,:) = ddata;
    
end

totalData = [PDdata;NCdata];

%%
%LLE降维

DRdata = lle(totalData',10,350);








