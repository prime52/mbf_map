%
% batch_ctp_v9.m
%
% see also: axial2shortAxis_v14.py
%
% 최대한 user개입 없이 한 번에 MBF short axis 영상 얻는 작업 수행.
%
% 2020-09-07
% 2020-09-14, 2020-09-15, 2020-09-16
% 2020-11-05
% 2020-11-24:  AIF ROI Unet 으로 자동계산, AIF 플롯 => U-net segment할 phase 자동계산
% 2021-12-09: 보다 향상된 딥러닝 모델로 segment, localize
%
%
% (방식) 
% 1. 512x512를 256x256으로 downsample하기.
% 2. registration하기.
% 3. AIF 계산하기.
% 4. phase별로 myocardium 영상 보기.
% 5. perfusion defect이 있는지 확인하기.
%
%

clear; close all;

python_exe = 'C:\Users\yoonc\anaconda3_20\envs\tensorflow1\python ';

test_or_train = true;  % if true, select test data. if false, select train data. 

dir_ctp_list = load_ctp_directory_list ( test_or_train ); % script 실행. 옵션 잘 조절하기 (테스트인지, 트레인 셋인지)
nsubj = numel(dir_ctp_list);
fprintf('\n nsubj = %d', nsubj);

%%
%% 사용자가 설정하기!!
%%
dispFig = false;
manual_select_phaseno = false;
manual_aorta_seg = false;
subjnos = (95);

for subjno = subjnos    
    
    directory = dir_ctp_list{subjno};
    [c, d] = fileparts(fileparts(fileparts(fileparts(fileparts(directory))))); 
    subj_id = d;   
    a = dir( directory );
    fprintf('\n subject directory = %s\n', directory);
    
    tic;
    % 다이콤 파일 이름의 여러 케이스를 핸들링하기 위해.
    if length(a) < 4,         error('directory is invalid');
    elseif strncmp(a(4).name, 'IMG0', 4),         fileext = 'IMG0*';
    elseif strncmp(a(4).name, '00', 2),         fileext = '*.dcm';
    elseif strncmp(a(4).name, 'PNUYH', 5),         fileext = '*.dcm';
    elseif strncmp(a(4).name, '12', 2),         fileext = '12*';
    end

    dcmfnames = dir([directory '/' fileext]); N = max(size(dcmfnames)); info_ctp = cell(1, N);
    %fprintf('\nN = %d\n', N);
    %dcmfnames(19).name;
    info0 = dicominfo(strcat(directory, '\', dcmfnames(1).name));        
    ctp_img_basic = zeros(info0.Rows, info0.Columns, N);

    if length(a) > 10      
        % fprintf('multiple dicom files... \n')
        ind = 1;
        for j = 1:length(a)
            if (strncmp(a(j).name, 'IM_0', 4) == true) || (strncmp(a(j).name, '00', 2) == true) || (strncmp(a(j).name, 'IMG0', 4) == true) || (strncmp(a(j).name, 'PNUYH', 5)==true) || (strncmp(a(j).name, '12', 2) == true)
               ctp_img_basic(:, :, ind) = dicomread(strcat(directory, '\', a(j).name));
               info_ctp{ind} = dicominfo(strcat(directory,'\',a(j).name));
               ind = ind + 1;
            end
        end
    end
    [nrow, ncol, nfiles] = size(ctp_img_basic);
    phase_start = info_ctp{1}.AcquisitionNumber;
    phase_end = info_ctp{N}.AcquisitionNumber;
    nphase = phase_end - phase_start + 1;
    nslice = nfiles/nphase;
    timestamp = zeros(nslice, nphase);
    img = zeros(nrow, ncol, nslice, nphase);

    for j = 1:nphase
        for k = 1:nslice
            index = (j-1)*nslice + k;
            img(:, :, k, j) = ctp_img_basic(:, :, index);
            timestamp(k, j) = str2double(info_ctp{index}.ContentTime);
        end
    end
    fprintf('\n nfiles = %d, nphase = %d, nslice = %d\n', N, nphase, nslice);
    
    dcmh = info_ctp{1};
    study_date = dcmh.StudyDate;
    series_description = dcmh.SeriesDescription;
    manufacturer = dcmh.Manufacturer;
    protocol_name = dcmh.ProtocolName;
    sex = dcmh.PatientSex;
    age = dcmh.PatientAge;
    timestamp = get_time_interval (info_ctp, nslice, nphase);  
    
    %% image resized to 256 x 256
    img2 = imresize(img, [256 256]);
    
    toc;
    
    %% image registration (256x256xnslice) inter-frame간 registration
    img2 = register_3d_v2( info_ctp, img2, python_exe );
    
    imgst = single(img2); 
    nslice_ = size(imgst, 3); 
    nphase = size(imgst, 4); 
    midslice = round(nslice_/2);
    
    % Cranial slice selection
    imgt_cranial = imgst(:, :, midslice-5, :); 
    
    % Caudal slice selection
    imgt_caudal = imgst(:, :, midslice+5, :);  % 5 is just picked in a range (30, 58)
    
    if manual_select_phaseno == true
        %for slno = 10:5:40

        slice_mid = round(nslice/2);

        for slno = [slice_mid-10, slice_mid-5, slice_mid, slice_mid+5, slice_mid+10]
            img2_t = squeeze(imgst(:, :, slno, :));
            figure;
            for tt = 1:nphase
                subplot(2, round(nphase/2), tt);
                imshow(img2_t(:, :, tt), [0 3000]); % Hounsfield unit
                hold on;
                line([0 ncol], [30, 30], 'color', 'red', 'linestyle', '--');
                title(sprintf('slice=%d, t=%d', slno,tt));
            end
        end
        phaseno = str2double(input('select frameno with the best myo/LV contrast: ', 's'));
    end
    
    if manual_aorta_seg    
        figure('name', 'roipoly() -- Aorta');
        if manual_select_phaseno==false
            phaseno = round(nphase/2); % arbitrary
        end
        imshow(imgt_cranial(:, :, phaseno), []); axis equal; axis tight;
        title(' roipoly() ');
        phno = phaseno; tmp = imgt_cranial(:,:,phno); tmp = tmp/max(tmp(:));
        bw_mask = roipoly(tmp);
        ind = find(bw_mask == 1);
        AIF_cranial = zeros(1, nphase);
        for t = 1:nphase
            img = imgt_cranial(:, :, t);
            roi = img(ind);
            AIF_cranial(t) = mean(roi(:)); 
        end    
        
    else
        
        phaseno_arb = round(nphase/2);       
        %% auto-segmentation with U-net
        img_cra = imgt_cranial(:,:,phaseno_arb+1);        img_cau = imgt_caudal(:,:,phaseno_arb+1);
        [bw_mask_cra, bw_mask_cau] = segment_aorta_auto2(img_cra, img_cau, python_exe);
        % erode mask
        se = strel('disk', 6, 4); bw_mask_cra = imerode(bw_mask_cra, se); bw_mask_cau = imerode(bw_mask_cau, se);
        if dispFig
            figure;
            subplot(121); imshow(img_cra, []); hold on; contour(bw_mask_cra, [0.5 0.5], 'r');
            subplot(122); imshow(img_cau, []); hold on; contour(bw_mask_cau, [0.5 0.5], 'r');
            title(sprintf('%s', directory));
        end
        ind1 = find(bw_mask_cra == 1); ind2 = find(bw_mask_cau == 1);
        AIF_cranial = zeros(1, nphase); AIF_caudal = zeros(1, nphase);
        for t = 1:nphase
            img1 = imgt_cranial(:, :, t); img2 = imgt_caudal(:, :, t);
            roi1 = img1(ind1); roi2 = img2(ind2);
            AIF_cranial(t) = mean(roi1(:)); AIF_caudal(t) = mean(roi2(:)); 
        end
    end
    
    if manual_aorta_seg
        figure;
        imshow(imgt_caudal(:, :, phno), []); axis equal; axis tight;
        title('roipoly()');
        tmp = imgt_caudal(:,:, phno); tmp = tmp/max(tmp(:));        bw_mask = roipoly(tmp);
        ind = find(bw_mask == 1);        AIF_caudal = zeros(1, nphase);
        for t = 1:nphase
            img = imgt_caudal(:, :, t);            roi = img(ind);            AIF_caudal(t) = mean(roi(:)); 
        end
    end
      
    aif1 = AIF_cranial; aif2 = AIF_caudal;
    aif = zeros(2*nphase, 1);    aif(1:2:end) = aif1;    aif(2:2:end) = aif2;
    aif_final = aif - min(aif);
    
    if 0
        % gaussian filtering of aif_final
        alpha = 3;
        w = gausswin(5, alpha);
        aif_final_filtered = conv(w, aif_final, 'same');
        figure;
        subplot(121);        stem(aif_final);
        subplot(122);        stem(aif_final_filtered);
    end
    
    
    %% 심근 분할에 적절한 contrast인 phase 선택하기.
    
    if manual_select_phaseno == true
        %for slno = 10:5:40
        for slno = round(nslice/2)
            img2_t = squeeze(imgst(:, :, slno, :));
            figure;
            for tt = 1:nphase
                subplot(2, round(nphase/2), tt);
                imshow(img2_t(:, :, tt), []);                hold on;
                line([0 ncol], [30, 30], 'color', 'red', 'linestyle', '--');
                title(sprintf('slice=%d, t=%d', slno,tt));
            end
        end

        %% user가 그림보고 phase 선택하기
        phaseno = str2double(input('select frameno with the best myo/LV contrast: ', 's'));
    else
        [~, phaseno1] = max(AIF_cranial);
        [~, phaseno2] = max(AIF_caudal);
        
        fprintf('\n Auto AIF max in cranial/caudal (phaseno1=%d, phaseno2=%d)\n', phaseno1, phaseno2);
        % 대체로 AIF가 maximum이 되는 phase에서 한 phase 이전에서 myocardium/blood
        % contrast가 높다. 
        phaseno = round( ( (phaseno1-1)+(phaseno2-1) )/2 );
    end
        
    fprintf('\n Selected phase index = %d\n', phaseno);
    
    %% save .mat
    timestamp_celltype = timestamp;
    fprintf('\n nphase = %d', nphase);
    time_sec = zeros(1, 2*nphase);

    for jj = 1:2*nphase
        % fprintf('\n timestamp = %s', timestamp_celltype{1, jj});
        str1 = timestamp_celltype{1,jj};
        vec1 = sscanf(str1, '%2d%2d%f');
        sec1 = sum([60*60; 60; 1].* vec1);
        time_sec(1, jj) = sec1;
    end
    dtime = [0 diff(time_sec)];
    t = cumsum(dtime);
    
    %% save 4-D image as .mat
    img4d = single(imgst);
    save('data/segment_SA/temp/ct_img4d.mat', 'img4d', 'phaseno', 'aif', 't', 'time_sec', 'subj_id');
    
    %% save 3-D image as .mat (for aorta segmentation)
    img3d = imgst(:, :, :, phaseno);  % phaseno+1이 aorta가 밝은 편임.
    % subject_id_number = sprintf('C%04d', subjno);
    %save(['data/segment_aorta/mat/' subj_id '_Axial_Ao_phase' num2str(phaseno) '_256.mat'], 'img3d', 'phaseno', 'subj_id');
    
    if 0
        figure;
        count = 1;
        for jj = [round(nslice/2)-5, round(nslice/2), round(nslice/2)+5]
            mbf = mbf_map_3d(:, :, jj);
            % show in axial slices    
            subplot(1,3,count);
            imshow(mbf, [0 180]);
            colormap(gca, jet);
            count = count+1;
        end
        colorbar;
    end

    %% short axis로 변환 MBF 오버레이하기.  
    %% 1. landmark detection (1. manual, 2. auto)
    %% 2. SA 영상에서 myocardium segmentation (automatic)
    commandstr = [python_exe ' axial2shortAxis_v14.py'];
    
    system(commandstr);
end


function [timestamp] = get_time_interval(dcminfo_ctp, nslice, nphase)

    % shuttle mode: table moving back and forth between the two
    % consecutive scanning positions

    N = nslice * nphase;
    timestamp = cell(1, 2*nphase);

    for jj = 1:N
        info = dcminfo_ctp{jj};
        if jj == 1
            info.ContentTime;
        end
        phaseno = floor( (jj-1)/nslice );    
        sliceno = mod(jj-1, nslice);
        RRno = 2 * phaseno;
        if sliceno == 0
            % CRANIAL slice
            timestamp{1, RRno+1} = info.ContentTime; 
        elseif sliceno == nslice-1
            % CAUDAL slice
            timestamp{1, RRno+2} = info.ContentTime;
        end
    end    
end


function [img] = register_3d_v2(dicominfo_ctp, imgst, python_exe)

    start_reg = tic;

    disp('3D registration of CT perfusion data - frame by frame');

    dcminfo1 = dicominfo_ctp{1};
    dcminfo2 = dicominfo_ctp{2};
    dx = dcminfo1.PixelSpacing(1);
    dy = dcminfo1.PixelSpacing(2);
    dz = dcminfo2.SliceLocation - dcminfo1.SliceLocation;
    [nrow, ncol, nslice, nphase] = size(imgst);

    %% save 3D images as .nii format
    info_template = niftiinfo('DynSerio4D_0_6_B23f_350ms_frame01.nii');   
    info1 = info_template;  % copy template info from a .nii file.   
    %info1.Description = 'ct_perfusion_data';
    info1.Datatype = 'uint16';
    info1.PixelDimensions = [dx*2, dy*2, dz];
    fprintf('\t dx=%f, dy=%f, dz=%f\n', dx*2, dy*2, dz);
    
    patientID = dcminfo1.PatientID;
    temp_directory = 'data\register'; %strcat(dat.script_directory, '\data\register');  
    mkdir(temp_directory);

    %% frame별로 .nii파일로 저장하기.
    for frameno = 1:nphase
        fname1 = strcat(temp_directory, '\', 'ctp_', patientID, sprintf('_frame%02d', frameno), '.nii');
        ctp_16bit = uint16(squeeze(imgst(:, :, :, frameno)));
        info2 = info1;
        info2.Filename = fname1;
        info2.ImageSize = [nrow, ncol, nslice];
        niftiwrite(ctp_16bit, fname1, info2);
    end

    %% run sitk and perform coregistration.
    commandStr = [python_exe ' '  'register_sitk_ctp_256x256_v4.py'];
    system(commandStr);

    disp('register done...');

    %% 파이선에서 레지스트레이션 돌리고 난 결과를 매트랩 mat파일로 저장
    %% mat파일을 읽어들임

    data_fname = 'data/register/ctp_reg_sitk_results.mat';

    var = whos('-file', data_fname);
    str1 = var(1).name; % final_reg
    x = load(data_fname, str1); 
    disp('loading done..');
    ctp_img_reg = x.final_reg.ctp_4d_reg;

    % 최종 CTP co-register된 영상 결과물
    img = single(ctp_img_reg); 
    time_for_3dreg = toc(start_reg);
    fprintf(' 3D registration took %6.2f seconds\n', time_for_3dreg);

    %% nii 파일 지우기
    delete([temp_directory '\*.nii']);
end


function [mask_aorta_cranial, mask_aorta_caudal] = segment_aorta_auto2(img_cranial, img_caudal, python_exe)

    save( 'data/segment_aorta/temp/ct_img2d.mat',  'img_cranial', 'img_caudal');

    commandstr = [python_exe ' unet_ctp_seg_Aorta_test_v2.py --option dwi_adc'];
    system(commandstr);

    data_fname = 'data/segment_aorta/temp/mask_aorta_2d.mat';   

    a = load(data_fname);
    mask_aorta_cranial = a.mask_aorta_cranial;
    mask_aorta_caudal = a.mask_aorta_caudal;

    save('data/segment_aorta/temp/ct_img2d_aorta_mask.mat', 'img_cranial', 'img_caudal', 'mask_aorta_cranial', 'mask_aorta_caudal');

end % segment_aorta_auto2


