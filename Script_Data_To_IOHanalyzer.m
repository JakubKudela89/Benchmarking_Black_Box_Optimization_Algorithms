%% script for the transformation of the log files for the IOHanalyzer (https://iohprofiler.github.io/IOHanalyzer/)
% parameters to change: 
% results_file_name (line 11)
% alg_name (line 12)
% selected_problems (line 24)
% afterwards, zip the subfolders in the "IOHdata" folder and upload the data into the https://iohanalyzer.liacs.nl/
% the script also downloads the results prezented in the paper from a zenodo repository

clear;clc;close all;

if not(isfolder('Result_files'))
    fullURL = 'https://zenodo.org/record/8362955/files/Result_files.zip?download=1';
    filename = 'Result_files.zip';
    disp('Downloading the result files');
    websave(filename, fullURL);
    disp('Download finished');
    disp('Unzipping the files');
    unzip('Result_files.zip'); 
    delete('Result_files.zip');
    disp('Unzipping finished');
end

addpath('Result_files');

% results_file_name = "Result_LSHADE"; % filename to be processed
% alg_name = "LSHADE"; % algorithm name

results_file_name = "Result_AGSK"; alg_name = "AGSK";
%results_file_name = "Result_EA4eig"; alg_name = "EA4eig";

load(strcat(results_file_name,".mat")); % loading data

problems = unique(DIRECTGOLib_Results(2:end,2));

mkdir("IOHdata"); % creating main directory
dirname = strcat("IOHdata/IOHdata_",results_file_name); % creating subdirectory for the algorithm
mkdir(dirname);
selected_problems = 1:2; % selection of the problems for which the data should be transformed; "1:length(problems)" selects all

for i=selected_problems  

    % creating the metadata file
    problem_name = problems{i};
    mkdir(strcat(dirname,"/data_f",num2str(i)));
    file_info_name = strcat(dirname,"/",'IOHprofiler_f',num2str(i),'.info');
    fileID = fopen(file_info_name,'w');
    ids_problem = find(strcmp(DIRECTGOLib_Results(:,2),problem_name));
    problem_results = DIRECTGOLib_Results(ids_problem,:);
    dimensions = cell2mat(problem_results(:,3));
    unique_dimensions = unique(dimensions);    
    for ii=1:length(unique_dimensions)
        d = unique_dimensions(ii);
        fprintf(fileID,"suite = 'DIRECTGOLib', funcId = %i, DIM = %i, algId = '%s', algInfo = '_'\n%%\n",i,d,alg_name);
        subfile_str = strcat("data_f",num2str(i),"/IOHprofiler_f",num2str(i),"_DIM",num2str(d,'%i'),".dat");
        fprintf(fileID,'%s, ',subfile_str);
        instances = find(dimensions == d);
        for iii = 1:length(instances)
            instance = problem_results{instances(iii),4};
            evals = problem_results{instances(iii),8}(end,2);
            fbest = problem_results{instances(iii),8}(end,3);
            if iii == length(instances)
                fprintf(fileID, '1:%i|%12.12e\n',evals,fbest);
            else
                fprintf(fileID, '1:%i|%12.12e, ',evals,fbest);
            end
        end
    end
    fclose(fileID);

    % creating the raw files
    for ii=1:length(unique_dimensions)
        d = unique_dimensions(ii);
        subfile_str = strcat(dirname,"/data_f",num2str(i),"/IOHprofiler_f",num2str(i),"_DIM",num2str(d,'%i'),".dat");
        fileID = fopen(subfile_str,'w');       
        instances = find(dimensions == d);
        for iii = 1:length(instances)
            fprintf(fileID,'"function evaluation" "best-so-far f(x)"');
            instance = problem_results{instances(iii),4};
            evals = problem_results{instances(iii),8}(:,2);
            fbest = problem_results{instances(iii),8}(:,3);
            for iiii=1:length(evals)
                fprintf(fileID,'\n%i %12.12e',evals(iiii),fbest(iiii));
            end
                fprintf(fileID,'\n');
        end
        fclose(fileID);
    end

end

