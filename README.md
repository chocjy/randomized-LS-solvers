# Randomized Solvers for Large-scale Least-Squares Problems with Spark

This is a collection of codes that implement algorithms for solving large-scale least-squares problems using randomized numerical linear algebra with Spark via Python API.

by *Jiyan Yang* (jiyanyang12@gmail.com)

## About
Given *A*(*n*-by-*d*) and *b*(*n*-by-*1*), the least-squares regression problem is to solve:
    min_*x* ||*Ax-b*||_2.
When solving least-squares problems, randomized numerical linear algebra algorithms first compute a sketch for the linear system, then use it in one of the following two ways to get either low-precision or high-precision solutions:
+ Low-precision solvers:
    solve the subproblem induced by computing a sketch
+ High-precision solvers:
compute a preconditioner using the sketch and invoke LSQR to solve the preconditioned problem

For sketch, there are two available choices:
+ projection:
perform a random projection on *A*
+ sampling:
using random projection to estimate the leverage scores first and use them to construct a sampling sketch

For projection method, there are four choices available:
+ cw:
sparse count-sketch like transform (http://arxiv.org/abs/1207.6365)
+ gaussian:
dense Gaussian transform
+ rademacher:
dense Rademacher transform
+ srdht:
subsampled randomized discrete Hartley transform

## Folders
+ `src/`: contains all the source codes
+ `data/`: default path to local files storing datasets
+ `test/`: contains basic test codes
+ `N_file/`: stores matrices obtained by the sketches that can be reused in the future
+ `result/`: stores the computed solutions and total running time
+ `log/`: stores the Spark log files (if the flag `--save_logs` in on)

## Input
  Current implementation assumes that the augmented matrix [*A* *b*] is stored in plain text format with file name `FILENAME.txt` (meaning the last column is the response vector b). It can be loaded from one of the following three sources:
+ local: from local disc (default)
+ HDFS: from Hadoop file system (using `--hdfs`)
+ S3: from Amazon S3 file system (using `--s3`)
For the purpose of evaluating the computed solutions, files named `FILENAME_x_opt.txt` and `FILENAME_f_opt.txt` which store the optimal solution vector and objective value should be provided in the local folder `data_dir` (see below); otherwise they will be computed in the program. To generate larger matrices, one can use the option `--nrepetitions NUM` to creat a larger matrix by stacking the original one vertically `NUM` times.

## Output
  A file which stores the computed solutions and total running time will be stored (default file name is ls.out) in the `result/` subdirectory. This file can be opened by the cPickle module.

## Configuration
  The Spark configurations can be set via the script `run_ls.sh` from which the Spark job is submitted. Two `.cfg` files storing general setting of the program and Python logging setting respectively are needed to be set.
  
  The default filename for general setting is `conf/setting.cfg` in which there are three sections, namley, `local_directories`, `hdfs` and `s3`. In `local_directories` section, two directories are needed to be set so that the files can be properly loaded and saved. 
+ `data_dir`: path to local data files
+ `spark_logs_dir`: path to the folder that stores the Spark log files (if the flag `--save_logs` in on)
In `hdfs` and `s3` sections, paths leading the dataset should be provided if either file system if used. For S3, `key_id` and `secret_key` are also required.

The default filename for Python logging module is `conf/logging.cfg`. Stage updates and computed accuracies will be logged. Its configuration (e.g., location of the log file) can be set via the configuration file.
  
Configuration files with names other than the default ones can be passed into the program using `--setting_filename settingConfFilename` and `--logging_filename settingConfFilename`.

## Usage
```sh
$ ./run_ls.sh [-h] --dims m n [--nrepetitions numRepetitions]
                 [--stack stackType] [--npartitions numPartitions]
                 [--setting_filename settingConfFilename]
                 [--logging_filename loggingConfFilename] [-c] [--hdfs | --s3]
                 [--low-precision | --high_precision]
                 [--projection | --sampling]
                 [-p {cw,gaussian,rademacher,srdht}] [-r projectionSize]
                 [-s samplingSize] [-q numIters] [-k numTrials] [-t]
                 [--save_logs] [--output_filename outputFilename] [--load_N]
                 [--save_N] [--debug]
                 dataset
```
Type `./run_ls.sh -h` for help message.

## Examples
Some toy datasets are placed in the folder `data/` along with a file `gen_nonunif_bad_mat.py` for generating datasets. Below are a few examples showing how the program can be executed.

```sh
./run_ls.sh nonunif_bad_1000_10 --dims 1000 10 --low --proj -p cw -r 100 -k 3 -c
./run_ls.sh nonunif_bad_1000_50 --dims 1000 50 --low --proj -p gaussian -r 200 -k 3 -t --save_N
./run_ls.sh nonunif_bad_1000_50 --dims 1000 50 --low --samp -p gaussian -s 400 -r 200 -k 3 -t --load_N --save_N
./run_ls.sh nonunif_bad_1000_50 --dims 1000 50 --high --proj -p gaussian -r 200 -q 5 -k 3 -t --load_N --save_logs
./run_ls.sh nonunif_bad_1000_50 --dims 1000 50 --high --samp -p rademacher -s 200 -r 300 -q 3 -k 3 -t --nrepetition 5 --save_logs
./run_ls.sh nonunif_bad_1000_10 --dims 1000 10 --high --proj -p srdht -r 200 --nrep 10 -q 3 -k 1 -t --load_N --save_logs --save_N --hdfs
```

## Reference
Jiyan Yang, Xiangrui Meng, and Michael W. Mahoney, [Implementing Randomized Matrix Algorithms in Parallel and Distributed Environments](http://arxiv.org/abs/1502.03032).

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
