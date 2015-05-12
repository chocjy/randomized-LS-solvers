FILENAME=nonunif_bad_250000_1000
#FILENAME=nonunif_good_250000_1000
M=250000
N=1000
Q=10

for r in 5000 50000
do
    for proj in cw gaussian
    do
        ./run_ls.sh $FILENAME --dims $M $N --high --proj -p $proj -r $r --nrep 40 --stack 2 -q $Q -k 1 -t --load_N --save_N --save_logs --hdfs --setting conf/settings_jiyan.cfg --logging conf/logging_exp_high.cfg
    done

    # using sampling
    ./run_ls.sh $FILENAME --dims $M $N --high --samp -p gaussian -r 50000 -s $r --nrep 40 --stack 2 -q $Q -k 1 -t --load_N --save_N --save_logs --hdfs --setting conf/settings_jiyan.cfg --logging conf/logging_exp_high.cfg
done

# using no preconditioner
./run_ls.sh $FILENAME --dims $M $N --high --nrep 40 --stack 2 -q $Q -k 1 -t --load_N --save_N --save_logs --hdfs --setting conf/settings_jiyan.cfg --logging conf/logging_exp_high.cfg
