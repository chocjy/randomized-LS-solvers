FILENAME=nonunif_bad_250000_1000
#FILENAME=nonunif_good_250000_1000
M=250000
N=1000

for r in 5000 10000 50000 100000
do
    for proj in cw gaussian rademacher srdht
    do
        ./run_ls.sh $FILENAME --dims $M $N --low --proj -p $proj -r $r --nrep 40 --stack 2 -k 3 -t --load_N --save_N --hdfs --setting settings_jiyan.cfg --logging logging_exp_low.cfg
    done

    # using sampling
    ./run_ls.sh $FILENAME --dims $M $N --low --samp -p cw -r 300000 -s $r --nrep 40 --stack 2 -k 3 -t --load_N --save_N --hdfs --setting settings_jiyan.cfg --logging logging_exp_low.cfg
done

