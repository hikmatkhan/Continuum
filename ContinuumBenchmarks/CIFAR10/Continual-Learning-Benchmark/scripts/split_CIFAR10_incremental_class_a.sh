GPUID=3
OUTDIR=outputs_splits_2/split_CIFAR10_incremental_class
REPEAT=5
mkdir -p $OUTDIR

BATCH_SIZE=256 #128

#IBATCHLEARNPATH=/home/hikmat/Desktop/JWorkspace/CL/Continuum/ContinuumBenchmarks/CIFAR10/Continual-Learning-Benchmark
IBATCHLEARNPATH=/home/khanhi83/JWorkspace/Continuum/ContinuumBenchmarks/ContinuumBenchmarks/CIFAR10/Continual-Learning-Benchmark


#IBATCH_LEARN_LAMBDA=/home/khanhi83/GitHub/ContinuumBenchmarks/ContinuumBenchmarks/CIFAR10/Continual-Learning-Benchmark
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet                        --lr 0.1 --momentum 0.9 --weight_decay 1e-4 --offline_training  | tee ${OUTDIR}/Offline_SGD_WideResNet_28_2_cifar.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet                        --lr 0.001                                  --offline_training  | tee ${OUTDIR}/Offline_Adam_WideResNet_28_2_cifar.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet                                             --lr 0.001                                 | tee ${OUTDIR}/Adam.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet                                             --lr 0.1                                   | tee ${OUTDIR}/SGD.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adagrad --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet                                             --lr 0.1                                   | tee ${OUTDIR}/Adagrad.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/EWC.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/EWC_online.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 0.001        | tee ${OUTDIR}/SI.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 500          | tee ${OUTDIR}/L2.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_1400  --lr 0.001              | tee ${OUTDIR}/Naive_Rehearsal_1400.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_5600  --lr 0.001              | tee ${OUTDIR}/Naive_Rehearsal_4600.log
#python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 1 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 0.001        |tee  ${OUTDIR}/MAS.log

python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_1100  --lr 0.001              | tee ${OUTDIR}/Naive_Rehearsal_1100.log
python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_1400  --lr 0.001              | tee ${OUTDIR}/Naive_Rehearsal_1400.log
python -u $IBATCHLEARNPATH/iBatchLearn.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 80 120 160 --batch_size $BATCH_SIZE --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_4000  --lr 0.001              | tee ${OUTDIR}/Naive_Rehearsal_4000.log

