

rep=$(pwd)

python main_holo_tsf.py --checkpoint_dir $rep/holography/checkpoints/$1 --sample_dir $rep/holography/eval_samples/ --params "chosenIteration=${2}, phase=train" --save_dir "./data1/"
