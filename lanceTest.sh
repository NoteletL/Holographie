

rep=$(pwd)

python main_holo.py --test_noisy_img $rep/Holography/DATABASE/PATTERN1/MFH_0/NoisyPhase.mat --test_noisy_key 'NoisyPhase' --test_clean_img $rep/Holography/DATABASE/PATTERN1/PhaseDATA.mat --test_clean_key 'Phase' --test_flip False --params "phase=test" --test_ckpt_index $rep/holography/checkpoints:run-test2021-01-13_09\:57:\27.958861/
