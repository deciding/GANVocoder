import sys
sys.path.insert(0, '../../../')
import os
from glob import glob
from tqdm import tqdm
from parallel_wavegan.utils import write_hdf5
import numpy as np

#TODO
dataname='kuangfei8k_gta'
data_dir='/workspace/ssd2/training_cn_kuangfei_prosody/8/'
root_dir='/workspace/ParallelWaveGAN/egs/csmsc/voc1/dump_gta_kf8k/'
meldirname='mels-gta'

datasets=['train_nodev', 'dev', 'eval']
mels_dir=f"{data_dir}/{meldirname}"
wave_dir=f"{data_dir}/audio"
dev_num=50
eval_num=50
num_split=16

mels=glob(f"{mels_dir}/*.npy")
mels_dataset=[mels[:len(mels)-dev_num-eval_num], mels[-dev_num-eval_num:-eval_num], mels[-eval_num:]]
for ind, dataset in enumerate(datasets):
    for cnt, melnpy in tqdm(enumerate(mels_dataset[ind])):
        job=(cnt%num_split)+1
        dump_dir=f'{root_dir}/{dataset}/norm/dump.{job}'
        os.makedirs(dump_dir, exist_ok=True)
        utt_id='_'.join(melnpy.split('.')[0].split('-')[1:])
        utt_id=dataname+utt_id
        wavenpy=melnpy.replace(meldirname, 'audio').replace('mel', 'audio')
        mel=np.load(melnpy)
        audio=np.load(wavenpy)
        assert audio.shape[0]==mel.shape[0]*300
        write_hdf5(os.path.join(dump_dir, f"{utt_id}.h5"), "wave", audio.astype(np.float32))
        write_hdf5(os.path.join(dump_dir, f"{utt_id}.h5"), "feats", mel.astype(np.float32))
