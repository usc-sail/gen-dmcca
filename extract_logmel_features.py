import sys
sys.path.insert(0, 'audioset_scripts')
from vggish_input import *
import glob
import tqdm

#1. Download the speech commands dataset in the data dir: 
    ## https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

wav_files = glob.glob("./data/*/*wav")
if wav_files:
    for wav_ in tqdm.tqdm(wav_files):
        out_name = wav_.replace(".wav", ".npy")
        logmel_arr = wavfile_to_examples(wav_)
        np.save(out_name, logmel_arr)
else: 
    sys.exit('Download the Speech Commands Dataset, in the right directory, and then run this script')

        


