
import argparse
import time
from subprocess import call
import GPUtil

# models_in = ["model_2022-12-04 14:36:56_convnext_iso_iso_0_not_orig_0_pre_1_aug_0_adv__50_at_crop_flip"]
# models = ['convnext_tiny', 'convneinfinfinfxt_tiny', 'convnext_tiny']
# nichts = []
paramss = [
          # ("model_2023-01-13 14:39:35_convnext_base_upd_0_not_orig_1_pre_0_aug_1_adv_50AT_allaug_N1.5N2N","convnext_base", 1, 1, 1, "Linf", 256, 46),
          # ("model_2022-12-31 15:33:27_convnext_base_upd_0_not_orig_0_pre_1_aug_1_adv_50_AT_allaugfrpre1k","convnext_base", 0, 1, 1, "Linf", 48),
          # ("model_2023-01-13 14:39:35_convnext_base_upd_0_not_orig_1_pre_0_aug_1_adv_50AT_allaug_N1.5N2N/big_convnext_weights","convnext_base", 1, 1, 1, "L", 75, 180),
          # ("model_2023-01-17 01:31:46_deit_s_upd_0_not_orig_1_pre_0_aug_1_adv_8_255_finetuning","deit_s", 1, 1, 1, "L2", 80, 164),
          ("model_2023-01-15 21:30:40_convnext_base_upd_0_not_orig_1_pre_0_aug_1_adv_250at_allaug_from_cleanpretrained_N_1.5N_2N","convnext_base", 1, 1, 0, "Linf", 100, 18),
          # ("model_2023-01-15 21:30:40_convnext_base_upd_0_not_orig_1_pre_0_aug_1_adv_250at_allaug_from_cleanpretrained_N_1.5N_2N","convnext_base", 1, 1, 1, "L1", 85, 150),

          ]
str_to_run = 'AA_eval.py'   
data_path = '/scratch/datasets/CIFAR10'
save_path = '/mnt/SHARED/nsingh/UFAT_results'
models_to_run = []

for minn, modd, not_o, a100, fullaa, norr, bs, ix in paramss:
  models_to_run.append('{} --model_in {} --mod {} --not-orig {} --a100 {} --full_aa {} --l_norms {} --batch_size {} --indx {}'
              .format(str_to_run, minn, modd, not_o, a100, fullaa, norr, bs, ix))

cart_prod = [a \
for a in models_to_run]
  
for job in cart_prod:
    print(job)

time.sleep(5)
count = 0
wait = 0
while wait<=1: 
  gpu_ids  = GPUtil.getAvailable(order = 'last', limit = 8, \
    maxLoad = .1, maxMemory = .5) # get free gpus listd
  if len(gpu_ids) > 0:
    print(gpu_ids)
    # time.sleep(5)
    for id in gpu_ids:
      if id == 5:
        pass
      else:
        if id != 10:
          temp_list = cart_prod[count]
    
          command_to_exec = '' +\
          ' CUDA_VISIBLE_DEVICES='+str(id)+\
          ' python3' +\
          ' ' + temp_list\
          + ' &' # for going to next iteration without job in background.
    
          print("Command executing is " + command_to_exec)
          call(command_to_exec, shell=True)
          print('done executing in '+str(id))
          count += 1
          time.sleep(2) # wait for processes to start
      # if count == 3:
      #   time.sleep(7200)
  else:
    print('No gpus free waiting for 30 seconds')
    time.sleep(15)
    wait+=1
# time.sleep(3600*3) # wait for processes to start
