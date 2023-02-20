
import argparse
import time
from subprocess import call
import GPUtil


paramss = [ 
          ("location_of_robust_model","convnext_base", 1, 1, 0, "Linf", 100),
          ]
str_to_run = 'AA_eval.py'   
models_to_run = []

for minn, modd, not_o, a100, fullaa, norr, bs in paramss:
  models_to_run.append('{} --model_in {} --mod {} --not-orig {} --a100 {} --full_aa {} --l_norms {} --batch_size {}'
              .format(str_to_run, minn, modd, not_o, a100, fullaa, norr, bs))

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
