import time
import argparse
import torch

# import torchvision
from resnet import resnet50



from torch.backends import mkldnn
from torch.utils import mkldnn as mkldnn_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=32,
                    help="[option] batch size.")
parser.add_argument('--itr', type=int, default=20,
                    help="[option] max itaration.")
type_list = ["gpu", "cpu_mkl", "cpu_nomkl", "cpu_mkltensor", "a64fx"]
parser.add_argument('--type', default="cpu_mkl", choices=type_list,
                    help='choose ' + ",".join(type_list))
# DEBUG option
parser.add_argument('--trace', action='store_true',
                    help='[option] get autograd.profiler')


def main():
  args = parser.parse_args()
  print(">> script option:",args)

  main_worker(args)

def main_worker(args):
  # set device, envflag
  if args.type == "gpu":
    device = torch.device('cuda') # forCUDA
  if args.type in ["cpu_mkl", "cpu_mkltensor", "a64fx"]:
    device = torch.device('cpu')
  if args.type == "cpu_nomkl":
    mkldnn.enabled = False
    device = torch.device('cpu')

  ## set net, images
  #net = torchvision.models.resnet50(pretrained=True)
  net = resnet50()
  net = net.to(device)
  net.eval()

  batch_size = args.batch
  images = torch.randn(batch_size, 3, 224, 224).to(device)

  if args.type == "cpu_mkltensor":
    mkldnn_utils.to_mkldnn(net)
    images = images.to_mkldnn()

  torch.manual_seed(1)               # forPerformance

  # start test
  if args.trace:
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
      laps = run_eval(args, net, images)
    prof.export_chrome_trace("./trace.json")
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
  else:
    laps = run_eval(args, net, images)

  output_statistics(laps)

def run_eval(args, net, images):
  ## test
  print("## Start Test")

  start = time.time()
  laps = []
  with torch.no_grad():
    for i in range(args.itr):
      outputs = net(images)
      if args.type == "gpu":
        torch.cuda.synchronize()

      with torch.autograd.profiler.record_function("measure-print"): # label the block 
        end = time.time()
        print('[%5d] time: %0.3f s' %
          (i+1, (end-start)))
        laps.append(end-start)
        start = time.time()

  return laps

def output_statistics(laps):
  try:
    import pandas as pd
  except:
    print("please install pandas.")
  else:
    ds = pd.Series(laps[1:])
    print(ds.describe())

if __name__ == '__main__':
  main()

