import os, sys, cv2
import numpy as np

if len(sys.argv) < 4:
  print("prototxt caffemodel imgpath sz_w,sz_h gpu")
  exit(0)

prototxt=sys.argv[1]
caffemodel=sys.argv[2]

imgpath=sys.argv[3]

sz_h=-1;
sz_w=-1;

if len(sys.argv) > 4:
  szs=sys.argv[4].split(',')
  sz_h=int(szs[1])
  sz_w=int(szs[0])
  
outname = sys.argv[5]

sys.path.insert(0,'/home/leejohnnie/code/chuanqi305/ssd/python')
import caffe

caffe.set_mode_cpu()

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
np.set_printoptions(threshold=np.inf)
os.environ['GLOG_minloglevel'] = '2'

if True:
  oriImg=cv2.imread(imgpath).astype(np.float32)
  #print(oriImg[0,0,0], oriImg[0,0,1], oriImg[0,0,2])

  tmp = oriImg[:,:,0].copy();
  oriImg[:,:,0] = oriImg[:,:,2]
  oriImg[:,:,2] = tmp
  #print(oriImg[0,0,0], oriImg[0,0,1], oriImg[0,0,2])

  heigh = oriImg.shape[0]
  width = oriImg.shape[1]

  if sz_w<=0 or sz_h <= 0:
    sz_w = width
    sz_h = heigh

  #oriImg=cv2.resize(oriImg, (sz_w, sz_h))
  oriImg=(oriImg-127.5)/127.5;

  im_blob = np.zeros((1, sz_h, sz_w, 3),
                         dtype=np.float32)
  im_blob[0, 0:sz_h, 0:sz_w, :] = oriImg
  channel_swap = (0, 3, 1, 2)
  im_blob = im_blob.transpose(channel_swap)
  
  net.blobs['data'].reshape(*(im_blob.shape))

  forward_kwargs = {'data': im_blob.astype(np.float32, copy=False)}

  res = net.forward(**forward_kwargs)
  
  if net.blobs.has_key(outname):
	outblob = net.blobs[outname]
	print outblob.data.shape
	print outblob.data.ravel()
	#for d in outblob.data:
	#	print d[:]
	print outblob.data.shape
  else:
	print("blobname not exist " + outname)
  
  #print(net.params['mobilenet_v2_layer4_bottlenect3_s1_conv2'][0].data.shape)
  #print net.params['mobilenet_v2_layer1_conv2d'][0].data.shape
