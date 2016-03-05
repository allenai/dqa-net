require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson')
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('--image_path_json', '', 'json containing ordered list of image paths')
cmd:option('--cnn_proto', '', 'path to the cnn prototxt')
cmd:option('--cnn_model', '', 'path to the cnn model')
cmd:option('--batch_size', 10, 'batch_size')

cmd:option('--out_path', 'image_rep.h5', 'output path')
cmd:option('--gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('--backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:evaluate()
net=net:cuda()

function loadim(imname)
    local im, im2
    im=image.load(imname)
    im=image.scale(im,224,224)
    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939
    return im2
end

local image_path_json_file = io.open(opt.image_path_json, 'r')
local image_path_json = cjson.decode(image_path_json_file:read())
image_path_json_file.close()

local image_path_list = {}
for i,image_path in pairs(image_path_json) do
    table.insert(image_path_list, image_path)
end

local ndims=4096
local batch_size = opt.batch_size

local sz = #image_path_list
local feat = torch.CudaTensor(sz, ndims)
print(string.format('processing %d images...', sz))
for i = 1, sz, batch_size do
    xlua.progress(i, sz)
    local r = math.min(sz, i + batch_size - 1)
    local ims = torch.CudaTensor(r-i+1, 3, 224, 224)
    for j = 1, r-i+1 do
        ims[j] = loadim(image_path_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat[{{i,r},{}}] = net.modules[43].output:clone()
    collectgarbage()
end

local h5_file = hdf5.open(opt.out_path, 'w')
h5_file:write('/data', feat:float())
h5_file:close()
