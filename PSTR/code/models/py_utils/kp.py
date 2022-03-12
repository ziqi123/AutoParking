import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .matcher import build_matcher
import copy
from sample.vis import save_debug_images_training, save_debug_images_joints,viz_infer_joints
from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion

BN_MOMENTUM = 0.1

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 kps_dim=None,
                 cls_dim=None,
                 mlp_layers=None,
                 norm_layer=FrozenBatchNorm2d):

        super(kp, self).__init__()
        self.flag = flag

        # above all waste not used
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, res_dims[0], layers[0], stride=res_strides[0])
        self.layer2 = self._make_layer(block, res_dims[1], layers[1], stride=res_strides[1])
        self.layer3 = self._make_layer(block, res_dims[2], layers[2], stride=res_strides[2])
        self.layer4 = self._make_layer(block, res_dims[3], layers[3], stride=res_strides[3])

        hidden_dim  = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 100 h
        self.input_proj = nn.Conv2d(res_dims[-1], attn_dim, kernel_size=1)  # the same as channel of self.layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)
        self.class_embed = nn.Linear(hidden_dim, cls_dim)  # 9 subclasses
        self.joints_embed = MLP(hidden_dim, hidden_dim, kps_dim, mlp_layers)  # 5 keypoints 5 * 2

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        """ This performs the PSTR forward computation.
            Parameters:
                 xs: [batch_images, batch_inmasks]
            Outputs: list of dicts. The expected keys in each dict depends on the losses applied.
        """
        B, num_roi, channel, roi_height, roi_width = xs[0].shape
        images = xs[0].view(-1, channel, roi_height, roi_width)
        masks  = xs[1].view(-1, 1, roi_height, roi_width)
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)

        masks_p = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]  # B, 8, 8
        pos = self.position_embedding(p, masks_p)
        hs = self.transformer(self.input_proj(p), masks_p, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_joints = self.joints_embed(hs).sigmoid()

        out = {'pred_boxes': outputs_joints[-1],'pred_classes':outputs_class[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_joints,outputs_class)

        return out

    def _test(self, *xs, **kwargs):
        B, num_roi, channel, roi_height, roi_width = xs[0].shape
        images = xs[0].view(-1, channel, roi_height, roi_width)
        masks  = xs[1].view(-1, 1, roi_height, roi_width)

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)


        masks_p = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]  # B, 8, 8
        pos = self.position_embedding(p, masks_p)
        hs = self.transformer(self.input_proj(p), masks_p, self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_joints = self.joints_embed(hs).sigmoid()
        out = {'pred_boxes': outputs_joints[-1],'pred_classes':outputs_class[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_joints,outputs_class)

        return out

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord,outputs_class):
        return [{'pred_boxes': b,'pred_classes':c} for b,c in zip(outputs_coord[:-1],outputs_class[:-1])]

class AELoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=True,
                 dec_layers=2,
                 bsize=16,
                 ):
        super(AELoss, self).__init__()
        weight_dict = {'loss_ce': 1,
                       'loss_boxes': 3,}
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ['cls','joints']
        self.debug_path = debug_path
        self.matcher = build_matcher(1,1,3,bsize)
        self.criterion = SetCriterion(num_classes=2,
                                      matcher=self.matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=0.1,
                                      losses=losses)
        self.Softmax=nn.Softmax(dim=-1)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets,use_indices=True,threshold=0.7):
        """ This performs the PSTR loss computation.
            Parameters:
                 iteration: num of training iters
                 save: save visualization results or not
                 viz_split: save results from test dataset or valid dataset
                 outputs: [batch_images, batch_inmasks]
                 targets: [batch_images, pslot, pslot_ign]
                 use_indices: use matching results or not
            Outputs: list of dicts. The expected keys in each dict depends on the losses applied.
        """


        set_loss  = 0
        loss_dict, indices = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        set_loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # save intermediate results
        if save:
            which_stack=0
            if use_indices==False:
                #inference
                save_dir = os.path.join(self.debug_path, viz_split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = 'iter_{}_layer_{}'.format(iteration, which_stack)
                save_path = os.path.join(save_dir, save_name)
                with torch.no_grad():

                    # roi_images
                    gt_viz_inputs = targets[0]
                    raw_batch, num_roi, c, roi_h, roi_w = gt_viz_inputs.shape
                    gt_viz_inputs = gt_viz_inputs.view(-1, c, roi_h, roi_w)
                    pred_joints = roi_h * outputs['pred_boxes'].detach()
                    pred_classes = outputs['pred_classes']
                    pred=self.Softmax(pred_classes)
                    mask=pred[:,:,1]>pred[:,:,0]
                    pred_pslots=[joi[mas] for joi,mas in zip(pred_joints,mask)]
                    viz_tgt_joints  = [i.detach()*roi_h for i in targets[1:raw_batch+1]]
                    ign_joints = [i.detach() * roi_h for i in targets[raw_batch + 1:]]

                    viz_infer_joints(gt_viz_inputs,
                                             viz_tgt_joints,
                                             None,
                                             joints_pred=pred_pslots,
                                             ign_joints=ign_joints,
                                             prefix=save_path)
            else:
                #validation
                which_stack = 0
                save_dir = os.path.join(self.debug_path, viz_split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = 'iter_{}_layer_{}'.format(iteration, which_stack)
                save_path = os.path.join(save_dir, save_name)

                with torch.no_grad():
                    gt_viz_inputs = targets[0]
                    raw_batch, num_roi, c, roi_h, roi_w = gt_viz_inputs.shape
                    gt_viz_inputs = gt_viz_inputs.view(-1, c, roi_h, roi_w)
                    pred_joints = roi_h * outputs['pred_boxes'].detach()
                    l_b, l_s, pslot = self._get_src_permutation_idx(indices)

                    cls_b, cls_s, val_pslot = self._make_mask(indices, l_b, l_s, targets, raw_batch, pslot)
                    viz_pred_joints=[]
                    k_idx=0
                    for i in range(raw_batch):
                        if i not in val_pslot:
                            viz_pred_joints.append(torch.tensor([]).cuda())
                        else:
                            viz_pred_joints.append(pred_joints[i][cls_s[k_idx]])
                            k_idx=k_idx+1

                    viz_tgt_joints  = [i.detach()*roi_h for i in targets[1:raw_batch+1]]
                    ign_joints=[i.detach()*roi_h for i in targets[raw_batch+1:]]

                    save_debug_images_joints(gt_viz_inputs,
                                             viz_tgt_joints,
                                             None,
                                             joints_pred=viz_pred_joints,
                                             ign_joints=ign_joints,
                                             prefix=save_path)

        return (set_loss.unsqueeze(0), loss_dict,)

    def _get_src_permutation_idx(self, indices):
        l_bidx = []
        l_sidx = []
        valid_pslot = []
        for i, src in enumerate(indices):
            # print(src)
            # print(src[0])
            if src[0][0] != -1:
                l_bidx.append(torch.full_like(src[0], i))
                l_sidx.append(src[0])
                valid_pslot.append(i)

        return l_bidx, l_sidx, valid_pslot

    def _make_mask(self,indices,l_b,l_s,targets,bs,pslot):
        lv_b=[]
        lv_s=[]
        val_pslot=copy.deepcopy(pslot)

        for id,(i,j) in enumerate(zip(l_b,l_s)):
            if targets[1+pslot[id]].shape[1]-1+targets[1+pslot[id]+bs].shape[1]-1 != i.shape[0]:
                print('contain discrete length between prediction and label')
                raise ValueError

            valid_length=targets[1+pslot[id]].shape[1]-1
            # ignore_length=targets[1+id+bs].shape[1]-1
            sort=indices[pslot[id]][-1]
            mask=sort<valid_length
            if valid_length == 0:
                val_pslot.remove(pslot[id])
                continue
            lv_b.append(i[mask])
            lv_s.append(j[mask])
        return lv_b,lv_s,val_pslot
