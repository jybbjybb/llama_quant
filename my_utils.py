import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc


def activation_quant(x, nbit=8, q_group_size=-1):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    maxx = 2**(nbit-1)-1
    
    if nbit > 0:
        org_x_shape = x.shape
        if q_group_size > 0:
            assert org_x_shape[-1] % q_group_size == 0
            x = x.reshape(-1, q_group_size)

        scale =  x.abs().max(dim=-1, keepdim=True).values.clamp_(min=2e-3)/(maxx-1.0)
        y = (x / scale).round().clamp_(-maxx+0.0, maxx-1.0) * scale
        #scale = (maxx-1.0) / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=2e-3)
        #y = (x * scale).round().clamp_(-maxx+0.0, maxx-1.0) / scale
    else:
        y = x
      
    if nbit > 0 and q_group_size > 0:
        y = y.reshape(org_x_shape)
    return y

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, get_scale_zp=False, special_channels=None,
):
    if n_bit < 0:
        return w
    org_w_shape = w.shape
        
    if special_channels is not None:
        T1 = torch.zeros_like(w)
        T1[:,special_channels] = w[:,special_channels]
        w -= T1
        
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        if q_group_size == 0:
            max_val = w.amax()
            min_val = w.amin()
        else:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
   
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    
    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
   
    if special_channels is not None:
        w[:,special_channels] = T1[:,special_channels]

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, x_nbit=8, w_nbit=8, out_nbit=-1, q_group_size=-1, T=-1.0, in_place=False,train_scaleW=False, loss_thr=1e6, name="QLinear"):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.x_nbit = x_nbit
        self.w_nbit = w_nbit
        self.q_group_size = q_group_size
        self.out_nbit = out_nbit

        ############################
        self.in_place = in_place # Inplace will change the FP weight values during inference, use with Caution
        # When in_place is enabled during train_scaleW, make sure to set w_nbit and x_nbit 
        # in train_smooth_factor(X=w, Y=x, nbit_x=self.w_nbit,  nbit_y=self.x_nbit)
        self.is_weight_modified = False
       
        ######## ########
        # for bi-smooth algorithms
        ######## ########
        # The logic is:
        # if self.T > 0, we go to compute scale_W and do bi-smoothing; else, we do not do any bi-smoothing
        # if self.train_scaleW is True, then we will train scale_W if default loss > loss threshold 
        self.T = T
        self.scale_W = None
        self.train_scaleW = train_scaleW
        self.loss_thr = loss_thr
        self.loss_function = 2
        self.lr = 0.001
        self.n_epoch = 1000
        self.name = name
        

    def forward(self, x):
        w = self.weight

        if self.T > 0:
            #y = F.linear(x, w)
            if self.scale_W is None or self.T == 314: # if self.T=314, calculate scale-W for each batch.
               
                # Step 1: Calculate max absolute values along the last dimension
                # x shape [batch, seq_len, hid_dim]
                max_X = torch.max(torch.abs(x), dim=1)[0]  # Shape [batch, hidden_dim]
                max_X = torch.median(max_X, dim=0)[0]  # Shape [B]

                max_W = torch.max(torch.abs(w), dim=0)[0]  # Shape [B]

                self.scale_W = torch.sqrt(max_X.to(max_W) / (max_W + 1e-8))
                
                self.scale_W = self.scale_W.to(self.weight.dtype)
                # apply some extra supress of W
                self.scale_W /= self.T

                #print("scale_W shape:", scale_W.shape)
                self.scale_W = self.scale_W.unsqueeze(0)

            # Normalize W and x
            if not self.is_weight_modified:
                w = w * self.scale_W
            x = x / self.scale_W.unsqueeze(0)
            
        if self.w_nbit > 0:
            if not self.in_place:
                w = w + (pseudo_quantize_tensor(w, self.w_nbit, q_group_size=self.q_group_size ) - w).detach()
            elif self.is_weight_modified is False:
                w = pseudo_quantize_tensor(w, self.w_nbit, q_group_size=self.q_group_size )
                self.is_weight_modified = True 
                self.weight.data = w.data

        x = activation_quant(x, self.x_nbit,q_group_size=self.q_group_size)
        output = F.linear(x, w, self.bias)
        if self.out_nbit > 0:
            output = activation_quant(output, self.out_nbit, q_group_size=self.q_group_size)
        return output

    def __repr__(self):
        return (f"QLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, x_nbit={self.x_nbit}, w_nbit={self.w_nbit}, "
                f"out_nbit={self.out_nbit}, q_group_size={self.q_group_size}, T={self.T} )"
                )

def save_scale_w(model, file_path):
    if file_path is None:
        print("No save path provided. Skip saving scaleW.")
        return False
    scale_w_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, QLinear) and layer.scale_W is not None:
            scale_w_dict[name] = layer.scale_W
    torch.save(scale_w_dict, file_path)
    print("\nScaleW file {} saved!\n".format(file_path))
    return True

def load_scale_w(model, file_path):
    if file_path is None:
        print("No load path provided. Skip loading scaleW.")
        return
    import os
    if os.path.exists(file_path):
        scale_w_dict = torch.load(file_path)
        for name, layer in model.named_modules():
            if isinstance(layer, QLinear) and name in scale_w_dict:
                layer.scale_W = scale_w_dict[name]
        print("\nScaleW file {} loaded!\n".format(file_path))
        return True
    else:
        print(f"Warning: The file '{file_path}' does not exist. Scale weights not loaded.")
        return False


def replace_selective_linear_layers_recursive(module, prefix='', args=None):
    with torch.no_grad():  # Disable gradient tracking
        # Iterate over all modules in the model (recursive)
        for name, child in list(module.named_children()):

            child_full_name = prefix + '.' + name if prefix else name

            # Recursively check nested modules for Linear layers
            replace_selective_linear_layers_recursive(child, child_full_name, args)
            

            if isinstance(child, nn.Linear):
                # Replace Linear with QLinear, keeping the same in_features, out_features, and bias
                in_features = child.in_features
                out_features = child.out_features
                bias = child.bias is not None

                weight = child.weight.data
                max_W = weight.abs().max()
                if 'k_proj' in name or 'v_proj' in name:
                    print(f"Replace Option 1 for  {prefix + '.' + name if prefix else name}")
                    qlinear_layer = QLinear(in_features, out_features, bias, x_nbit=args.x_nbit, w_nbit=args.w_nbit, out_nbit=args.out_nbit, q_group_size=args.q_group_size, in_place=args.in_place_w, T=args.T, train_scaleW=args.train_scaleW, loss_thr=args.loss_thr, name=child_full_name)
                else:
                    print(f"Replace Option 2 (default) for  {prefix + '.' + name if prefix else name}")
                    qlinear_layer = QLinear(in_features, out_features, bias, x_nbit=args.x_nbit, w_nbit=args.w_nbit, q_group_size=-1, in_place=args.in_place_w, T=1)

                # Copy weights and bias from the Linear layer to the new QLinear layer
                qlinear_layer.weight.data.copy_(child.weight.data)
                if bias:
                    qlinear_layer.bias.data.copy_(child.bias.data)

                # Ensure the new QLinear layer has the same dtype and device as the original Linear layer
                qlinear_layer = qlinear_layer.to(child.weight.device, child.weight.dtype)

                # Find the parent module and replace the child
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                parent_module = module.get_submodule(parent_name) if parent_name else module
                parent_module._modules[name.split('.')[-1]] = qlinear_layer

                # Clear the memory by removing references to the old layer and calling garbage collection
                del child  # Explicitly remove reference to old Linear layer
                gc.collect()  # Trigger garbage collection to free memory

                print(f"Memory cleared and layers replaced for {child_full_name} ...")
            else:
                print(f"Skipped for {child_full_name} ...")
    # Clear the GPU cache after all modifications are done
    torch.cuda.empty_cache()
    #print(f"Memory cleared and layers replaced for {prefix} ...")


def replace_all_linear_layers_recursive(module, prefix='', args=None):
    with torch.no_grad():  # Disable gradient tracking
        # Iterate over all modules in the model (recursive)
        for name, child in list(module.named_children()):

            #print("Processing {}...".format(prefix + '.' + name if prefix else name))
            child_full_name = prefix + '.' + name if prefix else name
            # Recursively check nested modules for Linear layers
            replace_all_linear_layers_recursive(child, child_full_name, args)
            

            if isinstance(child, nn.Linear):
                # Replace Linear with QLinear, keeping the same in_features, out_features, and bias
                in_features = child.in_features
                out_features = child.out_features
                bias = child.bias is not None

                # Create the QLinear layer with the same dimensions and bias
                qlinear_layer = QLinear(in_features, out_features, bias, x_nbit=args.x_nbit, w_nbit=args.w_nbit, out_nbit=args.out_nbit, q_group_size=args.q_group_size, in_place=args.in_place_w, T=args.T, train_scaleW=args.train_scaleW, loss_thr=args.loss_thr, name=child_full_name)

                # Copy weights and bias from the Linear layer to the new QLinear layer
                qlinear_layer.weight.data.copy_(child.weight.data)
                if bias:
                    qlinear_layer.bias.data.copy_(child.bias.data)

                # Ensure the new QLinear layer has the same dtype and device as the original Linear layer
                qlinear_layer = qlinear_layer.to(child.weight.device, child.weight.dtype)

                # Find the parent module and replace the child
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                parent_module = module.get_submodule(parent_name) if parent_name else module
                parent_module._modules[name.split('.')[-1]] = qlinear_layer

                # Clear the memory by removing references to the old layer and calling garbage collection
                del child  # Explicitly remove reference to old Linear layer
                gc.collect()  # Trigger garbage collection to free memory

                print(f"Memory cleared and layers replaced for {child_full_name} ...")
            else:
                print(f"Skipped for {child_full_name} ...")
    # Clear the GPU cache after all modifications are done
    torch.cuda.empty_cache()
    #print(f"Memory cleared and layers replaced for {prefix} ...")




if __name__ == "__main__":
    shape = (100,100)
    x = torch.randn(shape)
    noise_level = [0.5, 1.0]
    y = x.clone()
    w = inject_noise(x, noise_level, noise_type='uniform_multiplicative')
    #print(x)
    z = w / y
    z = z.cpu().numpy().flatten()
    print(z[:100])
    #hist_values, hist_edges = np.histogram(z, bins='auto')
    import matplotlib.pyplot as plt

    # Plot the histogram
    plt.hist(z, bins='auto', alpha=0.7, edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    #plt.yscale('log')
    plt.title('Histogram of Sample Data')
    plt.grid(True)
    plt.show()
