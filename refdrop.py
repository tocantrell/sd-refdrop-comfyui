from comfy.ldm.modules.attention import CrossAttention, BasicTransformerBlock, optimized_attention, optimized_attention_masked, default

import torch
import glob


CrossAttention.k_dict = {}
CrossAttention.v_dict = {}
CrossAttention.layer_index = 0
CrossAttention.current_step = 0
CrossAttention.rfg = 0.
CrossAttention.layer_name = 'input'
CrossAttention.to_disk = False
CrossAttention.custom_input = []
CrossAttention.custom_middle = []
CrossAttention.custom_output = []
CrossAttention.activation_limit = 0

def remove_latent_files():
    try:
        del CrossAttention.k_dict
    except:
        pass
    try:
        del CrossAttention.v_dict
    except:
        pass
    CrossAttention.k_dict = {}
    CrossAttention.v_dict = {}
    CrossAttention.layer_index = 0
    CrossAttention.current_step = 0

def custom_layer_removal(custom_input, custom_middle, custom_output, current_layer):
    use_layer = True
    _,current_layer = current_layer.split('_',1)
    if (current_layer in custom_input)|(current_layer in custom_middle)|(current_layer in custom_output):
        use_layer = False
    return use_layer

def run_refdrop():
    CrossAttention.layer_index = 0
    CrossAttention.current_step = 0
    def forward_crossattention(
                    self,
                    x,
                    context=None,
                    value=None,
                    mask=None,
                    transformer_options=None,
                    k_refdrop=None,
                    v_refdrop=None,
                    refdrop_save=False,
                    refdrop_use=False,
                    k_file=None,
                    v_file=None
                ):

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)


        if refdrop_save:
            if CrossAttention.to_disk: #Not enabled for ComfyUI
                #Save K and V to files on disk
                torch.save(k, k_file)
                torch.save(v, v_file)
            else:
                #Save K and V to files to memory via a dictionary
                CrossAttention.k_dict.update({k_file:k.to('cpu')})
                CrossAttention.v_dict.update({v_file:v.to('cpu')}) 

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
            if refdrop_use:
                out_refdrop = optimized_attention(q,
                                                  k_refdrop,
                                                  v_refdrop,
                                                  self.heads)
                                                  #attn_precision=self.attn_precision)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)
            if refdrop_use: 
                out_refdrop = optimized_attention_masked(q, k_refdrop, v_refdrop, self.heads)#, mask, attn_precision=self.attn_precision)

        if refdrop_use:
            out = (out * (1-CrossAttention.rfg)) + (out_refdrop * CrossAttention.rfg)

        
        return self.to_out(out)

    CrossAttention.forward = forward_crossattention
    
    def forwardBasicTransformerBlock(self, x, context=None, transformer_options={}):
        # Stolen from ComfyUI with some modifications
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        if CrossAttention.layer_name != block[0]:
            CrossAttention.layer_name = block[0]
            CrossAttention.layer_index = 0

        #Define file save or read location
        latentname = 'step'+ str(CrossAttention.current_step)+'_'+CrossAttention.layer_name +'_'+ str(CrossAttention.layer_index)
        if CrossAttention.to_disk:
            k_file = current_extension_directory + '/latents/k/' + latentname + '.pt'
            v_file = current_extension_directory + '/latents/v/' + latentname + '.pt'
        else:
            k_file = latentname
            v_file = latentname
        refdrop_save = False
        refdrop_use = False

        
        #Allows for running all or only some layer types
        #"to_disk" not currently implemented for ComfyUI. Only saves to system memory.
        if (CrossAttention.refdrop == 'Use')&(CrossAttention.to_disk)&(CrossAttention.layer_name in CrossAttention.layer_refdrop):
            try: 
                v_refdrop = torch.load(v_file, weights_only=True).to('cuda')
                k_refdrop = torch.load(k_file, weights_only=True).to('cuda')
                refdrop_use = True
            except:
                #Running without the last few K and V files will not significantly change the results.
                #Also allows for variable hires fix and adetailer without initial save
                print('Continuing without RefDrop from this point.')
                CrossAttention.refdrop = 'Done'
                v_refdrop = None
                k_refdrop = None
        elif (CrossAttention.refdrop == 'Use')&(CrossAttention.to_disk!=True)&(CrossAttention.layer_name in CrossAttention.layer_refdrop):
            try:
                v_refdrop = CrossAttention.v_dict[v_file].to('cuda')
                k_refdrop = CrossAttention.k_dict[k_file].to('cuda')
                refdrop_use = True
            except:
                #Running without the last few K and V files will not significantly change the results.
                #Also allows for variable hires fix and adetailer without initial save
                print('Continuing without RefDrop from this point.')
                CrossAttention.refdrop = 'Done'
                v_refdrop = None
                k_refdrop = None
        elif (CrossAttention.refdrop == 'Save')&(CrossAttention.activation_limit==0):
            v_refdrop = None
            k_refdrop = None
            refdrop_save = True
        elif (CrossAttention.refdrop == 'Save')&(CrossAttention.activation_limit>=CrossAttention.current_step):
            v_refdrop = None
            k_refdrop = None
            refdrop_save = True
        else:
            v_refdrop = None
            k_refdrop = None
            refdrop_use = False
            refdrop_save = False
        
        #Custom layer removal list parsing
        if (refdrop_use) & ((len(CrossAttention.custom_input) > 0)|(len(CrossAttention.custom_middle) > 0)|(len(CrossAttention.custom_output) > 0)):
            if not custom_layer_removal(CrossAttention.custom_input,CrossAttention.custom_middle,CrossAttention.custom_output,latentname):
                refdrop_use = False
                v_refdrop = None
                k_refdrop = None

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]
        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip
        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None
        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)
        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block
        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)

        else:
            #Apply RefDrop if the current layer is in the selected list
            if CrossAttention.layer_name in CrossAttention.layer_refdrop: 
                n = self.attn1(
                        n,
                        context=context_attn1,
                        value=value_attn1,
                        transformer_options=extra_options,
                        k_refdrop=k_refdrop,
                        v_refdrop=v_refdrop,
                        refdrop_save=refdrop_save,
                        refdrop_use=refdrop_use,
                        k_file=k_file,
                        v_file=v_file
                    )
            else:
                n = self.attn1(
                        n,
                        context=context_attn1,
                        value=value_attn1,
                        transformer_options=extra_options
                    )

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)
        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)
        if self.attn2 is not None:
            n = self.norm2(x)
            context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)
            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block
            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=extra_options)
        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)
        x += n
        x_skip = 0
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        CrossAttention.layer_index += 1
        CrossAttention.current_step += 1

        return x
    
    BasicTransformerBlock.forward = forwardBasicTransformerBlock

class RefDropSave:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    CATEGORY = "RefDrop"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", {
                        "tooltip": "Place between the latent generator and the KSampler."}),
                    "save_enabled": (["TRUE","FALSE"], {
                        "default": "FALSE",
                        "display": "Save RefDrop Data",
                        "tooltip": "Whether to enable saving RefDrop activation data to memory.",
                        "lazy": True}),
                    "activation_limit": ("INT", {
                    "default": 0,
                    "tooltip": "Maximum number of activations to save in memory. Limiting the number that are saved can help with system instability. Set to 0 for unlimited.",
                        "lazy": True}),
                    "delete_existing": (["TRUE","FALSE"], {"default": "FALSE", "tooltip": "Whether to remove previous run's save from memory. Will always happen regardless of selection here if save_enabled.",
                        "lazy": True}),
                }
            }
    FUNCTION = "save_refdrop"
    RETURN_TYPES = ("LATENT",)

    def save_refdrop(self, save_enabled,activation_limit,delete_existing,latent):
        
        CrossAttention.to_disk = False#Saving to disk not currently implemented

        if delete_existing=="TRUE":
            remove_latent_files()
            
        if save_enabled=="TRUE":
            remove_latent_files()
            CrossAttention.refdrop = 'Save'
            CrossAttention.layer_refdrop = ["input","middle","output"]
            CrossAttention.activation_limit = activation_limit
            run_refdrop()
            
        else:
            CrossAttention.refdrop = "Disabled"

        return (latent,)
            
class RefDropUseAdvanced:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    CATEGORY = "RefDrop"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", {
                        "tooltip": "Place between the latent generator and the KSampler."}),
                    "use_enabled": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "display": "Use RefDrop Data",
                        "tooltip": "Whether to enable using RefDrop activation stored in memory. Must run Save RefDrop first.",
                        "lazy": True}),
                    "rfg": ("FLOAT", {
                        "default": 0.,
                        "min": -1.,
                        "max": 1.,
                        "step": 0.01,
                        "display": "RFG",
                        "tooltip": "Whether to remove previous run's save from memory. Will always happen regardless of selection here if save_enabled.",
                        "lazy": True}),
                    "disable_all_layer_input": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Input' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_all_layer_middle": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Middle' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_all_layer_output": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Output' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_custom_input_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which input layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": True}),
                    "disable_custom_middle_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which middle layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": False}),
                    "disable_custom_output_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which output layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": True})
                }
            }
    FUNCTION = "use_refdrop"
    RETURN_TYPES = ("LATENT",)

    def use_refdrop(self,
                    use_enabled,
                    rfg,
                    disable_all_layer_input,
                    disable_all_layer_middle,
                    disable_all_layer_output,
                    disable_custom_input_layer_list,
                    disable_custom_middle_layer_list,
                    disable_custom_output_layer_list,
                    latent):

        CrossAttention.to_disk = False#Saving to disk not currently implemented
        
        #Prepare lists of block layers to disable during Use
        layer_list = ["input","middle","output"]
        if disable_all_layer_input=="FALSE":
            layer_input = True
        else:
            layer_input = False
        if disable_all_layer_middle=="FALSE":
            layer_middle = True
        else:
            layer_middle = False
        if disable_all_layer_output=="FALSE":
            layer_output = True
        else:
            layer_output = False
        CrossAttention.layer_refdrop = [x for x in layer_list if [layer_input,layer_middle,layer_output][layer_list.index(x)]]
        
        #Prepare custom lists of layers to disable during use
        CrossAttention.custom_input = []
        CrossAttention.custom_middle = []
        CrossAttention.custom_output = []
        if len(disable_custom_input_layer_list) > 0:
            CrossAttention.custom_input = disable_custom_input_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_input = ["input_"+x for x in CrossAttention.custom_input]
        if len(disable_custom_middle_layer_list) > 0:
            CrossAttention.custom_middle = disable_custom_middle_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_middle = ["middle_"+x for x in CrossAttention.custom_middle]
        if len(disable_custom_output_layer_list) > 0:
            CrossAttention.custom_output = disable_custom_output_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_output = ["output_"+x for x in CrossAttention.custom_output]

        if use_enabled=="TRUE":
            CrossAttention.refdrop = "Use"
            CrossAttention.rfg = rfg
            CrossAttention.layer_index = 0
            CrossAttention.current_step = 0
            CrossAttention.layer_name = "input"
            run_refdrop()
            
        else:
            CrossAttention.refdrop = "Disabled"

        return (latent,)

class RefDropUse:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    CATEGORY = "RefDrop"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", {
                        "tooltip": "Place between the latent generator and the KSampler."}),
                    "use_enabled": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "display": "Use RefDrop Data",
                        "tooltip": "Whether to enable using RefDrop activation stored in memory. Must run Save RefDrop first.",
                        "lazy": True}),
                    "rfg": ("FLOAT", {
                        "default": 0.,
                        "min": -1.,
                        "max": 1.,
                        "step": 0.01,
                        "display": "RFG",
                        "tooltip": "Reference Feature Guidance. Positive values increase consistency and negative values increase diversity.",
                        "lazy": True})
                }
            }
    FUNCTION = "use_refdrop"
    RETURN_TYPES = ("LATENT",)

    def use_refdrop(self, use_enabled, rfg,latent):

        CrossAttention.to_disk = False#Saving to disk not currently implemented
        
        CrossAttention.layer_refdrop = ["input","middle","output"]

        if use_enabled=="TRUE":
            CrossAttention.refdrop = 'Use'
            CrossAttention.rfg = rfg
            CrossAttention.layer_index = 0
            CrossAttention.current_step = 0
            run_refdrop()
            
        else:
            CrossAttention.refdrop = "Disabled"

        return (latent,)

class RefDropCombined:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    CATEGORY = "RefDrop"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", {
                        "tooltip": "Place between the latent generator and the KSampler."}),
                    "refdrop": (["SAVE","USE","DISABLED"], {
                        "default": "SAVE", 
                        "tooltip": "Whether to enable saving or using activations for RefDrop. Must run SAVE before USE first.",
                        "lazy": True}),
                    "rfg": ("FLOAT", {
                        "default": 0.,
                        "min": -1.,
                        "max": 1.,
                        "step": 0.01,
                        "display": "RFG",
                        "tooltip": "Reference Feature Guidance. Positive values increase consistency and negative values increase diversity.",
                        "lazy": True}),
                    "activation_limit": ("INT", {
                    "default": 0,
                    "tooltip": "Maximum number of activations to save in memory. Limiting the number that are saved can help with system instability. Set to 0 for unlimited."}),
                    "delete_existing": (["TRUE","FALSE"], {"default": "FALSE", "tooltip": "Whether to remove previous run's save from memory. Only set to TRUE when in DISABLED model. Previous activation data will always be deleted regardless of selection here if in SAVE mode.",
                        "lazy": True})
                }
            }
    FUNCTION = "run_refdrop"
    RETURN_TYPES = ("LATENT",)

    def run_refdrop(self, refdrop, rfg, activation_limit, delete_existing,latent):

        CrossAttention.to_disk = False#Saving to disk not currently implemented
        
        CrossAttention.layer_refdrop = ["input","middle","output"]

        if delete_existing=="TRUE":
            remove_latent_files()

        if refdrop=="SAVE":
            remove_latent_files()
            CrossAttention.refdrop = 'Save'
            CrossAttention.layer_refdrop = ["input","middle","output"]
            CrossAttention.activation_limit = activation_limit
            run_refdrop()

        elif refdrop=="USE":
            CrossAttention.refdrop = 'Use'
            CrossAttention.rfg = rfg
            CrossAttention.layer_index = 0
            CrossAttention.current_step = 0
            CrossAttention.layer_name = 'input'
            run_refdrop()
            
        else:
            CrossAttention.refdrop = "Disabled"

        return (latent,)

class RefDropCombinedAdvanced:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    CATEGORY = "RefDrop"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", {
                        "tooltip": "Place between the latent generator and the KSampler."}),
                    "refdrop": (["SAVE","USE","DISABLED"], {
                        "default": "SAVE", 
                        "tooltip": "Whether to enable saving or using activations for RefDrop. Must run SAVE before USE first.",
                        "lazy": False}),
                    "rfg": ("FLOAT", {
                        "default": 0.,
                        "min": -1.,
                        "max": 1.,
                        "step": 0.01,
                        "display": "RFG",
                        "tooltip": "Reference Feature Guidance. Positive values increase consistency and negative values increase diversity.",
                        "lazy": True}),
                    "activation_limit": ("INT", {
                    "default": 0,
                    "tooltip": "Maximum number of activations to save in memory. Limiting the number that are saved can help with system instability. Set to 0 for unlimited."}),
                    "delete_existing": (["TRUE","FALSE"], {"default": "FALSE", "tooltip": "Whether to remove previous run's save from memory. Only set to TRUE when in DISABLED model. Previous activation data will always be deleted regardless of selection here if in SAVE mode.",
                        "lazy": True}),
                    "disable_all_layer_input": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Input' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_all_layer_middle": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Middle' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_all_layer_output": (["TRUE","FALSE"], {
                        "default": "FALSE", 
                        "tooltip": "'True' will disable using all of the 'Output' layers of the attention U-net.",
                        "lazy": True}),
                    "disable_custom_input_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which input layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": True}),
                    "disable_custom_middle_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which middle layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": True}),
                    "disable_custom_output_layer_list": ("STRING", {
                        "multiline": False, 
                        "default": "",
                        "tooltip": "A list of numbers for which output layers of the attention U-net to not run. Formatted like '0,1,2'. Will do nothing if selected 'True' above.",
                        "lazy": True})
                }
            }
    FUNCTION = "run_refdrop"
    RETURN_TYPES = ("LATENT",)

    def run_refdrop(self,
                    refdrop,
                    rfg,
                    activation_limit,
                    delete_existing,
                    disable_all_layer_input,
                    disable_all_layer_middle,
                    disable_all_layer_output,
                    disable_custom_input_layer_list,
                    disable_custom_middle_layer_list,
                    disable_custom_output_layer_list,
                    latent):

        CrossAttention.to_disk = False#Saving to disk not currently implemented for ComfyUI
        
        #Prepare lists of block layers to disable during Use
        layer_list = ["input","middle","output"]
        if disable_all_layer_input=="FALSE":
            layer_input = True
        else:
            layer_input = False
        if disable_all_layer_middle=="FALSE":
            layer_middle = True
        else:
            layer_middle = False
        if disable_all_layer_output=="FALSE":
            layer_output = True
        else:
            layer_output = False
        CrossAttention.layer_refdrop = [x for x in layer_list if [layer_input,layer_middle,layer_output][layer_list.index(x)]]
        
        #Prepare custom lists of layers to disable during use
        CrossAttention.custom_input = []
        CrossAttention.custom_middle = []
        CrossAttention.custom_output = []
        if len(disable_custom_input_layer_list) > 0:
            CrossAttention.custom_input = disable_custom_input_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_input = ["input_"+x for x in CrossAttention.custom_input]
        if len(disable_custom_middle_layer_list) > 0:
            CrossAttention.custom_middle = disable_custom_middle_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_middle = ["middle_"+x for x in CrossAttention.custom_middle]
        if len(disable_custom_output_layer_list) > 0:
            CrossAttention.custom_output = disable_custom_output_layer_list.replace(" ", "").split(",")
            CrossAttention.custom_output = ["output_"+x for x in CrossAttention.custom_output]

        if delete_existing=="TRUE":
            remove_latent_files()

        if refdrop=="SAVE":
            remove_latent_files()
            CrossAttention.refdrop = "Save"
            CrossAttention.layer_refdrop = ["input","middle","output"]
            CrossAttention.activation_limit = activation_limit
            run_refdrop()

        elif refdrop=="USE":
            CrossAttention.refdrop = "Use"
            CrossAttention.rfg = rfg
            CrossAttention.layer_index = 0
            CrossAttention.current_step = 0
            CrossAttention.layer_name = "input"
            run_refdrop()
            
        else:
            CrossAttention.refdrop = "Disabled"

        return (latent,)
