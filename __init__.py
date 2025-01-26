from .refdrop import RefDropSave,RefDropUse,RefDropCombined,RefDropCombinedAdvanced,RefDropUseAdvanced

NODE_CLASS_MAPPINGS = {
    "RefDrop Combined" : RefDropCombined,
    "RefDrop Save" : RefDropSave,
    "RefDrop Use" : RefDropUse,
    "RefDrop Combined Advanced" : RefDropCombinedAdvanced,
    "RefDrop Use Advanced" : RefDropUseAdvanced,
}

__all__ = ['NODE_CLASS_MAPPINGS']
