"""Refit source colours after learned geometry changes, preserving learned links."""
import torch
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.material_refine import refine_head_material
from SkingToolkit.dense_uv_parser.final_head_uv_revision import tie_material


def refine_final_head_uv(model,result,renderer,views,steps):
    decoder=model.final_head_uv_decoder
    if decoder.revision!=2:raise ValueError('Final material refit requires revision 2 relations')
    original=result['uv'];details=result['details'];relations=result['final_head_uv']
    sources=details['color_source_support'];images=details['rendered']
    valid=torch.zeros_like(sources,dtype=torch.bool)
    y0,y1,x0,x1=head_bounds(*sources.shape[-2:]);valid[:,y0:y1,x0:x1]=sources[:,y0:y1,x0:x1]
    if not valid.any():
        result['final_head_material_refit']={'accepted':False,'reason':'no_confident_head_pixels','alpha_exact':True,'body_exact':True}
        return
    # Visibility is recomputed using FINAL alpha. Pre-edit texel support would
    # freeze newly exposed inner hair to its old, formerly hidden skin colour.
    fitted=refine_head_material(original,images,sources,renderer,views,steps=steps)
    if not torch.equal(original[:,3],fitted[:,3]) or not torch.equal(original[:,:,16:],fitted[:,:,16:]):
        raise RuntimeError('Material fitting changed geometry or body')
    values=fitted.flatten(2)[:,:,decoder.ids].transpose(1,2)
    shared=tie_material(decoder,values[:,:,:3],values[:,:,3],relations['mirror_link_logits'],relations['layer_link_logits'])
    candidate=fitted.clone();candidate.flatten(2)[:,:3,decoder.ids]=shared.transpose(1,2)
    if not torch.isfinite(candidate).all():raise RuntimeError('Non-finite final material fit')
    def error(uv):
        rendered=torch.stack([renderer.forward_view(uv,v) for v in views],1).flatten(0,1)[:,:3]
        return float((((rendered-images[:,:3])**2)*valid[:,None]).sum()/(3*valid.sum()).clamp_min(1))
    before,after=error(original),error(candidate)
    accepted=after<before
    result['uv']=candidate if accepted else original
    result['final_head_material_refit']={'steps':steps,'accepted':accepted,'source_mse_before':before,'source_mse_after_proposed':after,'source_mse_final':after if accepted else before,'confident_source_pixels':int(valid.sum()),'alpha_exact':True,'body_exact':True,'learned_material_links_preserved':True}
